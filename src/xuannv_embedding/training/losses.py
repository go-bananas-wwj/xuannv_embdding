from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# 重建损失 + batch uniformity 损失模块。


def _center_crop_tensor(x: torch.Tensor, crop_size: int | None) -> torch.Tensor:
    """Center-crop the last two spatial dimensions when requested."""
    if crop_size is None or x.dim() < 2:
        return x
    crop_size = int(crop_size)
    if crop_size <= 0:
        return x
    height, width = x.shape[-2:]
    if height == crop_size and width == crop_size:
        return x
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"无法中心裁剪到 {crop_size}: 输入空间尺寸为 {(height, width)}"
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return x[..., top : top + crop_size, left : left + crop_size]


def _center_crop_dict(
    values: dict[str, torch.Tensor],
    crop_size: int | None,
) -> dict[str, torch.Tensor]:
    return {name: _center_crop_tensor(value, crop_size) for name, value in values.items()}


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算带掩码的重建损失。

    支持非月度输入 ``[B, C, H, W]`` 与月度输入 ``[B, T, C, H, W]``（或 CE 的
    ``[B, T, H, W]`` target）。对于月度输入，时间维度会与 batch 维度合并后计算。

    Args:
        pred: 预测值。L1 时为 ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``；
            CE 时为 logits ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``。
        target: 目标值。L1 时为 ``[B, C, H, W]`` 或 ``[B, T, C, H, W]``；
            CE 时为类别索引 ``[B, H, W]`` 或 ``[B, T, H, W]`` (int64)。
        mask: 空间有效掩码，形状可为 ``[H, W]``、``[B, H, W]``、``[B, 1, H, W]``、
            ``[B, T, H, W]`` 或 ``[B, T, 1, H, W]``。
        loss_type: ``"l1"`` 或 ``"ce"``。
        eps: 防止除零的小常数。

    Returns:
        标量张量，表示掩码平均后的损失。
    """
    is_temporal = pred.dim() == 5

    if is_temporal:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        if target.dim() == 5:
            target = target.reshape(B * T, C, H, W)
        elif target.dim() == 4:
            # CE target: (B, T, H, W)
            target = target.reshape(B * T, H, W)
        if mask.dim() == 5:
            mask = mask.reshape(B * T, *mask.shape[2:])
        elif mask.dim() == 4 and mask.shape[1] == T:
            mask = mask.reshape(B * T, *mask.shape[2:])
        elif mask.dim() == 3:
            if mask.shape == (B, T, 1):
                mask = mask.reshape(B * T, 1, 1).expand(B * T, H, W)
            else:
                # (B, T, H, W) 已在 reshape 分支处理，其它形状保留供 expand_as 处理。
                pass
        elif mask.dim() == 2:
            # (B, T) 时间掩码，应用到所有空间位置。
            mask = mask.reshape(B * T, 1, 1).expand(B * T, H, W)

    if loss_type == "l1":
        # 逐元素 L1，然后在通道维度取平均，得到 [B, H, W]。
        loss = F.l1_loss(pred, target, reduction="none").mean(dim=1)
    elif loss_type == "ce":
        # cross_entropy 输出 [B, H, W]；以 0 作为 nodata/背景类别，不参与损失。
        loss = F.cross_entropy(pred, target, ignore_index=0, reduction="none")
        # 即使外部 mask 未显式屏蔽 class-0 像素，也确保其不进入平均 denominator。
        mask = mask * (target != 0).float()
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type!r}，仅支持 'l1' 或 'ce'")

    # 将 mask 广播到 [B, H, W] 后应用。
    mask = mask.expand_as(loss)
    masked_sum = (loss * mask).sum()
    masked_count = mask.sum()
    return masked_sum / (masked_count + eps)


def batch_uniformity_loss(emb: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """计算 batch 内场景级嵌入的均匀性损失。

    先将每个嵌入 L2 归一化到单位球面，再计算 Wang-Isola 风格的
    ``log(mean(exp(-temperature * pairwise_squared_distance)))``。该值在嵌入
    更分散时更小，因此可以用正权重直接加到总损失里进行最小化。

    Args:
        emb: 场景级嵌入，形状 ``[B, D]`` 或月度 ``[B, T, D]``。
        temperature: 距离温度，值越大越强调近邻排斥。

    Returns:
        标量张量，表示均匀性损失。
    """
    # 月度输出合并为 (B*T, D)。
    if emb.dim() == 3:
        emb = emb.reshape(-1, emb.shape[-1])

    # L2 归一化，避免除零。
    emb = F.normalize(emb, p=2, dim=1)
    batch_size = emb.shape[0]

    # pairwise squared distance = ||u_i - u_j||^2 = 2 - 2 * u_i @ u_j。
    similarity = emb @ emb.t()  # [B, B]
    squared_dist = 2.0 - 2.0 * similarity

    # 排除对角线。
    off_diag_count = batch_size * (batch_size - 1)
    if off_diag_count == 0:
        return torch.tensor(0.0, device=emb.device, dtype=emb.dtype)

    diag_mask = ~torch.eye(batch_size, device=emb.device, dtype=torch.bool)
    off_diag_dist = squared_dist[diag_mask]
    return torch.log(torch.exp(-temperature * off_diag_dist).mean())


def _embedding_tokens(
    embedding_map: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Convert dense embedding maps to pooled ``[N, D]`` tokens."""
    if embedding_map.dim() == 5:
        batch_size, num_months, dim, height, width = embedding_map.shape
        x = embedding_map.reshape(batch_size * num_months, dim, height, width)
    elif embedding_map.dim() == 4:
        x = embedding_map
    else:
        raise ValueError(f"不支持的 embedding_map 形状: {tuple(embedding_map.shape)}")
    pool_size = max(1, int(pool_size))
    pool_size = min(pool_size, x.shape[-2], x.shape[-1])
    if pool_size > 1:
        x = F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def covariance_regularization_loss(
    embedding_map: torch.Tensor,
    std_target: float = 0.08,
    pool_size: int = 8,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg/VCReg-style variance-covariance regularization.

    The vMF bottleneck already constrains vectors to the unit sphere. This term
    encourages every dimension to carry non-trivial variance and discourages
    redundant dimensions, improving effective rank without using downstream
    fine labels.
    """
    tokens = _embedding_tokens(embedding_map, pool_size=pool_size)
    tokens = F.normalize(tokens, p=2, dim=1)
    if tokens.shape[0] < 2:
        zero = embedding_map.sum() * 0.0
        return zero, {
            "variance": zero,
            "covariance": zero,
            "std_mean": zero,
            "std_min": zero,
        }

    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    std = torch.sqrt(tokens.var(dim=0, unbiased=False) + eps)
    variance = F.relu(float(std_target) - std).mean()
    cov = (tokens.t() @ tokens) / max(tokens.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance = off_diag.pow(2).sum() / tokens.shape[1]
    return variance + covariance, {
        "variance": variance.detach(),
        "covariance": covariance.detach(),
        "std_mean": std.mean().detach(),
        "std_min": std.min().detach(),
    }


def patch_discrimination_loss(
    embedding_map: torch.Tensor,
    temperature: float = 0.15,
    pool_size: int = 16,
    max_tokens: int = 512,
) -> torch.Tensor:
    """OlmoEarth-inspired token discrimination over pooled embedding patches.

    Each pooled token is matched with its detached copy and contrasted against
    other tokens in the same batch. This is a lightweight proxy for latent patch
    discrimination: it pushes local tokens to remain distinguishable while the
    reconstruction and semantic losses keep nearby same-class regions aligned.
    """
    tokens = _embedding_tokens(embedding_map, pool_size=pool_size)
    tokens = F.normalize(tokens, p=2, dim=1)
    max_tokens = max(2, int(max_tokens))
    if tokens.shape[0] > max_tokens:
        # Deterministic evenly spaced subsampling keeps distributed workers stable.
        indices = torch.linspace(
            0,
            tokens.shape[0] - 1,
            steps=max_tokens,
            device=tokens.device,
        ).long()
        tokens = tokens.index_select(0, indices)
    if tokens.shape[0] < 2:
        return embedding_map.sum() * 0.0
    logits = tokens @ tokens.detach().t()
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(tokens.shape[0], device=tokens.device)
    return F.cross_entropy(logits, labels)


def temporal_endpoint_separation_loss(
    emb: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Encourage the first and last monthly scene embeddings to be distinguishable.

    The downstream 202512/202605 tasks depend on temporal sensitivity. This hinge
    term is zero once the cosine distance ``1 - cos(first, last)`` reaches
    ``margin`` and positive when endpoint embeddings collapse together.
    """
    if emb.dim() != 3 or emb.shape[1] < 2:
        return torch.tensor(0.0, device=emb.device, dtype=emb.dtype)

    first = F.normalize(emb[:, 0, :], p=2, dim=1)
    last = F.normalize(emb[:, -1, :], p=2, dim=1)
    cosine = (first * last).sum(dim=1)
    distance = 1.0 - cosine
    return F.relu(float(margin) - distance).mean()


def _temporal_mask_to_spatial(
    mask: torch.Tensor,
    batch_size: int,
    num_months: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert common temporal masks to ``[B, T, H, W]``."""
    if mask.dim() == 5:
        mask = mask.squeeze(2)
    if mask.dim() == 4:
        return mask.float()
    if mask.dim() == 3 and mask.shape == (batch_size, num_months, 1):
        return mask.reshape(batch_size, num_months, 1, 1).expand(
            batch_size, num_months, height, width
        )
    if mask.dim() == 2:
        return mask.reshape(batch_size, num_months, 1, 1).expand(
            batch_size, num_months, height, width
        )
    raise ValueError(f"不支持的 temporal contrast mask 形状: {tuple(mask.shape)}")


def temporal_change_aware_contrast_loss(
    embedding_map: torch.Tensor,
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    source_names: list[str] | tuple[str, ...] = ("s2_recon", "s1_recon", "landsat_recon"),
    margin: float = 0.2,
    change_z: float = 0.75,
    stable_z: float = -0.25,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Use source deltas to pull stable pixels together and push changed pixels apart.

    The source deltas are weak pseudo labels. Pixels whose source difference is
    much lower than the batch mean are treated as stable. Pixels whose source
    difference is much higher than the batch mean are treated as likely changed.
    Ambiguous middle pixels are ignored.
    """
    if embedding_map.dim() != 5 or embedding_map.shape[1] < 2:
        zero = torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype)
        return zero, {"stable": zero, "change": zero, "stable_pixels": zero, "change_pixels": zero}

    batch_size, num_months, _, height, width = embedding_map.shape
    diff_sum = torch.zeros(batch_size, height, width, device=embedding_map.device)
    diff_count = torch.zeros(batch_size, height, width, device=embedding_map.device)

    for name in source_names:
        target = targets.get(name)
        mask = masks.get(name)
        if target is None or mask is None or target.dim() != 5 or target.shape[1] < 2:
            continue
        target = target.float()
        spatial_mask = _temporal_mask_to_spatial(
            mask,
            batch_size=batch_size,
            num_months=target.shape[1],
            height=target.shape[-2],
            width=target.shape[-1],
        ).to(device=embedding_map.device)
        valid = (spatial_mask[:, 0] > 0) & (spatial_mask[:, -1] > 0)
        source_diff = (target[:, -1] - target[:, 0]).abs().mean(dim=1)
        if source_diff.shape[-2:] != (height, width):
            source_diff = F.interpolate(
                source_diff.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            valid = (
                F.interpolate(
                    valid.float().unsqueeze(1),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(1)
                > 0
            )
        diff_sum = diff_sum + source_diff.to(embedding_map.device) * valid.float()
        diff_count = diff_count + valid.float()

    valid_pixels = diff_count > 0
    if not bool(valid_pixels.any().item()):
        zero = torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype)
        return zero, {"stable": zero, "change": zero, "stable_pixels": zero, "change_pixels": zero}

    pseudo_delta = diff_sum / diff_count.clamp(min=1.0)
    valid_values = pseudo_delta[valid_pixels]
    delta_mean = valid_values.mean()
    delta_std = valid_values.std(unbiased=False).clamp(min=eps)
    stable_mask = valid_pixels & (pseudo_delta <= delta_mean + float(stable_z) * delta_std)
    change_mask = valid_pixels & (pseudo_delta >= delta_mean + float(change_z) * delta_std)

    first = F.normalize(embedding_map[:, 0], p=2, dim=1)
    last = F.normalize(embedding_map[:, -1], p=2, dim=1)
    distance = 1.0 - (first * last).sum(dim=1)

    zero = torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype)
    stable_loss = distance[stable_mask].mean() if bool(stable_mask.any().item()) else zero
    change_loss = (
        F.relu(float(margin) - distance[change_mask]).mean()
        if bool(change_mask.any().item())
        else zero
    )
    loss = stable_loss + change_loss
    stats = {
        "stable": stable_loss,
        "change": change_loss,
        "stable_pixels": stable_mask.float().sum(),
        "change_pixels": change_mask.float().sum(),
    }
    return loss, stats


def supervised_change_alignment_loss(
    embedding_map: torch.Tensor,
    labels: dict[str, torch.Tensor] | None,
    label_masks: dict[str, torch.Tensor] | None,
    tasks: list[str] | tuple[str, ...] = (),
    pos_margin: float = 0.35,
    neg_margin: float = 0.05,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    hard_negative_ratio: float = 1.0,
    task_weights: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align endpoint embedding distance with sparse downstream label masks.

    Positive pixels should have larger before/after embedding distance. Valid
    background pixels should remain close. P1 computes this per task, applies
    optional task weights, and can mine only the hardest negative pixels.
    """
    zero = torch.tensor(0.0, device=embedding_map.device, dtype=embedding_map.dtype)
    empty_stats = {
        "positive": zero,
        "negative": zero,
        "positive_pixels": zero,
        "negative_pixels": zero,
    }
    if embedding_map.dim() != 5 or embedding_map.shape[1] < 2 or not labels:
        return zero, empty_stats

    first = F.normalize(embedding_map[:, 0], p=2, dim=1)
    last = F.normalize(embedding_map[:, -1], p=2, dim=1)
    distance = 1.0 - (first * last).sum(dim=1)
    height, width = distance.shape[-2:]

    task_names = tuple(tasks) if tasks else tuple(labels.keys())
    label_masks = label_masks or {}
    task_weights = task_weights or {}
    ratio = max(0.0, min(1.0, float(hard_negative_ratio)))

    loss_sum = zero
    task_weight_sum = zero
    pos_loss_sum = zero
    neg_loss_sum = zero
    pos_task_count = zero
    neg_task_count = zero
    pos_pixel_count = zero
    neg_pixel_count = zero
    sampled_neg_pixel_count = zero
    stats: dict[str, torch.Tensor] = {}

    for task in task_names:
        label = labels.get(task)
        if label is None:
            continue
        label = label.to(device=embedding_map.device, dtype=embedding_map.dtype)
        if label.dim() == 3:
            label = label.unsqueeze(1)
        if label.shape[-2:] != (height, width):
            label = F.interpolate(label, size=(height, width), mode="nearest")
        label = label.squeeze(1)

        avail = label_masks.get(task)
        if avail is None:
            valid = torch.ones_like(label, dtype=torch.bool)
        else:
            avail = avail.to(device=embedding_map.device, dtype=embedding_map.dtype)
            valid = avail.reshape(-1, 1, 1) > 0
            valid = valid.expand_as(label)

        positive = valid & (label > 0.5)
        negative = valid & (label <= 0.5)
        task_pos_pixels = positive.float().sum()
        task_neg_pixels = negative.float().sum()
        if task_pos_pixels <= 0 and task_neg_pixels <= 0:
            continue

        pos_loss_map = F.relu(float(pos_margin) - distance)
        neg_loss_map = F.relu(distance - float(neg_margin))
        if task_pos_pixels > 0:
            task_pos_loss = (pos_loss_map * positive.float()).sum() / task_pos_pixels.clamp(min=eps)
            pos_loss_sum = pos_loss_sum + task_pos_loss
            pos_task_count = pos_task_count + 1.0
        else:
            task_pos_loss = zero

        if task_neg_pixels > 0 and ratio > 0.0:
            task_neg_values = neg_loss_map[negative]
            if ratio < 1.0:
                k = max(1, int(task_neg_values.numel() * ratio))
                task_neg_values = torch.topk(task_neg_values, k=k, largest=True).values
            task_neg_loss = task_neg_values.mean()
            sampled_neg_pixels = torch.tensor(
                float(task_neg_values.numel()),
                device=embedding_map.device,
                dtype=embedding_map.dtype,
            )
            neg_loss_sum = neg_loss_sum + task_neg_loss
            neg_task_count = neg_task_count + 1.0
        else:
            task_neg_loss = zero
            sampled_neg_pixels = zero

        task_weight = torch.tensor(
            float(task_weights.get(task, 1.0)),
            device=embedding_map.device,
            dtype=embedding_map.dtype,
        )
        task_loss = float(pos_weight) * task_pos_loss + float(neg_weight) * task_neg_loss
        loss_sum = loss_sum + task_weight * task_loss
        task_weight_sum = task_weight_sum + task_weight
        pos_pixel_count = pos_pixel_count + task_pos_pixels
        neg_pixel_count = neg_pixel_count + task_neg_pixels
        sampled_neg_pixel_count = sampled_neg_pixel_count + sampled_neg_pixels
        stats[f"supervised_change_{task}_positive_pixels"] = task_pos_pixels
        stats[f"supervised_change_{task}_negative_pixels"] = task_neg_pixels
        stats[f"supervised_change_{task}_sampled_negative_pixels"] = sampled_neg_pixels
        stats[f"supervised_change_{task}_loss"] = task_loss.detach()

    if task_weight_sum <= 0:
        return zero, empty_stats

    positive_loss = pos_loss_sum / pos_task_count.clamp(min=eps)
    negative_loss = neg_loss_sum / neg_task_count.clamp(min=eps)
    loss = loss_sum / task_weight_sum.clamp(min=eps)
    stats.update(
        {
            "positive": positive_loss,
            "negative": negative_loss,
            "positive_pixels": pos_pixel_count,
            "negative_pixels": sampled_neg_pixel_count,
            "valid_negative_pixels": neg_pixel_count,
        }
    )
    return loss, stats

class SemanticProbeLoss(nn.Module):
    """Training-only semantic probes that make embedding maps directly decodable.

    Each task is a tiny probe applied to one monthly embedding map. By default
    this is a 1x1 MLP. When ``hidden_dim <= 0`` it becomes a pure 1x1 linear
    probe, which is useful when we want to force linearly readable embeddings.
    The probes are optimized during embedding training and discarded after
    training.
    """

    def __init__(
        self,
        embed_dim: int,
        tasks: list[str] | tuple[str, ...],
        hidden_dim: int = 64,
        task_weights: dict[str, float] | None = None,
        pos_weight: float = 1.0,
        pos_weights: dict[str, float] | None = None,
        month_index: int = -1,
        hard_negative_ratio: float = 0.0,
        hard_negative_weight: float = 0.0,
        hard_negative_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.task_weights = dict(task_weights or {})
        self.pos_weight = float(pos_weight)
        self.pos_weights = dict(pos_weights or {})
        self.month_index = int(month_index)
        self.hard_negative_ratio = max(0.0, float(hard_negative_ratio))
        self.hard_negative_weight = max(0.0, float(hard_negative_weight))
        self.hard_negative_warmup_epochs = max(0, int(hard_negative_warmup_epochs))
        self.current_epoch = 0
        hidden_dim = int(hidden_dim)
        modules: dict[str, nn.Module] = {}
        for task in self.tasks:
            if hidden_dim <= 0:
                modules[task] = nn.Conv2d(embed_dim, 1, kernel_size=1)
            else:
                modules[task] = nn.Sequential(
                    nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, 1, kernel_size=1),
                )
        self.probes = nn.ModuleDict(modules)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _current_hard_negative_weight(self) -> float:
        if self.hard_negative_weight == 0.0:
            return 0.0
        if self.hard_negative_warmup_epochs <= 0:
            return self.hard_negative_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.hard_negative_warmup_epochs,
        )
        return self.hard_negative_weight * progress

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs * mask
        target = target * mask
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

    def _hard_negative_loss(
        self,
        bce_map: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.hard_negative_ratio <= 0.0 or self.hard_negative_weight <= 0.0:
            return bce_map.sum() * 0.0
        negative_mask = (target < 0.5) & (mask > 0.0)
        values = bce_map[negative_mask]
        if values.numel() == 0:
            return bce_map.sum() * 0.0
        k = max(1, int(values.numel() * self.hard_negative_ratio))
        k = min(k, values.numel())
        return values.topk(k).values.mean()

    def forward(
        self,
        embedding_map: torch.Tensor,
        labels: dict[str, torch.Tensor] | None,
        label_masks: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = embedding_map.sum() * 0.0
        if not self.tasks or labels is None:
            return zero, {
                "semantic_probe_positive_pixels": zero.detach(),
                "semantic_probe_valid_pixels": zero.detach(),
            }

        emb = embedding_map[:, self.month_index]
        total = zero
        task_weight_sum = zero
        total_positive = zero
        total_valid = zero
        stats: dict[str, torch.Tensor] = {}
        hard_negative_weight = self._current_hard_negative_weight()

        for task in self.tasks:
            if task not in labels or task not in self.probes:
                continue
            label = labels[task].to(device=emb.device, dtype=emb.dtype)
            if label.dim() != 3:
                raise ValueError(f"semantic label {task!r} 形状应为 [B,H,W]，实际为 {tuple(label.shape)}")
            label = label[:, None]
            if label.shape[-2:] != emb.shape[-2:]:
                label = F.interpolate(label, size=emb.shape[-2:], mode="nearest")
            label = (label > 0.5).to(dtype=emb.dtype)

            sample_mask = None
            if label_masks is not None and task in label_masks:
                sample_mask = label_masks[task].to(device=emb.device, dtype=emb.dtype)
            if sample_mask is None:
                sample_mask = torch.ones((emb.shape[0],), device=emb.device, dtype=emb.dtype)
            valid = sample_mask[:, None, None, None].expand_as(label)
            if bool((valid.sum() <= 0).item()):
                stats[f"semantic_probe_{task}_loss"] = zero.detach()
                stats[f"semantic_probe_{task}_positive_pixels"] = zero.detach()
                stats[f"semantic_probe_{task}_valid_pixels"] = zero.detach()
                continue

            logits = self.probes[task](emb)
            pw = float(self.pos_weights.get(task, self.pos_weight))
            bce_map = F.binary_cross_entropy_with_logits(
                logits,
                label,
                pos_weight=torch.tensor(pw, device=emb.device, dtype=emb.dtype),
                reduction="none",
            )
            bce = self._masked_mean(bce_map, valid)
            dice = self._dice_loss(logits, label, valid)
            hard_negative = self._hard_negative_loss(bce_map, label, valid)
            task_loss = bce + dice + hard_negative_weight * hard_negative
            weight = torch.tensor(
                float(self.task_weights.get(task, 1.0)),
                device=emb.device,
                dtype=emb.dtype,
            )
            total = total + weight * task_loss
            task_weight_sum = task_weight_sum + weight
            positive_pixels = (label * valid).sum()
            valid_pixels = valid.sum()
            total_positive = total_positive + positive_pixels
            total_valid = total_valid + valid_pixels
            stats[f"semantic_probe_{task}_loss"] = task_loss.detach()
            stats[f"semantic_probe_{task}_hard_negative"] = hard_negative.detach()
            stats[f"semantic_probe_{task}_hard_negative_weight"] = torch.tensor(
                hard_negative_weight,
                device=emb.device,
                dtype=emb.dtype,
            ).detach()
            stats[f"semantic_probe_{task}_positive_pixels"] = positive_pixels.detach()
            stats[f"semantic_probe_{task}_valid_pixels"] = valid_pixels.detach()

        loss = total / task_weight_sum.clamp(min=1.0)
        stats["semantic_probe_positive_pixels"] = total_positive.detach()
        stats["semantic_probe_valid_pixels"] = total_valid.detach()
        return loss, stats


def _sample_masked_pixels(
    emb: torch.Tensor,
    pixel_mask: torch.Tensor,
    max_pixels: int,
) -> torch.Tensor:
    """从 ``emb``（[B, D, H, W]）中按布尔掩码采样最多 ``max_pixels`` 个像素嵌入。

    使用等间隔下采样保证分布式各 rank 行为一致，返回 ``[N, D]``。
    """
    flat_emb = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
    flat_mask = pixel_mask.reshape(-1)
    indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return flat_emb.new_zeros((0, emb.shape[1]))
    if indices.numel() > max_pixels:
        positions = torch.linspace(
            0, indices.numel() - 1, steps=max_pixels, device=indices.device
        ).long()
        indices = indices.index_select(0, positions)
    return flat_emb.index_select(0, indices)


class SemanticPrototypeContrastLoss(nn.Module):
    """OSM 弱语义类别原型对比损失（v3/P14 新增）。

    为每个弱语义类别维护一个可学习的单位球面原型：正类像素被拉向本类
    原型、以 margin 推离其他类原型；负类像素以 margin 推离本类原型；并
    显式分离原型两两方向。目标是在不使用下游手工标注的前提下，让
    embedding 空间形成按类聚拢、类间分离的方向结构，直接服务相似性检索
    （centroid/kNN）与 few-shot 新类别。

    OSM 类别之间存在空间重叠（如 building 与 residential），因此不使用
    单标签 softmax，而使用多标签安全的 margin 公式。原型作为可学习参数
    由 DDP 自动同步梯度，训练结束后与 probe 一样丢弃。
    """

    def __init__(
        self,
        embed_dim: int,
        tasks: list[str] | tuple[str, ...],
        task_weights: dict[str, float] | None = None,
        negative_margin: float = 0.25,
        separation_margin: float = 0.35,
        max_pixels_per_task: int = 256,
        month_index: int = -1,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        self.task_weights = dict(task_weights or {})
        self.negative_margin = float(negative_margin)
        self.separation_margin = float(separation_margin)
        self.max_pixels_per_task = max(1, int(max_pixels_per_task))
        self.month_index = int(month_index)
        self.prototypes = nn.Parameter(
            torch.randn(len(self.tasks), embed_dim) * 0.02
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        labels: dict[str, torch.Tensor] | None,
        label_masks: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = embedding_map.sum() * 0.0
        stats: dict[str, torch.Tensor] = {
            "prototype_contrast_positive_pixels": zero.detach(),
        }
        if not self.tasks or labels is None:
            return zero, stats

        emb = embedding_map[:, self.month_index]  # [B, D, H, W]
        protos = F.normalize(self.prototypes, p=2, dim=-1)  # [K, D]

        total = zero
        weight_sum = zero
        total_positive = zero
        for task_idx, task in enumerate(self.tasks):
            if task not in labels:
                continue
            label = labels[task].to(device=emb.device, dtype=emb.dtype)
            if label.dim() != 3:
                raise ValueError(
                    f"prototype label {task!r} 形状应为 [B,H,W]，实际为 {tuple(label.shape)}"
                )
            label = label[:, None]
            if label.shape[-2:] != emb.shape[-2:]:
                label = F.interpolate(label, size=emb.shape[-2:], mode="nearest")
            binary = (label > 0.5).to(dtype=emb.dtype)

            sample_mask = None
            if label_masks is not None and task in label_masks:
                sample_mask = label_masks[task].to(device=emb.device, dtype=emb.dtype)
            if sample_mask is None:
                sample_mask = torch.ones(
                    (emb.shape[0],), device=emb.device, dtype=emb.dtype
                )
            valid = sample_mask[:, None, None, None].expand_as(binary)

            pos_mask = (binary > 0.5) & (valid > 0.0)
            neg_mask = (binary < 0.5) & (valid > 0.0)
            pos_emb = _sample_masked_pixels(
                emb, pos_mask[:, 0], self.max_pixels_per_task
            )
            if pos_emb.shape[0] == 0:
                continue
            pos_emb = F.normalize(pos_emb, p=2, dim=1)
            pos_sim = pos_emb @ protos.t()  # [N, K]

            pull = (1.0 - pos_sim[:, task_idx]).mean()
            if len(self.tasks) > 1:
                other_sim = torch.cat(
                    [pos_sim[:, :task_idx], pos_sim[:, task_idx + 1 :]], dim=1
                )
                push_other = F.relu(other_sim - self.negative_margin).mean()
            else:
                push_other = zero

            neg_emb = _sample_masked_pixels(
                emb, neg_mask[:, 0], self.max_pixels_per_task
            )
            if neg_emb.shape[0] > 0:
                neg_emb = F.normalize(neg_emb, p=2, dim=1)
                neg_sim = neg_emb @ protos[task_idx]
                push_neg = F.relu(neg_sim - self.negative_margin).mean()
            else:
                push_neg = zero

            task_loss = pull + push_other + push_neg
            weight = torch.tensor(
                float(self.task_weights.get(task, 1.0)),
                device=emb.device,
                dtype=emb.dtype,
            )
            total = total + weight * task_loss
            weight_sum = weight_sum + weight
            total_positive = total_positive + pos_mask.sum()
            stats[f"prototype_contrast_{task}_loss"] = task_loss.detach()

        loss = total / weight_sum.clamp(min=1.0)
        if len(self.tasks) > 1:
            proto_sim = protos @ protos.t()
            off_diag = proto_sim - torch.diag(torch.diag(proto_sim))
            separation = F.relu(off_diag - self.separation_margin).sum() / (
                len(self.tasks) * (len(self.tasks) - 1)
            )
            loss = loss + separation
            stats["prototype_contrast_separation"] = separation.detach()
        stats["prototype_contrast_positive_pixels"] = total_positive.detach()
        return loss, stats


def semantic_boundary_contrast_loss(
    embedding_map: torch.Tensor,
    labels: dict[str, torch.Tensor] | None,
    label_masks: dict[str, torch.Tensor] | None,
    tasks: list[str] | tuple[str, ...],
    band_radius: int = 2,
    margin: float = 0.35,
    max_pixels_per_task: int = 256,
    month_index: int = -1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """OSM 边界带对比损失（v3/P14 新增）。

    对每个弱语义类别，用形态学膨胀/腐蚀（max_pool2d 实现）在标签边界两侧
    取窄带像素：边界内侧与外侧的 embedding 余弦相似度应低于 ``margin``。
    该项直接针对"建筑假正例多、边界模糊"的问题，锐化类别边界处的
    embedding 过渡，且不使用下游手工标注。
    """
    zero = embedding_map.sum() * 0.0
    stats: dict[str, torch.Tensor] = {"boundary_contrast_pairs": zero.detach()}
    if not tasks or labels is None:
        return zero, stats

    emb = embedding_map[:, month_index]  # [B, D, H, W]
    kernel = 2 * max(1, int(band_radius)) + 1
    pad = kernel // 2

    total = zero
    active_tasks = 0
    total_pairs = zero
    for task in tasks:
        if task not in labels:
            continue
        label = labels[task].to(device=emb.device, dtype=emb.dtype)
        if label.dim() != 3:
            raise ValueError(
                f"boundary label {task!r} 形状应为 [B,H,W]，实际为 {tuple(label.shape)}"
            )
        label = label[:, None]
        if label.shape[-2:] != emb.shape[-2:]:
            label = F.interpolate(label, size=emb.shape[-2:], mode="nearest")
        binary = (label > 0.5).to(dtype=emb.dtype)

        sample_mask = None
        if label_masks is not None and task in label_masks:
            sample_mask = label_masks[task].to(device=emb.device, dtype=emb.dtype)
        if sample_mask is None:
            sample_mask = torch.ones(
                (emb.shape[0],), device=emb.device, dtype=emb.dtype
            )
        valid = sample_mask[:, None, None, None].expand_as(binary) > 0.0

        dilated = F.max_pool2d(binary, kernel_size=kernel, stride=1, padding=pad)
        eroded = 1.0 - F.max_pool2d(
            1.0 - binary, kernel_size=kernel, stride=1, padding=pad
        )
        inner_band = (binary > 0.5) & (eroded < 0.5) & valid
        outer_band = (dilated > 0.5) & (binary < 0.5) & valid

        inner_emb = _sample_masked_pixels(emb, inner_band[:, 0], max_pixels_per_task)
        outer_emb = _sample_masked_pixels(emb, outer_band[:, 0], max_pixels_per_task)
        if inner_emb.shape[0] == 0 or outer_emb.shape[0] == 0:
            continue
        n = min(inner_emb.shape[0], outer_emb.shape[0])
        inner_emb = F.normalize(inner_emb[:n], p=2, dim=1)
        outer_emb = F.normalize(outer_emb[:n], p=2, dim=1)
        cross_sim = (inner_emb * outer_emb).sum(dim=1)
        task_loss = F.relu(cross_sim - margin).mean()
        total = total + task_loss
        active_tasks += 1
        total_pairs = total_pairs + torch.tensor(
            float(n), device=emb.device, dtype=emb.dtype
        )
        stats[f"boundary_contrast_{task}_loss"] = task_loss.detach()

    if active_tasks == 0:
        return zero, stats
    stats["boundary_contrast_pairs"] = total_pairs.detach()
    return total / active_tasks, stats


class LatentReconstructionLoss(nn.Module):
    """Direct latent probes from embedding maps to stable source targets.

    The main reconstruction decoders can learn task-specific decoding paths. In
    contrast, these tiny 1x1 probes force the embedding map itself to retain
    modality information that is linearly readable from each pixel token. This
    follows the OlmoEarth-style idea of predicting stable latent targets from a
    masked multimodal context while keeping the representation useful for small
    downstream heads.
    """

    def __init__(
        self,
        embed_dim: int,
        target_cfg: dict[str, dict],
        target_names: list[str] | tuple[str, ...],
        target_weights: dict[str, float] | None = None,
        hidden_dim: int = 0,
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.target_names = tuple(target_names)
        self.target_weights = dict(target_weights or {})
        self.loss_type = str(loss_type)
        hidden_dim = int(hidden_dim)
        probes: dict[str, nn.Module] = {}
        for name in self.target_names:
            cfg = target_cfg.get(name)
            if cfg is None:
                raise KeyError(f"latent target {name!r} not found in target_cfg")
            if cfg["loss_type"] != "l1":
                raise ValueError(
                    f"latent target {name!r} must be continuous/l1, got {cfg['loss_type']!r}"
                )
            channels = int(cfg["channels"])
            if hidden_dim <= 0:
                probes[name] = nn.Conv2d(embed_dim, channels, kernel_size=1)
            else:
                probes[name] = nn.Sequential(
                    nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, channels, kernel_size=1),
                )
        self.probes = nn.ModuleDict(probes)

    @staticmethod
    def _masked_channel_standardize(
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        valid = mask[:, None].to(dtype=target.dtype)
        count = valid.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)
        mean = (target * valid).sum(dim=[2, 3], keepdim=True) / count
        var = ((target - mean).pow(2) * valid).sum(dim=[2, 3], keepdim=True) / count
        return (target - mean) / torch.sqrt(var + eps)

    @staticmethod
    def _loss_map(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        if loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none").mean(dim=1)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="none", beta=0.5).mean(dim=1)
        if loss_type == "mse":
            return F.mse_loss(pred, target, reduction="none").mean(dim=1)
        raise ValueError(f"Unsupported latent reconstruction loss_type: {loss_type!r}")

    def forward(
        self,
        embedding_map: torch.Tensor,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = embedding_map.sum() * 0.0
        if not self.target_names:
            return zero, {"latent_reconstruction_valid_pixels": zero.detach()}

        if embedding_map.dim() != 5:
            raise ValueError(
                f"latent reconstruction expects monthly embedding_map [B,T,D,H,W], got {tuple(embedding_map.shape)}"
            )
        batch_size, num_months, dim, height, width = embedding_map.shape
        emb = embedding_map.reshape(batch_size * num_months, dim, height, width)
        total = zero
        weight_sum = zero
        total_valid_pixels = zero
        stats: dict[str, torch.Tensor] = {}

        for name in self.target_names:
            target = targets.get(name)
            mask = masks.get(name)
            if target is None or mask is None:
                continue
            if target.dim() != 5:
                raise ValueError(
                    f"latent target {name!r} must be [B,T,C,H,W], got {tuple(target.shape)}"
                )
            if target.shape[0] != batch_size or target.shape[1] != num_months:
                raise ValueError(
                    f"latent target {name!r} batch/time mismatch: expected "
                    f"({batch_size}, {num_months}), got {tuple(target.shape[:2])}"
                )
            target = target.to(device=embedding_map.device, dtype=embedding_map.dtype)
            mask = _temporal_mask_to_spatial(
                mask.to(device=embedding_map.device, dtype=embedding_map.dtype),
                batch_size=batch_size,
                num_months=num_months,
                height=target.shape[-2],
                width=target.shape[-1],
            )
            if mask.shape[:2] != (batch_size, num_months):
                raise ValueError(
                    f"latent mask {name!r} batch/time mismatch: expected "
                    f"({batch_size}, {num_months}), got {tuple(mask.shape[:2])}"
                )
            if target.shape[-2:] != (height, width):
                target = target.reshape(
                    batch_size * num_months,
                    target.shape[2],
                    target.shape[-2],
                    target.shape[-1],
                )
                target = F.interpolate(
                    target,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                target = target.reshape(batch_size, num_months, target.shape[1], height, width)
                mask = F.interpolate(
                    mask.reshape(batch_size * num_months, 1, mask.shape[-2], mask.shape[-1]),
                    size=(height, width),
                    mode="nearest",
                ).reshape(batch_size, num_months, height, width)
            target = target.reshape(batch_size * num_months, target.shape[2], height, width)
            mask = mask.reshape(batch_size * num_months, height, width)
            finite_mask = torch.isfinite(target).all(dim=1).to(dtype=mask.dtype)
            mask = mask * finite_mask
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            valid_pixels = mask.sum()
            if bool((valid_pixels <= 0).item()):
                stats[f"latent_reconstruction_{name}"] = zero.detach()
                stats[f"latent_reconstruction_{name}_valid_pixels"] = zero.detach()
                continue

            target = self._masked_channel_standardize(target, mask)
            pred = self.probes[name](emb)
            loss_map = self._loss_map(pred, target.detach(), self.loss_type)
            loss = (loss_map * mask).sum() / valid_pixels.clamp(min=1.0)
            weight = torch.tensor(
                float(self.target_weights.get(name, 1.0)),
                device=embedding_map.device,
                dtype=embedding_map.dtype,
            )
            total = total + weight * loss
            weight_sum = weight_sum + weight
            total_valid_pixels = total_valid_pixels + valid_pixels
            stats[f"latent_reconstruction_{name}"] = loss.detach()
            stats[f"latent_reconstruction_{name}_valid_pixels"] = valid_pixels.detach()

        if bool((weight_sum <= 0).item()):
            return zero, {"latent_reconstruction_valid_pixels": zero.detach(), **stats}
        return total / weight_sum.clamp(min=1.0), {
            "latent_reconstruction_valid_pixels": total_valid_pixels.detach(),
            **stats,
        }


class DinoDistillLoss(nn.Module):
    """DINOv3 教师特征蒸馏：cosine 对齐 + Gram anchoring。

    教师特征离线预计算（32x32xC_t dense tokens）。学生 embedding_map 池化到
    教师网格后经 1x1 投影头对齐教师维度做 cosine 蒸馏；Gram anchoring 直接
    对齐 token 间相似度结构（与维度无关），保持 dense feature 空间结构。
    投影头仅训练期使用，发布时丢弃。
    """

    def __init__(
        self,
        embed_dim: int,
        teacher_dim: int = 1024,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, teacher_dim, kernel_size=1)
        self.max_tokens = int(max_tokens)

    def forward(
        self,
        embedding_map: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """计算蒸馏损失。

        Args:
            embedding_map: [B, T, C, H, W] 或 [B, C, H, W]。
            teacher_features: [B, Gh, Gw, C_t]。
            teacher_valid: [B] bool，样本是否有教师特征。
        """
        if embedding_map.dim() == 5:
            student = embedding_map.mean(dim=1)
        else:
            student = embedding_map
        batch, _, _, _ = student.shape
        grid_h, grid_w = teacher_features.shape[1], teacher_features.shape[2]
        teacher = teacher_features.permute(0, 3, 1, 2).to(student.dtype)

        if teacher_valid is None:
            teacher_valid = torch.ones(
                batch, dtype=torch.bool, device=student.device
            )
        valid_count = int(teacher_valid.sum().item())

        pooled = F.adaptive_avg_pool2d(student, (grid_h, grid_w))
        pooled = F.normalize(pooled, p=2, dim=1)
        projected = self.proj(pooled)

        zero = embedding_map.sum() * 0.0 + sum(
            p.sum() for p in self.proj.parameters()
        ) * 0.0
        if valid_count == 0:
            stats = {
                "distill_valid_samples": torch.tensor(
                    0.0, device=student.device, dtype=student.dtype
                )
            }
            return zero, zero.clone(), stats

        proj_flat = projected.flatten(2).transpose(1, 2)  # [B, N, C_t]
        teach_flat = teacher.flatten(2).transpose(1, 2)  # [B, N, C_t]
        cos = F.cosine_similarity(proj_flat, teach_flat, dim=-1)  # [B, N]
        valid_f = teacher_valid.to(student.dtype)
        cos_loss = ((1.0 - cos).mean(dim=1) * valid_f).sum() / valid_f.sum()

        num_tokens = grid_h * grid_w
        if num_tokens > self.max_tokens:
            step = max(1, num_tokens // self.max_tokens)
            idx = torch.arange(0, num_tokens, step, device=student.device)[
                : self.max_tokens
            ]
        else:
            idx = torch.arange(num_tokens, device=student.device)
        stud_tok = F.normalize(
            pooled.flatten(2).transpose(1, 2)[:, idx, :], p=2, dim=-1
        )
        teach_tok = F.normalize(teach_flat[:, idx, :], p=2, dim=-1)
        gram_s = stud_tok @ stud_tok.transpose(1, 2)
        gram_t = teach_tok @ teach_tok.transpose(1, 2)
        gram_diff = (gram_s - gram_t).pow(2).mean(dim=[1, 2])
        gram_loss = (gram_diff * valid_f).sum() / valid_f.sum()

        stats = {
            "distill_valid_samples": torch.tensor(
                float(valid_count), device=student.device, dtype=student.dtype
            ),
            "distill_cosine_mean": (
                (cos.mean(dim=1) * valid_f).sum() / valid_f.sum()
            ).detach(),
        }
        return cos_loss + zero, gram_loss, stats


class TotalLoss(nn.Module):
    """AEF 训练总损失：加权重建损失 + 表征正则项。"""

    def __init__(
        self,
        target_cfg: dict[str, dict],
        uniformity_weight: float = 1.0,
        uniformity_warmup_epochs: int = 0,
        uniformity_temperature: float = 2.0,
        covariance_weight: float = 0.0,
        covariance_warmup_epochs: int = 0,
        covariance_std_target: float = 0.08,
        covariance_pool_size: int = 8,
        patch_discrimination_weight: float = 0.0,
        patch_discrimination_warmup_epochs: int = 0,
        patch_discrimination_temperature: float = 0.15,
        patch_discrimination_pool_size: int = 16,
        patch_discrimination_max_tokens: int = 512,
        temporal_endpoint_weight: float = 0.0,
        temporal_endpoint_warmup_epochs: int = 0,
        temporal_endpoint_margin: float = 0.15,
        temporal_contrast_weight: float = 0.0,
        temporal_contrast_warmup_epochs: int = 0,
        temporal_contrast_margin: float = 0.2,
        temporal_contrast_change_z: float = 0.75,
        temporal_contrast_stable_z: float = -0.25,
        temporal_contrast_sources: list[str] | tuple[str, ...] = (
            "s2_recon",
            "s1_recon",
            "landsat_recon",
        ),
        supervised_change_weight: float = 0.0,
        supervised_change_warmup_epochs: int = 0,
        supervised_change_pos_margin: float = 0.35,
        supervised_change_neg_margin: float = 0.05,
        supervised_change_tasks: list[str] | tuple[str, ...] = (),
        supervised_change_pos_weight: float = 1.0,
        supervised_change_neg_weight: float = 1.0,
        supervised_change_hard_negative_ratio: float = 1.0,
        supervised_change_task_weights: dict[str, float] | None = None,
        semantic_probe_embed_dim: int | None = None,
        semantic_probe_weight: float = 0.0,
        semantic_probe_warmup_epochs: int = 0,
        semantic_probe_tasks: list[str] | tuple[str, ...] = (),
        semantic_probe_task_weights: dict[str, float] | None = None,
        semantic_probe_pos_weight: float = 1.0,
        semantic_probe_pos_weights: dict[str, float] | None = None,
        semantic_probe_hidden_dim: int = 64,
        semantic_probe_hard_negative_ratio: float = 0.0,
        semantic_probe_hard_negative_weight: float = 0.0,
        semantic_probe_hard_negative_warmup_epochs: int = 0,
        prototype_contrast_embed_dim: int | None = None,
        prototype_contrast_weight: float = 0.0,
        prototype_contrast_warmup_epochs: int = 0,
        prototype_contrast_tasks: list[str] | tuple[str, ...] = (),
        prototype_contrast_task_weights: dict[str, float] | None = None,
        prototype_contrast_negative_margin: float = 0.25,
        prototype_contrast_separation_margin: float = 0.35,
        prototype_contrast_max_pixels: int = 256,
        boundary_contrast_weight: float = 0.0,
        boundary_contrast_warmup_epochs: int = 0,
        boundary_contrast_tasks: list[str] | tuple[str, ...] = (),
        boundary_contrast_band_radius: int = 2,
        boundary_contrast_margin: float = 0.35,
        boundary_contrast_max_pixels: int = 256,
        latent_reconstruction_embed_dim: int | None = None,
        latent_reconstruction_weight: float = 0.0,
        latent_reconstruction_warmup_epochs: int = 0,
        latent_reconstruction_targets: list[str] | tuple[str, ...] = (),
        latent_reconstruction_target_weights: dict[str, float] | None = None,
        latent_reconstruction_hidden_dim: int = 0,
        latent_reconstruction_loss_type: str = "smooth_l1",
        distill_embed_dim: int | None = None,
        distill_weight: float = 0.0,
        distill_gram_weight: float = 0.0,
        distill_warmup_epochs: int = 0,
        distill_final_weight: float | None = None,
        distill_decay_start_epoch: int = 0,
        distill_decay_end_epoch: int = 0,
        distill_teacher_dim: int = 1024,
        distill_max_tokens: int = 256,
        loss_crop_size: int | None = None,
    ) -> None:
        """初始化。

        Args:
            target_cfg: 各目标模态配置，格式::

                {
                    "name": {
                        "loss_type": "l1" | "ce",
                        "channels": int,
                        "weight": float,
                    },
                    ...
                }
        """
        super().__init__()
        self.target_cfg = target_cfg
        self.uniformity_weight = float(uniformity_weight)
        self.uniformity_warmup_epochs = int(uniformity_warmup_epochs)
        self.uniformity_temperature = float(uniformity_temperature)
        self.covariance_weight = float(covariance_weight)
        self.covariance_warmup_epochs = int(covariance_warmup_epochs)
        self.covariance_std_target = float(covariance_std_target)
        self.covariance_pool_size = int(covariance_pool_size)
        self.patch_discrimination_weight = float(patch_discrimination_weight)
        self.patch_discrimination_warmup_epochs = int(
            patch_discrimination_warmup_epochs
        )
        self.patch_discrimination_temperature = float(patch_discrimination_temperature)
        self.patch_discrimination_pool_size = int(patch_discrimination_pool_size)
        self.patch_discrimination_max_tokens = int(patch_discrimination_max_tokens)
        self.temporal_endpoint_weight = float(temporal_endpoint_weight)
        self.temporal_endpoint_warmup_epochs = int(temporal_endpoint_warmup_epochs)
        self.temporal_endpoint_margin = float(temporal_endpoint_margin)
        self.temporal_contrast_weight = float(temporal_contrast_weight)
        self.temporal_contrast_warmup_epochs = int(temporal_contrast_warmup_epochs)
        self.temporal_contrast_margin = float(temporal_contrast_margin)
        self.temporal_contrast_change_z = float(temporal_contrast_change_z)
        self.temporal_contrast_stable_z = float(temporal_contrast_stable_z)
        self.temporal_contrast_sources = tuple(temporal_contrast_sources)
        self.supervised_change_weight = float(supervised_change_weight)
        self.supervised_change_warmup_epochs = int(supervised_change_warmup_epochs)
        self.supervised_change_pos_margin = float(supervised_change_pos_margin)
        self.supervised_change_neg_margin = float(supervised_change_neg_margin)
        self.supervised_change_tasks = tuple(supervised_change_tasks)
        self.supervised_change_pos_weight = float(supervised_change_pos_weight)
        self.supervised_change_neg_weight = float(supervised_change_neg_weight)
        self.supervised_change_hard_negative_ratio = float(
            supervised_change_hard_negative_ratio
        )
        self.supervised_change_task_weights = dict(supervised_change_task_weights or {})
        self.semantic_probe_weight = float(semantic_probe_weight)
        self.semantic_probe_warmup_epochs = int(semantic_probe_warmup_epochs)
        self.semantic_probe_tasks = tuple(semantic_probe_tasks)
        if self.semantic_probe_tasks and semantic_probe_embed_dim is None:
            raise ValueError("semantic_probe_embed_dim is required when semantic_probe_tasks is not empty")
        self.semantic_probe = (
            SemanticProbeLoss(
                embed_dim=int(semantic_probe_embed_dim or 1),
                tasks=self.semantic_probe_tasks,
                hidden_dim=int(semantic_probe_hidden_dim),
                task_weights=semantic_probe_task_weights,
                pos_weight=semantic_probe_pos_weight,
                pos_weights=semantic_probe_pos_weights,
                hard_negative_ratio=semantic_probe_hard_negative_ratio,
                hard_negative_weight=semantic_probe_hard_negative_weight,
                hard_negative_warmup_epochs=(
                    semantic_probe_hard_negative_warmup_epochs
                ),
            )
            if self.semantic_probe_tasks
            else None
        )
        self.distill_weight = float(distill_weight)
        self.distill_gram_weight = float(distill_gram_weight)
        self.distill_warmup_epochs = int(distill_warmup_epochs)
        self.distill_final_weight = (
            float(distill_final_weight)
            if distill_final_weight is not None
            else self.distill_weight
        )
        self.distill_decay_start_epoch = int(distill_decay_start_epoch)
        self.distill_decay_end_epoch = int(distill_decay_end_epoch)
        if self.distill_weight > 0.0 and distill_embed_dim is None:
            raise ValueError("distill_embed_dim is required when distill_weight > 0")
        self.distill = (
            DinoDistillLoss(
                embed_dim=int(distill_embed_dim or 1),
                teacher_dim=int(distill_teacher_dim),
                max_tokens=int(distill_max_tokens),
            )
            if self.distill_weight > 0.0
            else None
        )
        self.prototype_contrast_weight = float(prototype_contrast_weight)
        self.prototype_contrast_warmup_epochs = int(prototype_contrast_warmup_epochs)
        self.prototype_contrast_tasks = tuple(prototype_contrast_tasks)
        if self.prototype_contrast_tasks and prototype_contrast_embed_dim is None:
            raise ValueError(
                "prototype_contrast_embed_dim is required when prototype_contrast_tasks is not empty"
            )
        self.prototype_contrast = (
            SemanticPrototypeContrastLoss(
                embed_dim=int(prototype_contrast_embed_dim or 1),
                tasks=self.prototype_contrast_tasks,
                task_weights=prototype_contrast_task_weights,
                negative_margin=prototype_contrast_negative_margin,
                separation_margin=prototype_contrast_separation_margin,
                max_pixels_per_task=prototype_contrast_max_pixels,
            )
            if self.prototype_contrast_tasks
            else None
        )
        self.boundary_contrast_weight = float(boundary_contrast_weight)
        self.boundary_contrast_warmup_epochs = int(boundary_contrast_warmup_epochs)
        self.boundary_contrast_tasks = tuple(boundary_contrast_tasks)
        self.boundary_contrast_band_radius = int(boundary_contrast_band_radius)
        self.boundary_contrast_margin = float(boundary_contrast_margin)
        self.boundary_contrast_max_pixels = int(boundary_contrast_max_pixels)
        self.latent_reconstruction_weight = float(latent_reconstruction_weight)
        self.latent_reconstruction_warmup_epochs = int(
            latent_reconstruction_warmup_epochs
        )
        self.latent_reconstruction_targets = tuple(latent_reconstruction_targets)
        if self.latent_reconstruction_targets and latent_reconstruction_embed_dim is None:
            raise ValueError(
                "latent_reconstruction_embed_dim is required when latent_reconstruction_targets is not empty"
            )
        self.latent_reconstruction = (
            LatentReconstructionLoss(
                embed_dim=int(latent_reconstruction_embed_dim or 1),
                target_cfg=target_cfg,
                target_names=self.latent_reconstruction_targets,
                target_weights=latent_reconstruction_target_weights,
                hidden_dim=latent_reconstruction_hidden_dim,
                loss_type=latent_reconstruction_loss_type,
            )
            if self.latent_reconstruction_targets
            else None
        )
        self.loss_crop_size = int(loss_crop_size) if loss_crop_size else None
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于 uniformity 权重 warmup。"""
        self.current_epoch = int(epoch)
        if self.semantic_probe is not None:
            self.semantic_probe.set_epoch(epoch)

    def _current_uniformity_weight(self) -> float:
        if self.uniformity_weight == 0.0:
            return 0.0
        if self.uniformity_warmup_epochs <= 0:
            return self.uniformity_weight
        progress = min(1.0, float(self.current_epoch + 1) / self.uniformity_warmup_epochs)
        return self.uniformity_weight * progress

    def _warmup_weight(self, weight: float, warmup_epochs: int) -> float:
        if weight == 0.0:
            return 0.0
        if warmup_epochs <= 0:
            return weight
        progress = min(1.0, float(self.current_epoch + 1) / warmup_epochs)
        return weight * progress

    def _current_covariance_weight(self) -> float:
        return self._warmup_weight(
            self.covariance_weight,
            self.covariance_warmup_epochs,
        )

    def _current_patch_discrimination_weight(self) -> float:
        return self._warmup_weight(
            self.patch_discrimination_weight,
            self.patch_discrimination_warmup_epochs,
        )

    def _current_temporal_endpoint_weight(self) -> float:
        if self.temporal_endpoint_weight == 0.0:
            return 0.0
        if self.temporal_endpoint_warmup_epochs <= 0:
            return self.temporal_endpoint_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.temporal_endpoint_warmup_epochs,
        )
        return self.temporal_endpoint_weight * progress

    def _current_temporal_contrast_weight(self) -> float:
        if self.temporal_contrast_weight == 0.0:
            return 0.0
        if self.temporal_contrast_warmup_epochs <= 0:
            return self.temporal_contrast_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.temporal_contrast_warmup_epochs,
        )
        return self.temporal_contrast_weight * progress

    def _current_supervised_change_weight(self) -> float:
        if self.supervised_change_weight == 0.0:
            return 0.0
        if self.supervised_change_warmup_epochs <= 0:
            return self.supervised_change_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.supervised_change_warmup_epochs,
        )
        return self.supervised_change_weight * progress

    def _current_semantic_probe_weight(self) -> float:
        if self.semantic_probe_weight == 0.0:
            return 0.0
        if self.semantic_probe_warmup_epochs <= 0:
            return self.semantic_probe_weight
        progress = min(
            1.0,
            float(self.current_epoch + 1) / self.semantic_probe_warmup_epochs,
        )
        return self.semantic_probe_weight * progress

    def _current_distill_weight(self) -> float:
        base = self._warmup_weight(self.distill_weight, self.distill_warmup_epochs)
        if (
            self.distill_decay_end_epoch <= self.distill_decay_start_epoch
            or self.distill_final_weight >= self.distill_weight
        ):
            return base
        epoch = self.current_epoch
        if epoch < self.distill_decay_start_epoch:
            return base
        if epoch >= self.distill_decay_end_epoch:
            return self.distill_final_weight
        span = self.distill_decay_end_epoch - self.distill_decay_start_epoch
        progress = (epoch - self.distill_decay_start_epoch) / span
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.distill_final_weight + (
            base - self.distill_final_weight
        ) * cos_factor

    def _current_prototype_contrast_weight(self) -> float:
        return self._warmup_weight(
            self.prototype_contrast_weight, self.prototype_contrast_warmup_epochs
        )

    def _current_boundary_contrast_weight(self) -> float:
        return self._warmup_weight(
            self.boundary_contrast_weight, self.boundary_contrast_warmup_epochs
        )

    def _current_latent_reconstruction_weight(self) -> float:
        return self._warmup_weight(
            self.latent_reconstruction_weight,
            self.latent_reconstruction_warmup_epochs,
        )

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        supervised_labels: dict[str, torch.Tensor] | None = None,
        supervised_label_masks: dict[str, torch.Tensor] | None = None,
        teacher_features: torch.Tensor | None = None,
        teacher_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算总损失。

        Args:
            output: 模型前向输出，需包含 ``embedding`` 与 ``reconstructions``。
            targets: 各目标模态的真值。
            masks: 各目标模态的有效掩码。

        Returns:
            包含 ``total``、``recon``、``uniformity`` 以及各目标重建损失
            ``recon_{name}`` 的字典。
        """
        recon_losses: dict[str, torch.Tensor] = {}
        embedding_map = _center_crop_tensor(output.embedding_map, self.loss_crop_size)
        if embedding_map.dim() == 5:
            embedding = embedding_map.mean(dim=[3, 4])
        elif embedding_map.dim() == 4:
            embedding = embedding_map.mean(dim=[2, 3])
        else:
            raise ValueError(
                "embedding_map must have shape [B, C, H, W] or [B, T, C, H, W], "
                f"got {tuple(embedding_map.shape)}"
            )
        embedding = F.normalize(embedding, p=2, dim=-1)
        reconstructions = _center_crop_dict(output.reconstructions, self.loss_crop_size)
        targets = _center_crop_dict(targets, self.loss_crop_size)
        masks = _center_crop_dict(masks, self.loss_crop_size)
        total_recon = torch.tensor(
            0.0, device=embedding.device, dtype=embedding.dtype
        )

        for name, cfg in self.target_cfg.items():
            pred = reconstructions[name]
            target = targets[name]
            mask = masks[name]
            loss = reconstruction_loss(
                pred,
                target,
                mask,
                loss_type=cfg["loss_type"],
            )
            weighted = cfg["weight"] * loss
            recon_losses[f"recon_{name}"] = loss
            total_recon = total_recon + weighted

        uniformity = batch_uniformity_loss(
            embedding,
            temperature=self.uniformity_temperature,
        )
        uniformity_weight = self._current_uniformity_weight()
        weighted_uniformity = uniformity * uniformity_weight
        covariance_weight = self._current_covariance_weight()
        if covariance_weight == 0.0:
            covariance = embedding_map.sum() * 0.0
            covariance_stats = {
                "variance": covariance.detach(),
                "covariance": covariance.detach(),
                "std_mean": covariance.detach(),
                "std_min": covariance.detach(),
            }
        else:
            covariance, covariance_stats = covariance_regularization_loss(
                embedding_map,
                std_target=self.covariance_std_target,
                pool_size=self.covariance_pool_size,
            )
        weighted_covariance = covariance * covariance_weight
        patch_discrimination_weight = self._current_patch_discrimination_weight()
        if patch_discrimination_weight == 0.0:
            patch_discrimination = embedding_map.sum() * 0.0
        else:
            patch_discrimination = patch_discrimination_loss(
                embedding_map,
                temperature=self.patch_discrimination_temperature,
                pool_size=self.patch_discrimination_pool_size,
                max_tokens=self.patch_discrimination_max_tokens,
            )
        weighted_patch_discrimination = (
            patch_discrimination * patch_discrimination_weight
        )
        temporal_endpoint = temporal_endpoint_separation_loss(
            embedding,
            margin=self.temporal_endpoint_margin,
        )
        temporal_endpoint_weight = self._current_temporal_endpoint_weight()
        weighted_temporal_endpoint = temporal_endpoint * temporal_endpoint_weight
        temporal_contrast, temporal_contrast_stats = temporal_change_aware_contrast_loss(
            embedding_map,
            targets,
            masks,
            source_names=self.temporal_contrast_sources,
            margin=self.temporal_contrast_margin,
            change_z=self.temporal_contrast_change_z,
            stable_z=self.temporal_contrast_stable_z,
        )
        temporal_contrast_weight = self._current_temporal_contrast_weight()
        weighted_temporal_contrast = temporal_contrast * temporal_contrast_weight
        supervised_change, supervised_change_stats = supervised_change_alignment_loss(
            embedding_map,
            supervised_labels,
            supervised_label_masks,
            tasks=self.supervised_change_tasks,
            pos_margin=self.supervised_change_pos_margin,
            neg_margin=self.supervised_change_neg_margin,
            pos_weight=self.supervised_change_pos_weight,
            neg_weight=self.supervised_change_neg_weight,
            hard_negative_ratio=self.supervised_change_hard_negative_ratio,
            task_weights=self.supervised_change_task_weights,
        )
        supervised_change_weight = self._current_supervised_change_weight()
        weighted_supervised_change = supervised_change * supervised_change_weight
        if self.semantic_probe is None:
            semantic_probe = embedding_map.sum() * 0.0
            semantic_probe_stats = {
                "semantic_probe_positive_pixels": semantic_probe.detach(),
                "semantic_probe_valid_pixels": semantic_probe.detach(),
            }
        else:
            semantic_probe, semantic_probe_stats = self.semantic_probe(
                embedding_map,
                supervised_labels,
                supervised_label_masks,
            )
        semantic_probe_weight = self._current_semantic_probe_weight()
        weighted_semantic_probe = semantic_probe * semantic_probe_weight
        if self.prototype_contrast is None:
            prototype_contrast = embedding_map.sum() * 0.0
            prototype_contrast_stats = {
                "prototype_contrast_positive_pixels": prototype_contrast.detach()
            }
        else:
            prototype_contrast, prototype_contrast_stats = self.prototype_contrast(
                embedding_map,
                supervised_labels,
                supervised_label_masks,
            )
        prototype_contrast_weight = self._current_prototype_contrast_weight()
        weighted_prototype_contrast = prototype_contrast * prototype_contrast_weight
        boundary_contrast_weight = self._current_boundary_contrast_weight()
        if boundary_contrast_weight == 0.0 or not self.boundary_contrast_tasks:
            boundary_contrast = embedding_map.sum() * 0.0
            boundary_contrast_stats = {
                "boundary_contrast_pairs": boundary_contrast.detach()
            }
        else:
            boundary_contrast, boundary_contrast_stats = semantic_boundary_contrast_loss(
                embedding_map,
                supervised_labels,
                supervised_label_masks,
                tasks=self.boundary_contrast_tasks,
                band_radius=self.boundary_contrast_band_radius,
                margin=self.boundary_contrast_margin,
                max_pixels_per_task=self.boundary_contrast_max_pixels,
            )
        weighted_boundary_contrast = boundary_contrast * boundary_contrast_weight
        if self.latent_reconstruction is None:
            latent_reconstruction = embedding_map.sum() * 0.0
            latent_reconstruction_stats = {
                "latent_reconstruction_valid_pixels": latent_reconstruction.detach()
            }
        else:
            latent_reconstruction, latent_reconstruction_stats = self.latent_reconstruction(
                embedding_map,
                targets,
                masks,
            )
        latent_reconstruction_weight = self._current_latent_reconstruction_weight()
        weighted_latent_reconstruction = (
            latent_reconstruction * latent_reconstruction_weight
        )
        if self.distill is None or teacher_features is None:
            distill_cos = embedding_map.sum() * 0.0
            distill_gram = embedding_map.sum() * 0.0
            distill_stats = {}
            if self.distill is not None:
                distill_cos = distill_cos + sum(
                    p.sum() for p in self.distill.proj.parameters()
                ) * 0.0
        else:
            distill_cos, distill_gram, distill_stats = self.distill(
                output.embedding_map,
                teacher_features,
                teacher_valid,
            )
        distill_weight = self._current_distill_weight()
        weighted_distill = (
            distill_cos * distill_weight
            + distill_gram
            * self.distill_gram_weight
            * (distill_weight / self.distill_weight if self.distill_weight > 0 else 0.0)
            if self.distill is not None
            else distill_cos * 0.0
        )
        total = (
            total_recon
            + weighted_distill
            + weighted_uniformity
            + weighted_covariance
            + weighted_patch_discrimination
            + weighted_temporal_endpoint
            + weighted_temporal_contrast
            + weighted_supervised_change
            + weighted_semantic_probe
            + weighted_prototype_contrast
            + weighted_boundary_contrast
            + weighted_latent_reconstruction
        )

        result: dict[str, torch.Tensor] = {
            "total": total,
            "recon": total_recon,
            "uniformity": uniformity,
            "uniformity_weighted": weighted_uniformity,
            "uniformity_weight": torch.tensor(
                uniformity_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "covariance": covariance,
            "covariance_variance": covariance_stats["variance"],
            "covariance_covariance": covariance_stats["covariance"],
            "covariance_std_mean": covariance_stats["std_mean"],
            "covariance_std_min": covariance_stats["std_min"],
            "covariance_weighted": weighted_covariance,
            "covariance_weight": torch.tensor(
                covariance_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "patch_discrimination": patch_discrimination,
            "patch_discrimination_weighted": weighted_patch_discrimination,
            "patch_discrimination_weight": torch.tensor(
                patch_discrimination_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "temporal_endpoint": temporal_endpoint,
            "temporal_endpoint_weighted": weighted_temporal_endpoint,
            "temporal_endpoint_weight": torch.tensor(
                temporal_endpoint_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "temporal_contrast": temporal_contrast,
            "temporal_contrast_stable": temporal_contrast_stats["stable"],
            "temporal_contrast_change": temporal_contrast_stats["change"],
            "temporal_contrast_weighted": weighted_temporal_contrast,
            "temporal_contrast_weight": torch.tensor(
                temporal_contrast_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "temporal_contrast_stable_pixels": temporal_contrast_stats["stable_pixels"],
            "temporal_contrast_change_pixels": temporal_contrast_stats["change_pixels"],
            "supervised_change": supervised_change,
            "supervised_change_positive": supervised_change_stats["positive"],
            "supervised_change_negative": supervised_change_stats["negative"],
            "supervised_change_weighted": weighted_supervised_change,
            "supervised_change_weight": torch.tensor(
                supervised_change_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "supervised_change_positive_pixels": supervised_change_stats["positive_pixels"],
            "supervised_change_negative_pixels": supervised_change_stats["negative_pixels"],
            "supervised_change_valid_negative_pixels": supervised_change_stats.get(
                "valid_negative_pixels", supervised_change_stats["negative_pixels"]
            ),
            "semantic_probe": semantic_probe,
            "semantic_probe_weighted": weighted_semantic_probe,
            "semantic_probe_weight": torch.tensor(
                semantic_probe_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "semantic_probe_positive_pixels": semantic_probe_stats["semantic_probe_positive_pixels"],
            "semantic_probe_valid_pixels": semantic_probe_stats["semantic_probe_valid_pixels"],
            "prototype_contrast": prototype_contrast,
            "prototype_contrast_weighted": weighted_prototype_contrast,
            "prototype_contrast_weight": torch.tensor(
                prototype_contrast_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "boundary_contrast": boundary_contrast,
            "boundary_contrast_weighted": weighted_boundary_contrast,
            "boundary_contrast_weight": torch.tensor(
                boundary_contrast_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "latent_reconstruction": latent_reconstruction,
            "latent_reconstruction_weighted": weighted_latent_reconstruction,
            "latent_reconstruction_weight": torch.tensor(
                latent_reconstruction_weight,
                device=output.embedding.device,
                dtype=output.embedding.dtype,
            ),
            "latent_reconstruction_valid_pixels": latent_reconstruction_stats[
                "latent_reconstruction_valid_pixels"
            ],
        }
        result["distill_cosine"] = distill_cos
        result["distill_gram"] = distill_gram
        result["distill_weighted"] = weighted_distill
        result["distill_weight"] = torch.tensor(
            distill_weight,
            device=output.embedding.device,
            dtype=output.embedding.dtype,
        )
        for name, value in distill_stats.items():
            result[name] = value
        for name, value in supervised_change_stats.items():
            if name.startswith("supervised_change_"):
                result[name] = value
        for name, value in semantic_probe_stats.items():
            if name.startswith("semantic_probe_"):
                result[name] = value
        for name, value in prototype_contrast_stats.items():
            if name.startswith("prototype_contrast_"):
                result[name] = value
        for name, value in boundary_contrast_stats.items():
            if name.startswith("boundary_contrast_"):
                result[name] = value
        for name, value in latent_reconstruction_stats.items():
            if name.startswith("latent_reconstruction_"):
                result[name] = value
        result.update(recon_losses)
        return result
