from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# 重建损失 + batch uniformity 损失模块。


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

class TotalLoss(nn.Module):
    """AEF 训练总损失：加权重建损失 + 表征正则项。"""

    def __init__(
        self,
        target_cfg: dict[str, dict],
        uniformity_weight: float = 1.0,
        uniformity_warmup_epochs: int = 0,
        uniformity_temperature: float = 2.0,
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
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于 uniformity 权重 warmup。"""
        self.current_epoch = int(epoch)

    def _current_uniformity_weight(self) -> float:
        if self.uniformity_weight == 0.0:
            return 0.0
        if self.uniformity_warmup_epochs <= 0:
            return self.uniformity_weight
        progress = min(1.0, float(self.current_epoch + 1) / self.uniformity_warmup_epochs)
        return self.uniformity_weight * progress

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

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        supervised_labels: dict[str, torch.Tensor] | None = None,
        supervised_label_masks: dict[str, torch.Tensor] | None = None,
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
        total_recon = torch.tensor(
            0.0, device=output.embedding.device, dtype=output.embedding.dtype
        )

        for name, cfg in self.target_cfg.items():
            pred = output.reconstructions[name]
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
            output.embedding,
            temperature=self.uniformity_temperature,
        )
        uniformity_weight = self._current_uniformity_weight()
        weighted_uniformity = uniformity * uniformity_weight
        temporal_endpoint = temporal_endpoint_separation_loss(
            output.embedding,
            margin=self.temporal_endpoint_margin,
        )
        temporal_endpoint_weight = self._current_temporal_endpoint_weight()
        weighted_temporal_endpoint = temporal_endpoint * temporal_endpoint_weight
        temporal_contrast, temporal_contrast_stats = temporal_change_aware_contrast_loss(
            output.embedding_map,
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
            output.embedding_map,
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
        total = (
            total_recon
            + weighted_uniformity
            + weighted_temporal_endpoint
            + weighted_temporal_contrast
            + weighted_supervised_change
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
        }
        for name, value in supervised_change_stats.items():
            if name.startswith("supervised_change_"):
                result[name] = value
        result.update(recon_losses)
        return result
