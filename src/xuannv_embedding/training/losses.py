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

    if loss_type == "l1":
        # 逐元素 L1，然后在通道维度取平均，得到 [B, H, W]。
        loss = F.l1_loss(pred, target, reduction="none").mean(dim=1)
    elif loss_type == "ce":
        # cross_entropy 输出 [B, H, W]。
        loss = F.cross_entropy(pred, target, reduction="none")
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type!r}，仅支持 'l1' 或 'ce'")

    # 将 mask 广播到 [B, H, W] 后应用。
    mask = mask.expand_as(loss)
    masked_sum = (loss * mask).sum()
    masked_count = mask.sum()
    return masked_sum / (masked_count + eps)


def batch_uniformity_loss(emb: torch.Tensor) -> torch.Tensor:
    """计算 batch 内场景级嵌入的均匀性损失。

    先将每个嵌入 L2 归一化到单位球面，再计算 pairwise squared distance 的
    非对角均值。值越大表示嵌入在球面上分布越分散。

    Args:
        emb: 场景级嵌入，形状 ``[B, D]`` 或月度 ``[B, T, D]``。

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
    diag_mask = 1.0 - torch.eye(batch_size, device=emb.device, dtype=emb.dtype)
    off_diag_count = batch_size * (batch_size - 1)
    if off_diag_count == 0:
        return torch.tensor(0.0, device=emb.device, dtype=emb.dtype)

    return (squared_dist * diag_mask).sum() / off_diag_count


class TotalLoss(nn.Module):
    """AEF 训练总损失：加权重建损失 + batch uniformity 损失。"""

    def __init__(self, target_cfg: dict[str, dict]) -> None:
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

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
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

        uniformity = batch_uniformity_loss(output.embedding)
        total = total_recon + uniformity

        result: dict[str, torch.Tensor] = {
            "total": total,
            "recon": total_recon,
            "uniformity": uniformity,
        }
        result.update(recon_losses)
        return result
