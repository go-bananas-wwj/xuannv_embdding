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


def relation_distillation_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """关系蒸馏（Contrastive Representation Distillation 风格）。

    对学生与教师的场景级嵌入分别计算 pairwise cosine similarity，
    用 KL 散度约束两者的相似度分布一致，从而传递样本间的结构关系。

    Args:
        student_emb: 学生场景嵌入 ``[B, D]`` 或 ``[B, T, D]``。
        teacher_emb: 教师场景嵌入 ``[B, D]``。
        temperature: 温度系数。

    Returns:
        标量 KL 损失。
    """
    if student_emb.dim() == 3:
        # 对时间维度取平均得到 patch 级表示。
        student_emb = student_emb.mean(dim=1)

    student_emb = F.normalize(student_emb, p=2, dim=1)
    teacher_emb = F.normalize(teacher_emb, p=2, dim=1)

    s_sim = student_emb @ student_emb.t() / temperature
    t_sim = teacher_emb @ teacher_emb.t() / temperature

    # 双向 KL，数值更稳定。
    loss_st = F.kl_div(
        F.log_softmax(s_sim, dim=1),
        F.softmax(t_sim.detach(), dim=1),
        reduction="batchmean",
    )
    loss_ts = F.kl_div(
        F.log_softmax(t_sim, dim=1),
        F.softmax(s_sim.detach(), dim=1),
        reduction="batchmean",
    )
    return (loss_st + loss_ts) / 2.0


class TotalLoss(nn.Module):
    """AEF 训练总损失：加权重建损失 + batch uniformity 损失 + 可选蒸馏损失。"""

    def __init__(
        self,
        target_cfg: dict[str, dict],
        distill_weight: float = 0.0,
        distill_months: int = 12,
        relation_weight: float = 0.0,
        relation_temperature: float = 0.5,
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
            distill_weight: 蒸馏损失权重；为 0 时不启用蒸馏。
            distill_months: 蒸馏监督的月度数量，通常对应 2025 年 1-12 月。
            relation_weight: 关系蒸馏损失权重；为 0 时不启用。
            relation_temperature: 关系蒸馏温度系数。
        """
        super().__init__()
        self.target_cfg = target_cfg
        self.distill_weight = distill_weight
        self.distill_months = distill_months
        self.relation_weight = relation_weight
        self.relation_temperature = relation_temperature

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        teacher_embedding_map: torch.Tensor | None = None,
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

        # 蒸馏损失：在学生 base embedding map（高分辨率融合前、vMF 瓶颈前）
        # 与 AEF 2025 年度教师 embedding 之间计算逐像素 L2 归一化后的 MSE。
        distill = torch.tensor(
            0.0, device=output.embedding.device, dtype=output.embedding.dtype
        )
        if (
            teacher_embedding_map is not None
            and self.distill_weight > 0.0
            and getattr(output, "base_embedding_map", None) is not None
        ):
            student = output.base_embedding_map[:, : self.distill_months]  # (B, M, D, H, W)
            B, M, D, H, W = student.shape
            # 教师为年度 embedding，复制到 M 个月。
            teacher = teacher_embedding_map[:, None, :, :, :].expand(-1, M, -1, -1, -1)
            student_norm = F.normalize(
                student.reshape(B * M, D, H, W), p=2, dim=1
            ).reshape(B, M, D, H, W)
            teacher_norm = F.normalize(
                teacher.reshape(B * M, -1, H, W), p=2, dim=1
            ).reshape(B, M, -1, H, W)
            # 若教师通道数与学生不同，可在此处投影；本项目均为 64，无需投影。
            distill = F.mse_loss(student_norm, teacher_norm)
            weighted_distill = self.distill_weight * distill
        else:
            weighted_distill = torch.tensor(
                0.0, device=output.embedding.device, dtype=output.embedding.dtype
            )

        # 关系蒸馏：保持 batch 内样本相似度结构与教师一致。
        relation = torch.tensor(
            0.0, device=output.embedding.device, dtype=output.embedding.dtype
        )
        if (
            teacher_embedding_map is not None
            and self.relation_weight > 0.0
            and getattr(output, "embedding", None) is not None
        ):
            # 教师为年度空间 embedding map，先在空间上池化到场景级。
            teacher_scene = teacher_embedding_map.mean(dim=(-2, -1))  # (B, D)
            relation = relation_distillation_loss(
                output.embedding,
                teacher_scene,
                temperature=self.relation_temperature,
            )
            weighted_relation = self.relation_weight * relation
        else:
            weighted_relation = torch.tensor(
                0.0, device=output.embedding.device, dtype=output.embedding.dtype
            )

        total = total_recon + uniformity + weighted_distill + weighted_relation

        result: dict[str, torch.Tensor] = {
            "total": total,
            "recon": total_recon,
            "uniformity": uniformity,
            "distill": distill,
            "weighted_distill": weighted_distill,
            "relation": relation,
            "weighted_relation": weighted_relation,
        }
        result.update(recon_losses)
        return result
