from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

# optimizer 与带 warmup 的 cosine annealing scheduler 工厂函数。


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Optimizer:
    """构造 AdamW 优化器。

    参数:
        model: 待训练模型。
        lr: 学习率。
        weight_decay: 权重衰减系数。

    返回:
        配置好的 ``AdamW`` 优化器实例。
    """
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """构造带线性 warmup 的 cosine annealing 学习率调度器。

    前 ``warmup_epochs`` 个 epoch 线性增长，之后按 cosine 退火到 0。

    参数:
        optimizer: 优化器实例。
        warmup_epochs: warmup 的 epoch 数，必须 >= 0。
        total_epochs: 总训练 epoch 数。

    返回:
        ``LambdaLR`` 调度器实例。
    """
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs 必须 >= 0，实际得到 {warmup_epochs}")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs 必须 > 0，实际得到 {total_epochs}")

    def lr_lambda(epoch: int) -> float:
        # epoch 从 0 开始计数
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # cosine annealing 阶段
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
