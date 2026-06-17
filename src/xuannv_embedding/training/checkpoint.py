from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# checkpoint 存取工具：只保存 state_dict，不保存完整模型对象。


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """保存训练状态到 checkpoint 文件。

    参数:
        path: checkpoint 文件路径。
        model: 模型实例，保存 ``state_dict()``。
        optimizer: 优化器实例，保存 ``state_dict()``。
        scheduler: 学习率调度器实例，可选。
        epoch: 当前 epoch 编号。
        metrics: 需要一起保存的指标字典。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """从 checkpoint 文件恢复训练状态。

    参数:
        path: checkpoint 文件路径。
        model: 模型实例，将被 ``load_state_dict``。
        optimizer: 优化器实例，可选；提供时恢复状态。
        scheduler: 学习率调度器实例，可选；提供时恢复状态。
        device: 加载目标设备，默认 ``cpu``。

    返回:
        checkpoint 字典，包含 ``epoch``、``metrics`` 等字段。
    """
    path = Path(path)
    # 使用 weights_only=True 防止不受信任 checkpoint 的 pickle 反序列化漏洞，
    # 同时避免 PyTorch 2.6+ 的安全警告。
    state = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    return state
