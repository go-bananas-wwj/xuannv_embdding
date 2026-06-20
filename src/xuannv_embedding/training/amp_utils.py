from __future__ import annotations

# AMP 工具：屏蔽 NPU / CUDA / CPU 的 autocast 与 GradScaler 差异。
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch


class _NoOpScaler:
    """AMP 关闭或设备不支持时使用的占位 GradScaler。

    保持与 ``torch.cuda.amp.GradScaler`` 相同的接口：``scale``、``step``、
    ``update``、``unscale_``，但内部不执行任何缩放/反缩放操作。
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """返回未缩放的损失。"""
        return loss

    def step(self, optimizer: Any) -> None:
        """直接调用优化器 ``step``。"""
        optimizer.step()

    def update(self) -> None:
        """无操作。"""
        pass

    def unscale_(self, optimizer: Any) -> None:
        """无操作。"""
        pass


def get_autocast(device: torch.device, enabled: bool) -> Any:
    """返回适合当前设备的 autocast 上下文管理器。

    参数:
        device: 目标训练设备。
        enabled: 是否启用 AMP。为 ``False`` 时返回无操作上下文。

    返回:
        可直接用于 ``with ...:`` 的上下文管理器。
    """
    if not enabled:
        return nullcontext()

    if device.type == "npu":
        try:
            import torch_npu  # noqa: F401

            return torch_npu.npu.amp.autocast(enabled=True)
        except ImportError:
            raise RuntimeError("device 为 npu 但未安装 torch_npu，无法启用 AMP")

    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)

    # CPU 或其它设备：torch.autocast 自 2.0+ 起支持 cpu，但可能未实现全部 dtype。
    try:
        return torch.autocast(device_type=device.type, enabled=True)
    except Exception:
        return nullcontext()


@contextmanager
def _autocast_ctx(enabled: bool, device_type: str) -> Iterator[None]:
    """在 ``get_autocast`` 返回 torch.autocast 时使用的兼容上下文。"""
    with torch.autocast(device_type=device_type, enabled=enabled):
        yield


def get_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """返回适合当前设备的 GradScaler。

    参数:
        device: 目标训练设备。
        enabled: 是否启用 AMP。为 ``False`` 或非 CUDA/NPU 设备时返回 no-op scaler。

    返回:
        ``GradScaler`` 实例或兼容的 no-op scaler。
    """
    if not enabled:
        return _NoOpScaler()

    if device.type == "npu":
        try:
            import torch_npu  # noqa: F401

            return torch_npu.npu.amp.GradScaler()
        except ImportError:
            raise RuntimeError("device 为 npu 但未安装 torch_npu，无法启用 AMP")

    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()

    return _NoOpScaler()
