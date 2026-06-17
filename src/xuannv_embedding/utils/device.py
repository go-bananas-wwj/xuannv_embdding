from __future__ import annotations

# 设备选择工具：优先 NPU，其次 CUDA，最后 CPU。
import torch


def get_device(preference: str | None = None) -> torch.device:
    """根据可用硬件返回合适的 torch 设备。

    参数:
        preference: 如果提供，直接构造对应的 ``torch.device``。

    返回:
        可用的 ``torch.device`` 实例；按 NPU -> CUDA -> CPU 顺序回退。
    """
    if preference:
        # NPU 设备类型需要 torch_npu 注册，未安装时直接构造会失败
        if preference.startswith("npu"):
            import torch_npu  # noqa: F401
        return torch.device(preference)

    # 优先检测 NPU（昇腾）
    try:
        import torch_npu  # noqa: F401

        if torch_npu.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        # torch_npu 未安装时静默回退
        pass

    # 其次检测 CUDA
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    # 默认 CPU
    return torch.device("cpu")
