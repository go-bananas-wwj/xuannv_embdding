from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def setup_ddp() -> tuple[int, int, int]:
    """初始化 NPU 上的分布式训练环境。

    返回 (rank, local_rank, world_size)。当 world_size > 1 时，使用 hccl 后端；
    单卡运行时仅设置 NPU 设备并返回 rank=0。
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 为每个 NPU 进程隔离 ACL op compiler cache，避免多进程编译缓存冲突
    os.environ["ACL_OP_COMPILER_CACHE_DIR"] = f"/tmp/acl_op_cache_{local_rank}"

    if world_size > 1:
        dist.init_process_group(backend="hccl")

    torch.npu.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
