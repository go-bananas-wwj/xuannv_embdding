#!/usr/bin/env python3
from __future__ import annotations

# DDP / 单卡训练入口。
import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.trainer import Trainer
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 可选参数列表，用于测试；为 ``None`` 时解析 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="AEF 月度地理嵌入训练入口")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="待恢复的 checkpoint 路径（恢复 optimizer/scheduler/epoch）",
    )
    parser.add_argument(
        "--init-from",
        dest="init_from",
        type=str,
        default=None,
        help="仅加载模型权重的 checkpoint 路径（用于阶段二从阶段一初始化）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖自动设备选择，例如 cpu / cuda:0 / npu:0",
    )
    return parser.parse_args(argv)


def setup_distributed() -> tuple[bool, int]:
    """若由 ``torchrun`` 启动，则初始化分布式进程组。

    返回:
        (是否进入分布式模式, local_rank)。
    """
    if "RANK" not in os.environ:
        return False, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "gloo"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            backend = "hccl"
    except ImportError:
        pass

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if backend == "hccl":
        torch.npu.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, local_rank


def _set_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_device(args_device: str | None, is_distributed: bool) -> torch.device:
    """根据用户参数与分布式环境解析训练设备。

    分布式模式下固定按 ``LOCAL_RANK`` 选择设备，忽略用户传入的 ``--device``，
    避免所有 rank 被绑定到同一设备导致 DDP 失败。
    """
    if not is_distributed:
        if args_device is not None:
            return get_device(args_device)
        return get_device()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = dist.get_backend() if dist.is_initialized() else "gloo"
    if backend == "hccl":
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _build_loader(
    cfg: Config,
    target_heads: dict[str, dict[str, Any]],
    split: str,
    is_distributed: bool,
) -> DataLoader:
    """构造训练或验证 DataLoader，内置 ``prepare_batch`` 转换。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=cfg.data.manifest_path,
        statistics_dir=cfg.data.statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        max_patches=cfg.data.max_patches,
        num_months=cfg.model.num_months,
        teacher_embedding_root=cfg.data.teacher_embedding_root,
        region=cfg.data.region,
        cache_dir=cfg.data.cache_dir,
        augment=cfg.data.augment and split == "train",
        sensor_dropout_prob=cfg.data.sensor_dropout_prob,
        temporal_dropout_prob=cfg.data.temporal_dropout_prob,
        noise_std=cfg.data.noise_std,
    )

    sampler = None
    if is_distributed:
        # 验证集也 drop_last，避免不同 rank 的 batch 数不一致导致 DDP 潜在 hang。
        sampler = DistributedSampler(
            dataset,
            shuffle=(split == "train"),
            drop_last=True,
        )

    def training_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        collated = collate_fn(batch)
        return prepare_batch(collated, target_heads)

    shuffle = (split == "train") and (sampler is None)
    # num_workers=0 时 pin_memory 在 NPU 上可能带来额外开销，且收益有限。
    pin_memory = cfg.data.num_workers > 0

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=training_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )


def _init_model_from_checkpoint(
    model: nn.Module,
    init_from: str,
    is_main_process: bool,
) -> None:
    """仅加载模型权重，用于阶段二从阶段一 checkpoint 初始化。

    不恢复 optimizer、scheduler 与 epoch；阶段二使用全新的训练状态。
    """
    state = torch.load(init_from, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)

    if is_main_process:
        logger.info("从 %s 初始化模型权重", init_from)
        if missing:
            logger.info("缺失 keys (%d): %s ...", len(missing), missing[:5])
        if unexpected:
            logger.info("多余 keys (%d): %s ...", len(unexpected), unexpected[:5])


def main() -> None:
    """训练入口主函数。"""
    args = parse_args()
    is_distributed, local_rank = setup_distributed()

    cfg = Config.from_yaml(args.config)
    _set_seed(cfg.experiment.seed)

    device = _resolve_device(args.device, is_distributed)
    if is_distributed and device.type == "cuda":
        torch.cuda.set_device(local_rank)

    if not is_distributed or dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        logger.info("实验名称: %s", cfg.experiment.name)
        logger.info("配置文件: %s", Path(args.config).resolve())
        logger.info("训练设备: %s", device)
        logger.info("分布式模式: %s", is_distributed)

    # 构造 AEFModel 所需的 target_heads: {name -> (kind, channels)}。
    aef_target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=aef_target_heads,
        stem_dim=cfg.model.stem_dim,
        num_months=cfg.model.num_months,
        stp=cfg.model.stp,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )

    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume 与 --init-from 不能同时使用")

    is_main = not is_distributed or dist.get_rank() == 0
    if args.init_from is not None:
        _init_model_from_checkpoint(model, args.init_from, is_main)

    # 构造 TotalLoss 所需的 target_cfg。
    loss_type_map = {"continuous": "l1", "categorical": "ce"}
    target_cfg = {
        name: {
            "loss_type": loss_type_map[head_cfg["loss_type"]],
            "channels": head_cfg["channels"],
            "weight": head_cfg.get("weight", 1.0),
        }
        for name, head_cfg in cfg.model.target_heads.items()
    }
    criterion = TotalLoss(
        target_cfg,
        distill_weight=cfg.training.distill_weight,
        distill_months=cfg.training.distill_months,
        relation_weight=cfg.training.relation_weight,
        relation_temperature=cfg.training.relation_temperature,
    )

    train_loader = _build_loader(
        cfg,
        cfg.model.target_heads,
        split="train",
        is_distributed=is_distributed,
    )
    val_loader = _build_loader(
        cfg,
        cfg.model.target_heads,
        split="val",
        is_distributed=is_distributed,
    )

    train_sampler = (
        train_loader.sampler
        if isinstance(train_loader.sampler, DistributedSampler)
        else None
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        train_sampler=train_sampler,
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.fit()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
