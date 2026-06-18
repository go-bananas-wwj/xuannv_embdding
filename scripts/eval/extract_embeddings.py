#!/usr/bin/env python3
from __future__ import annotations

# 月度地理嵌入提取脚本：从训练好的 AEFModel 中导出场景级与像素级 embedding。
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.data.builder import build_dataloader
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.training.checkpoint import load_checkpoint
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 可选参数列表，用于测试；为 ``None`` 时解析 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="AEF 月度地理嵌入提取脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 路径（.pt）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 .npz 文件路径",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="月份标识，仅用于输出文件名与日志记录（当前 dataset 未按月份过滤）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖自动设备选择，例如 cpu / cuda:0 / npu:0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="推断 batch size，默认使用配置文件中的 data.batch_size",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="数据集划分，默认 train",
    )
    parser.add_argument(
        "--include-maps",
        action="store_true",
        help="同时保存像素级 embedding_map（内存占用较大）",
    )
    return parser.parse_args(argv)


def _move_tensor(value: Any, device: torch.device) -> Any:
    """将张量移动到指定设备；非张量原样返回。"""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """将 batch 中所有需要喂给模型的张量搬到 device（支持嵌套字典）。"""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {k: _move_tensor(v, device) for k, v in value.items()}
        else:
            moved[key] = _move_tensor(value, device)
    return moved


def main() -> None:
    """嵌入提取主函数。"""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = Config.from_yaml(args.config)
    device = get_device(args.device)
    logger.info("使用设备: %s", device)

    # 构造 AEFModel 所需的 target_heads: {name -> (kind, channels)}。
    aef_target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    spatial_stride = getattr(cfg.model, "spatial_stride", 1)
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=aef_target_heads,
        spatial_stride=spatial_stride,
    )

    load_checkpoint(args.checkpoint, model, device=device)
    logger.info("已加载 checkpoint: %s", args.checkpoint)

    # 支持命令行覆盖 batch_size。
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    dataloader = build_dataloader(cfg.data, split=args.split)
    logger.info("数据集样本数: %d, batch_size: %d", len(dataloader.dataset), cfg.data.batch_size)

    model.to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    all_embedding_maps: list[np.ndarray] = []
    all_patch_ids: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            prepared = prepare_batch(batch, cfg.model.target_heads, spatial_stride=spatial_stride)
            prepared = _move_batch_to_device(prepared, device)

            output = model(
                source_frames=prepared["source_frames"],
                source_masks=prepared["source_masks"],
                timestamps=prepared["timestamps"],
                highres_frames=prepared.get("highres_frames"),
                highres_masks=prepared.get("highres_masks"),
            )

            # 将结果移回 CPU 并转换为 numpy，避免占用显存/NPU 内存。
            embedding = output.embedding.detach().cpu().numpy().astype(np.float32)
            all_embeddings.append(embedding)
            all_patch_ids.extend(prepared["patch_ids"])

            if args.include_maps:
                embedding_map = output.embedding_map.detach().cpu().numpy().astype(np.float32)
                all_embedding_maps.append(embedding_map)

    embeddings = np.concatenate(all_embeddings, axis=0)
    patch_ids = np.array(all_patch_ids, dtype=object)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "embeddings": embeddings,
        "patch_ids": patch_ids,
    }
    if args.include_maps:
        embedding_maps = np.concatenate(all_embedding_maps, axis=0)
        save_kwargs["embedding_maps"] = embedding_maps

    np.savez(output_path, **save_kwargs)

    month_info = f"（月份: {args.month}）" if args.month else ""
    logger.info(
        "已保存 %d 个样本的嵌入到 %s %s",
        embeddings.shape[0],
        output_path,
        month_info,
    )
    logger.info("embeddings 形状: %s, dtype: %s", embeddings.shape, embeddings.dtype)
    if args.include_maps:
        logger.info(
            "embedding_maps 形状: %s, dtype: %s",
            embedding_maps.shape,  # type: ignore[has-type]
            embedding_maps.dtype,  # type: ignore[has-type]
        )


if __name__ == "__main__":
    main()
