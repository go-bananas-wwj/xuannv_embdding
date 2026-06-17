from __future__ import annotations

# Dataset、collate 与 transforms 的单元测试
from pathlib import Path

import torch

from xuannv_embedding.data.builder import build_dataloader
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.data.transforms import parse_timestamp_from_filename

MANIFEST_PATH = Path("/data/xuannv_embedding/processed/harbin/scenes/manifest.json")
STATISTICS_DIR = Path("/data/xuannv_embedding/statistics/harbin")


def test_parse_timestamp_from_filename() -> None:
    """文件名解析应返回 ``YYYYMM`` 整数。"""
    assert parse_timestamp_from_filename("s2_20250129_p000_r000.tif") == 202501


def test_dataset_loads_manifest() -> None:
    """使用真实 manifest 创建 dataset，应能正确返回样本并处理缺失 source。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2", "s1"],
        max_patches=2,
    )

    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["patch_id"].startswith("20250129_")
    assert "s2" in sample["source_frames"]

    s2_frames = sample["source_frames"]["s2"]
    assert s2_frames.ndim == 4
    assert s2_frames.shape[0] == 1  # 当前每个 patch 仅 1 个时相
    assert s2_frames.shape[1] == 12  # Sentinel-2 12 波段
    assert s2_frames.shape[2] == 256
    assert s2_frames.shape[3] == 256
    assert s2_frames.dtype == torch.float32

    assert sample["source_masks"]["s2"].sum().item() == 1.0
    assert sample["timestamps"]["s2"][0].item() == 202501

    # s1 在当前数据中不存在，应为空张量
    s1_frames = sample["source_frames"]["s1"]
    assert s1_frames.shape[0] == 0


def test_collate_fn() -> None:
    """collate 应对不同长度的时序进行正确补齐。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2"],
        max_patches=2,
    )
    batch = [dataset[0], dataset[1]]
    collated = collate_fn(batch)

    assert collated["source_frames"]["s2"].shape == (2, 1, 12, 256, 256)
    assert collated["source_masks"]["s2"].shape == (2, 1)
    assert collated["timestamps"]["s2"].shape == (2, 1)
    assert len(collated["patch_ids"]) == 2


def test_build_dataloader() -> None:
    """builder 应能构造 DataLoader 并产生合法 batch。"""
    from xuannv_embedding.config import DataConfig

    cfg = DataConfig(
        root=MANIFEST_PATH.parent,
        region="harbin",
        manifest_path=MANIFEST_PATH,
        num_samples=2,
        max_patches=2,
        batch_size=2,
        num_workers=0,
        sources=["s2"],
    )
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert batch["source_frames"]["s2"].shape == (2, 1, 12, 256, 256)
    assert batch["source_masks"]["s2"].shape == (2, 1)
    assert batch["timestamps"]["s2"].shape == (2, 1)
