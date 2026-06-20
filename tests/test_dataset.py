from __future__ import annotations

# Dataset、collate 与 transforms 的单元测试
from pathlib import Path

import pytest
import torch

from xuannv_embedding.data.builder import build_dataloader
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.data.transforms import parse_timestamp_from_filename

MANIFEST_PATH = Path("/data/xuannv_embedding/processed/harbin/manifest.json")
STATISTICS_DIR = Path("/data/xuannv_embedding/statistics/harbin")

pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists() or not STATISTICS_DIR.exists(),
    reason="Harbin manifest 或统计量尚未生成，跳过需要真实数据的测试",
)


def test_parse_timestamp_from_filename() -> None:
    """文件名解析应返回 ``YYYYMM`` 整数。"""
    assert parse_timestamp_from_filename("s2_20250129_patch_000000.tif") == 202501


def test_dataset_loads_manifest() -> None:
    """使用真实 manifest 创建 dataset，应能正确返回样本并处理缺失 source。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2", "s1", "landsat"],
        max_patches=2,
    )

    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["patch_id"].startswith("patch_")
    assert "s2" in sample["source_frames"]

    s2_frames = sample["source_frames"]["s2"]
    assert s2_frames.ndim == 4
    assert s2_frames.shape[1] == 12  # Sentinel-2 12 波段
    assert s2_frames.shape[2] == 128
    assert s2_frames.shape[3] == 128
    assert s2_frames.dtype == torch.float32

    assert sample["source_masks"]["s2"].sum().item() >= 1.0
    assert sample["timestamps"]["s2"][0].item() == 202501

    # s1/landsat 可能缺失，缺失时应返回空张量
    for source in ("s1", "landsat"):
        frames = sample["source_frames"][source]
        assert frames.ndim == 4
        if frames.shape[0] > 0:
            assert frames.shape[1] in (2, 7)
            assert frames.shape[2] == 128
            assert frames.shape[3] == 128


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

    max_t = max(len(sample["timestamps"]["s2"]) for sample in batch)
    assert collated["source_frames"]["s2"].shape == (2, max_t, 12, 128, 128)
    assert collated["source_masks"]["s2"].shape == (2, max_t)
    assert collated["timestamps"]["s2"].shape == (2, max_t)
    assert len(collated["patch_ids"]) == 2


def test_dataset_normalizes_with_statistics() -> None:
    """dataset 应读取统计量并对 S2 数据进行归一化，数值应脱离原始反射率范围。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2"],
        max_patches=1,
    )
    sample = dataset[0]
    s2_frames = sample["source_frames"]["s2"]

    # 原始 S2 反射率通常在 [0, 10000+]；归一化后应大致在 [-10, 10] 内
    assert s2_frames.abs().max().item() < 10.0


def test_build_dataloader() -> None:
    """builder 应能构造 DataLoader 并产生合法 batch。"""
    from xuannv_embedding.config import DataConfig

    cfg = DataConfig(
        root=MANIFEST_PATH.parent,
        region="harbin",
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        max_patches=2,
        batch_size=2,
        num_workers=0,
        patch_size=128,
        sources=["s2"],
    )
    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    assert batch["source_frames"]["s2"].shape[0] == 2
    assert batch["source_frames"]["s2"].shape[2] == 12
    assert batch["source_frames"]["s2"].shape[3] == 128
    assert batch["source_frames"]["s2"].shape[4] == 128
    t = batch["source_frames"]["s2"].shape[1]
    assert batch["source_masks"]["s2"].shape == (2, t)
    assert batch["timestamps"]["s2"].shape == (2, t)
