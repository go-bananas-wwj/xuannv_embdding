from __future__ import annotations

# Dataset、collate 与 transforms 的单元测试
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

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


def _write_tiff(path: Path, array: np.ndarray) -> None:
    """将数组写入为单/多波段 GeoTIFF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=Affine.identity(),
    ) as dst:
        dst.write(array)


def test_parse_timestamp_from_filename() -> None:
    """文件名解析应返回 ``YYYYMM`` 整数。"""
    assert parse_timestamp_from_filename("s2_20250129_patch_000000.tif") == 202501


def test_dataset_loads_manifest() -> None:
    """使用真实 manifest 创建 dataset，应能正确返回月度 binned 样本并处理缺失 source。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2", "s1", "landsat", "worldcover"],
        max_patches=2,
    )

    assert len(dataset) == 2
    assert dataset.num_months == 17
    assert dataset.month_bins[0] == 202501
    assert dataset.month_bins[-1] == 202605

    sample = dataset[0]
    assert sample["patch_id"].startswith("patch_")
    assert "s2" in sample["source_frames"]

    s2_frames = sample["source_frames"]["s2"]
    assert s2_frames.ndim == 4
    assert s2_frames.shape[0] == dataset.num_months
    assert s2_frames.shape[1] == 12  # Sentinel-2 12 波段
    assert s2_frames.shape[2] == 128
    assert s2_frames.shape[3] == 128
    assert s2_frames.dtype == torch.float32

    assert sample["source_masks"]["s2"].shape == (dataset.num_months,)
    assert sample["source_masks"]["s2"].sum().item() >= 1.0
    assert torch.equal(
        sample["timestamps"]["s2"],
        torch.tensor(dataset.month_bins, dtype=torch.long),
    )

    # s1/landsat 可能缺失，缺失时应返回月度全 0 张量
    for source in ("s1", "landsat"):
        frames = sample["source_frames"][source]
        assert frames.ndim == 4
        assert frames.shape[0] == dataset.num_months
        if frames.shape[1] > 0:
            assert frames.shape[1] in (2, 7)
            assert frames.shape[2] == 128
            assert frames.shape[3] == 128

    # worldcover 是静态 target-only 源，应被复制到所有月度 bin
    worldcover = sample["source_frames"]["worldcover"]
    assert worldcover.shape[0] == dataset.num_months
    assert worldcover.shape[2] == 128
    assert worldcover.shape[3] == 128
    assert torch.equal(
        sample["timestamps"]["worldcover"],
        torch.tensor(dataset.month_bins, dtype=torch.long),
    )


def test_collate_fn() -> None:
    """collate 应将月度 binned 样本堆叠为 (B, T_month, C, H, W)。"""
    dataset = MonthlyEmbeddingDataset(
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        sources=["s2"],
        max_patches=2,
    )
    batch = [dataset[0], dataset[1]]
    collated = collate_fn(batch)

    assert collated["source_frames"]["s2"].shape == (2, 17, 12, 128, 128)
    assert collated["source_masks"]["s2"].shape == (2, 17)
    assert collated["timestamps"].shape == (2, 17)
    assert torch.equal(collated["timestamps"][0], collated["timestamps"][1])
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
    """builder 应能构造 DataLoader 并产生月度合法 batch。"""
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

    assert batch["source_frames"]["s2"].shape == (2, 17, 12, 128, 128)
    assert batch["source_masks"]["s2"].shape == (2, 17)
    assert batch["timestamps"].shape == (2, 17)


def test_monthly_binning_with_synthetic_data(tmp_path: Path) -> None:
    """dataset 应按月份 bin 同一月份的多个观测，并对静态 worldcover 复制到各月。"""
    root = tmp_path / "processed" / "test"
    root.mkdir(parents=True)
    stats_dir = tmp_path / "statistics" / "test"
    stats_dir.mkdir(parents=True)

    s2_dir = root / "patches" / "s2"
    wc_dir = root / "labels" / "worldcover"

    # 202501 两个观测，值分别为 0 和 2；202502 一个观测，值为 5。
    _write_tiff(s2_dir / "s2_20250102_patch_000000.tif", np.zeros((1, 4, 4), dtype=np.float32))
    _write_tiff(
        s2_dir / "s2_20250115_patch_000000.tif",
        np.full((1, 4, 4), 2.0, dtype=np.float32),
    )
    _write_tiff(
        s2_dir / "s2_20250203_patch_000000.tif",
        np.full((1, 4, 4), 5.0, dtype=np.float32),
    )

    # 静态 worldcover 标签
    _write_tiff(
        wc_dir / "worldcover_20230101_patch_000000.tif",
        np.full((1, 4, 4), 7, dtype=np.uint8),
    )

    manifest = [
        {
            "patch_id": "patch_000000",
            "s2": [
                "patches/s2/s2_20250102_patch_000000.tif",
                "patches/s2/s2_20250115_patch_000000.tif",
                "patches/s2/s2_20250203_patch_000000.tif",
            ],
            "worldcover": ["labels/worldcover/worldcover_20230101_patch_000000.tif"],
        }
    ]
    import json

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=stats_dir,
        sources=["s2", "worldcover"],
        num_months=2,
        patch_size=4,
    )

    sample = dataset[0]
    s2_frames = sample["source_frames"]["s2"]
    assert s2_frames.shape == (2, 1, 4, 4)
    assert torch.allclose(s2_frames[0], torch.full((1, 4, 4), 1.0))
    assert torch.allclose(s2_frames[1], torch.full((1, 4, 4), 5.0))
    assert torch.equal(
        sample["timestamps"]["s2"],
        torch.tensor([202501, 202502], dtype=torch.long),
    )

    wc_frames = sample["source_frames"]["worldcover"]
    assert wc_frames.shape == (2, 1, 4, 4)
    assert torch.allclose(wc_frames, torch.full((2, 1, 4, 4), 7.0))
    assert torch.equal(
        sample["timestamps"]["worldcover"],
        torch.tensor([202501, 202502], dtype=torch.long),
    )


def test_mixed_region_manifest_uses_region_statistics(tmp_path: Path) -> None:
    """混合区域 manifest 应保留 region 字段，并按 region 选择统计量。"""
    processed = tmp_path / "processed"
    manifest_root = processed / "v2"
    manifest_root.mkdir(parents=True)

    for region, value in (("haidian", 10.0), ("harbin", 20.0)):
        _write_tiff(
            processed / region / "patches" / "s2" / f"s2_20251201_patch_000000.tif",
            np.full((1, 4, 4), value, dtype=np.float32),
        )
        stats_dir = tmp_path / "statistics" / region
        stats_dir.mkdir(parents=True)
        stats_dir.joinpath("s2_stats.json").write_text(
            '{"mean": [%.1f], "std": [2.0]}' % value,
            encoding="utf-8",
        )

    manifest = [
        {
            "region": "haidian",
            "patch_id": "haidian_patch_000000",
            "source_patch_id": "patch_000000",
            "s2": ["../haidian/patches/s2/s2_20251201_patch_000000.tif"],
        },
        {
            "region": "harbin",
            "patch_id": "harbin_patch_000000",
            "source_patch_id": "patch_000000",
            "s2": ["../harbin/patches/s2/s2_20251201_patch_000000.tif"],
        },
    ]
    import json

    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=tmp_path / "statistics" / "fallback",
        statistics_dirs_by_region={
            "haidian": tmp_path / "statistics" / "haidian",
            "harbin": tmp_path / "statistics" / "harbin",
        },
        sources=["s2"],
        num_months=1,
        ref_year=2025,
        ref_month=12,
        patch_size=4,
    )

    assert dataset[0]["patch_id"] == "haidian_patch_000000"
    assert dataset[1]["patch_id"] == "harbin_patch_000000"
    assert torch.allclose(dataset[0]["source_frames"]["s2"], torch.zeros(1, 1, 4, 4))
    assert torch.allclose(dataset[1]["source_frames"]["s2"], torch.zeros(1, 1, 4, 4))

    harbin_only = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=tmp_path / "statistics" / "fallback",
        statistics_dirs_by_region={
            "haidian": tmp_path / "statistics" / "haidian",
            "harbin": tmp_path / "statistics" / "harbin",
        },
        sources=["s2"],
        num_months=1,
        ref_year=2025,
        ref_month=12,
        patch_size=4,
        region_filter="harbin",
    )
    assert len(harbin_only) == 1
    assert harbin_only[0]["patch_id"] == "harbin_patch_000000"
