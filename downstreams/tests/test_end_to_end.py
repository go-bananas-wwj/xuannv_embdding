from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

from downstreams.data.end_to_end_dataset import (
    EndToEndSegmentationDataset,
    JointEndToEndSegmentationDataset,
    collate_end_to_end,
)
from downstreams.heads.segmentation_head import UperNetHead
from downstreams.models.end_to_end_model import EndToEndModel


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    transform = Affine.identity()
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(count):
            dst.write(array[i], i + 1)


@pytest.fixture
def synthetic_patch(tmp_path: Path) -> tuple[Path, Path, str]:
    patch_id = "patch_000000"
    month = "202512"
    image_root = tmp_path / "patches" / "highres_optical"
    label_root = tmp_path / "labels" / "construction"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.random.rand(3, 64, 64).astype(np.float32)
    mask = (np.random.rand(64, 64) > 0.9).astype(np.uint8)

    _write_tiff(image_root / f"highres_optical_{month}01_{patch_id}.tif", image)
    _write_tiff(mask_dir / f"{patch_id}_{month}.tif", mask[None, ...])

    return image_root, label_root, patch_id


def test_end_to_end_dataset_single_month(synthetic_patch: tuple[Path, Path, str]) -> None:
    image_root, label_root, patch_id = synthetic_patch
    ds = EndToEndSegmentationDataset(
        image_root=image_root,
        label_root=label_root,
        patch_ids=[patch_id],
        months="202512",
        augment=False,
    )
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["patch_id"] == patch_id


def test_end_to_end_dataset_bitemporal(synthetic_patch: tuple[Path, Path, str]) -> None:
    image_root, label_root, patch_id = synthetic_patch
    # 创建第二个时相
    image2 = np.random.rand(3, 64, 64).astype(np.float32)
    _write_tiff(image_root / f"highres_optical_20260501_{patch_id}.tif", image2)

    ds = EndToEndSegmentationDataset(
        image_root=image_root,
        label_root=label_root,
        patch_ids=[patch_id],
        months=["202512", "202605"],
        bitemporal=True,
        include_diff=True,
        augment=False,
    )
    sample = ds[0]
    assert sample["image"].shape == (9, 64, 64)  # t1(3) + t2(3) + diff(3)
    assert sample["mask"].shape == (64, 64)


def test_end_to_end_dataset_augment_crop(synthetic_patch: tuple[Path, Path, str]) -> None:
    image_root, label_root, patch_id = synthetic_patch
    ds = EndToEndSegmentationDataset(
        image_root=image_root,
        label_root=label_root,
        patch_ids=[patch_id],
        months="202512",
        augment=True,
        crop_size=32,
    )
    sample = ds[0]
    assert sample["image"].shape[-2:] == (32, 32)
    assert sample["mask"].shape == (32, 32)


def test_collate_end_to_end() -> None:
    batch = [
        {"image": torch.randn(9, 64, 64), "mask": torch.zeros(64, 64).long(), "patch_id": "a"},
        {"image": torch.randn(9, 64, 64), "mask": torch.ones(64, 64).long(), "patch_id": "b"},
    ]
    collated = collate_end_to_end(batch)
    assert collated["image"].shape == (2, 9, 64, 64)
    assert collated["mask"].shape == (2, 64, 64)
    assert collated["patch_ids"] == ["a", "b"]


class DummyEncoder(torch.nn.Module):
    """模拟 AEFModel 输出 AEFOutput。"""

    def __init__(self, embed_dim: int = 64, num_months: int = 17) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_months = num_months

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frames: dict[str, torch.Tensor] | None = None,
        highres_masks: dict[str, torch.Tensor] | None = None,
    ) -> SimpleNamespace:
        highres = next(iter(highres_frames.values()))
        B = highres.shape[0]
        H, W = 128, 128
        return SimpleNamespace(
            embedding_map=torch.randn(B, self.num_months, self.embed_dim, H, W)
        )


def test_end_to_end_model_forward_bitemporal() -> None:
    sensor_channels = {"highres_optical_harbin": 3, "s2": 12}
    encoder = DummyEncoder(embed_dim=64)
    head = UperNetHead(in_channels=64 * 3, num_classes=2)
    model = EndToEndModel(
        encoder=encoder,
        head=head,
        highres_source_name="highres_optical_harbin",
        sensor_channels=sensor_channels,
        months=["202512", "202605"],
        include_diff=True,
    )
    image = torch.randn(2, 9, 2560, 2560)
    out = model(image)
    assert out.shape == (2, 2, 128, 128)


def test_end_to_end_model_forward_single() -> None:
    sensor_channels = {"highres_optical_harbin": 3, "s2": 12}
    encoder = DummyEncoder(embed_dim=64)
    head = UperNetHead(in_channels=64, num_classes=2)
    model = EndToEndModel(
        encoder=encoder,
        head=head,
        highres_source_name="highres_optical_harbin",
        sensor_channels=sensor_channels,
        months=["202605"],
        include_diff=False,
    )
    image = torch.randn(2, 3, 2560, 2560)
    out = model(image)
    assert out.shape == (2, 2, 128, 128)


def test_end_to_end_model_tuple_output() -> None:
    """验证 wrapper 可处理 encoder 返回 tuple 的情况。"""
    sensor_channels = {"highres_optical_harbin": 3}

    class TupleEncoder(torch.nn.Module):
        def forward(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            highres = next(iter(kwargs["highres_frames"].values()))
            B = highres.shape[0]
            emb = torch.randn(B, 17, 64, 128, 128)
            scene = torch.randn(B, 17, 64)
            return emb, scene

    encoder = TupleEncoder()
    head = UperNetHead(in_channels=64, num_classes=2)
    model = EndToEndModel(
        encoder=encoder,
        head=head,
        highres_source_name="highres_optical_harbin",
        sensor_channels=sensor_channels,
        months=["202605"],
    )
    out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 2, 128, 128)


def test_joint_end_to_end_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        regions = ["harbin", "haidian"]
        mask_dirs: dict[str, Path] = {}
        region_of: dict[str, str] = {}
        patch_ids: list[str] = []
        for region in regions:
            image_root = tmp_path / region / "patches" / "highres_optical"
            mask_dir = tmp_path / region / "labels" / "construction" / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_dirs[region] = mask_dir
            patch_id = f"{region}_patch_000000"
            patch_ids.append(patch_id)
            region_of[patch_id] = region
            image = np.random.rand(3, 32, 32).astype(np.float32)
            mask = (np.random.rand(32, 32) > 0.9).astype(np.uint8)
            _write_tiff(image_root / f"highres_optical_20251201_patch_000000.tif", image)
            _write_tiff(mask_dir / "patch_000000_202512.tif", mask[None, ...])

        ds = JointEndToEndSegmentationDataset(
            image_root=tmp_path,
            mask_dirs=mask_dirs,
            patch_ids=patch_ids,
            region_of=region_of,
            months=["202512"],
        )
        assert len(ds) == 2
        sample = ds[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["mask"].shape == (32, 32)
