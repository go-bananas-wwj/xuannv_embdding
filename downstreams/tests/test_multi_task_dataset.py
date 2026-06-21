from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from downstreams.data.multi_task_dataset import (
    MultiTaskEmbeddingDataset,
    collate_embeddings,
)


def _make_fixture(
    tmp_path: Path,
    patch_id: str = "patch_000000",
    months: list[str] = ("202512", "202605"),
    shape: tuple[int, int, int] = (64, 16, 16),
) -> tuple[Path, Path]:
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)
    (emb_root / patch_id).mkdir(parents=True)

    for month in months:
        emb = torch.randn(*shape)
        torch.save(emb, emb_root / patch_id / f"{month}_embedding_map.pt")

    mask = np.zeros((shape[-2], shape[-1]), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=shape[-2],
        width=shape[-1],
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    return emb_root, label_root


def test_single_month_dataset(tmp_path: Path) -> None:
    """单时相数据集返回 (B, D, H, W)。"""
    emb_root, label_root = _make_fixture(tmp_path, months=["202605"])
    ds = MultiTaskEmbeddingDataset(
        emb_root,
        label_root,
        ["patch_000000"],
        months="202605",
        bitemporal=False,
    )
    sample = ds[0]
    assert sample["embedding_map"].shape == (64, 16, 16)
    assert sample["mask"].shape == (16, 16)

    batch = collate_embeddings([ds[0], ds[0]])
    assert batch["embedding_map"].shape == (2, 64, 16, 16)
    assert batch["mask"].shape == (2, 16, 16)


def test_bitemporal_with_diff(tmp_path: Path) -> None:
    """双时相 + 差分拼接返回 (B, 3*D, H, W)。"""
    emb_root, label_root = _make_fixture(tmp_path, months=["202512", "202605"])
    ds = MultiTaskEmbeddingDataset(
        emb_root,
        label_root,
        ["patch_000000"],
        months=["202512", "202605"],
        bitemporal=True,
        include_diff=True,
    )
    sample = ds[0]
    assert sample["embedding_map"].shape == (3 * 64, 16, 16)
    assert sample["mask"].shape == (16, 16)

    batch = collate_embeddings([sample, sample])
    assert batch["embedding_map"].shape == (2, 3 * 64, 16, 16)
    assert batch["mask"].shape == (2, 16, 16)


def test_bitemporal_without_diff(tmp_path: Path) -> None:
    """双时相不包含差分时返回 (B, 2*D, H, W)。"""
    emb_root, label_root = _make_fixture(tmp_path, months=["202512", "202605"])
    ds = MultiTaskEmbeddingDataset(
        emb_root,
        label_root,
        ["patch_000000"],
        months=["202512", "202605"],
        bitemporal=True,
        include_diff=False,
    )
    sample = ds[0]
    assert sample["embedding_map"].shape == (2 * 64, 16, 16)
