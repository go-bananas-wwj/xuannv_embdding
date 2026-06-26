from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import torch
from downstreams.data.embedding_dataset import EmbeddingDataset, collate_embeddings


def test_embedding_dataset(tmp_path: Path) -> None:
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    emb = torch.randn(64, 16, 16)
    torch.save(emb, emb_root / patch_id / "202604_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202604)
    sample = ds[0]
    assert sample["embedding_map"].shape == (64, 16, 16)
    assert sample["mask"].shape == (16, 16)

    batch = collate_embeddings([sample, sample])
    assert batch["embedding_map"].shape == (2, 64, 16, 16)


def test_embedding_dataset_concat_diff(tmp_path: Path) -> None:
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    emb_t1 = torch.ones(2, 4, 4)
    emb_t2 = torch.full((2, 4, 4), 3.0)
    torch.save(emb_t1, emb_root / patch_id / "202512_embedding_map.pt")
    torch.save(emb_t2, emb_root / patch_id / "202605_embedding_map.pt")

    mask = np.zeros((4, 4), dtype=np.uint8)
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(
        emb_root,
        label_root,
        [patch_id],
        months=[202512, 202605],
        temporal_mode="concat_diff",
    )
    sample = ds[0]
    assert sample["embedding_map"].shape == (6, 4, 4)
    assert torch.equal(sample["embedding_map"][:2], emb_t1)
    assert torch.equal(sample["embedding_map"][2:4], emb_t2)
    assert torch.equal(sample["embedding_map"][4:], torch.full((2, 4, 4), 2.0))


def test_embedding_dataset_region_prefixed_embedding_root(tmp_path: Path) -> None:
    """区域混合导出的 embedding 目录带 region 前缀时，仍可用基础 patch id 读取。"""
    emb_root = tmp_path / "embeddings" / "harbin"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"
    prefixed_patch_id = "harbin_patch_000000"
    (emb_root / prefixed_patch_id).mkdir(parents=True)
    emb = torch.randn(64, 16, 16)
    torch.save(emb, emb_root / prefixed_patch_id / "202605_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202605)
    sample = ds[0]
    assert torch.equal(sample["embedding_map"], emb)


def test_embedding_dataset_joint_region_falls_back_to_source_region(tmp_path: Path) -> None:
    """construction_joint 可按 patch_id 前缀回退到 haidian/harbin sibling embedding。"""
    embedding_root = tmp_path / "embeddings"
    emb_root = embedding_root / "construction_joint"
    source_root = embedding_root / "harbin"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "harbin_patch_000159"
    (source_root / patch_id).mkdir(parents=True)
    emb = torch.randn(64, 16, 16)
    torch.save(emb, source_root / patch_id / "202605_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202605)
    sample = ds[0]
    assert torch.equal(sample["embedding_map"], emb)


def test_embedding_dataset_month_suffixed_masks_use_latest(tmp_path: Path) -> None:
    """当 mask 带 YYYYMM 后缀时，应选择最新月份 mask。"""
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    emb = torch.randn(64, 16, 16)
    torch.save(emb, emb_root / patch_id / "202605_embedding_map.pt")

    old_mask = np.zeros((16, 16), dtype=np.uint8)
    new_mask = np.ones((16, 16), dtype=np.uint8)
    for name, mask in [
        (f"{patch_id}_202512.tif", old_mask),
        (f"{patch_id}_202605.tif", new_mask),
    ]:
        with rasterio.open(
            mask_dir / name,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=mask.dtype,
            crs=None,
            transform=rasterio.Affine.identity(),
        ) as dst:
            dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202605)
    sample = ds[0]
    assert torch.equal(sample["mask"], torch.ones(16, 16, dtype=torch.long))


def test_embedding_missing_embedding(tmp_path: Path) -> None:
    """embedding 文件不存在时抛出 FileNotFoundError。"""
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)
    patch_id = "patch_000000"

    mask = np.zeros((16, 16), dtype=np.uint8)
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(tmp_path / "embeddings", label_root, [patch_id], month=202604)
    with pytest.raises(FileNotFoundError, match="embedding"):
        ds[0]


def test_embedding_missing_mask(tmp_path: Path) -> None:
    """mask 文件不存在时抛出 FileNotFoundError。"""
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    (label_root / "masks").mkdir(parents=True)
    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    torch.save(torch.randn(64, 16, 16), emb_root / patch_id / "202604_embedding_map.pt")

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202604)
    with pytest.raises(FileNotFoundError, match="mask"):
        ds[0]


def test_embedding_shape_mismatch(tmp_path: Path) -> None:
    """embedding 与 mask 尺寸不一致时抛出 ValueError。"""
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)
    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    torch.save(torch.randn(64, 8, 8), emb_root / patch_id / "202604_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202604)
    with pytest.raises(ValueError, match="尺寸不一致"):
        ds[0]


def test_embedding_augment_sync(tmp_path: Path) -> None:
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)
    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)

    # 构造左右、上下不对称的 embedding 和 mask
    emb = torch.zeros(1, 16, 16)
    emb[:, :, :8] = 1.0
    emb[:, :8, :] = emb[:, :8, :] + 2.0  # 左上=3, 右上=2, 左下=1, 右下=0
    torch.save(emb, emb_root / patch_id / "202604_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:8, :8] = 1
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202604, augment=True)
    emb_orig = emb.clone()
    mask_orig = torch.from_numpy(mask).long()

    # 水平翻转，不垂直翻转
    with patch("torch.rand", side_effect=[torch.tensor(0.6), torch.tensor(0.3)]):
        sample = ds[0]
    assert torch.equal(sample["mask"], torch.flip(mask_orig, dims=[-1]))
    assert torch.equal(sample["embedding_map"], torch.flip(emb_orig, dims=[-1]))

    # 不水平翻转，垂直翻转
    with patch("torch.rand", side_effect=[torch.tensor(0.3), torch.tensor(0.6)]):
        sample = ds[0]
    assert torch.equal(sample["mask"], torch.flip(mask_orig, dims=[-2]))
    assert torch.equal(sample["embedding_map"], torch.flip(emb_orig, dims=[-2]))
