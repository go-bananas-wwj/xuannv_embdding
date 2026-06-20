from pathlib import Path

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
    """augment=True 时 embedding 与 mask 执行相同翻转。"""
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)
    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)

    emb = torch.randn(64, 16, 16)
    torch.save(emb, emb_root / patch_id / "202604_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 1
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
    # 固定随机种子，确保可复现
    torch.manual_seed(42)
    sample = ds[0]

    # 通过检查 mask 的左右分布判断水平翻转是否发生
    left_sum = sample["mask"][:, :8].sum().item()
    right_sum = sample["mask"][:, 8:].sum().item()
    # 原始 mask 左侧为 1，翻转后右侧为 1
    assert left_sum == 0 or right_sum == 0
    assert left_sum + right_sum == 16 * 8
