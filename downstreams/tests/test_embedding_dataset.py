from pathlib import Path

import numpy as np
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
