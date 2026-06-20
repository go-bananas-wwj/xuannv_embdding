# downstreams/tests/test_split.py
from pathlib import Path

import numpy as np
import rasterio
from downstreams.data.split import create_stratified_folds


def test_create_folds(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for i in range(10):
        mask = np.zeros((16, 16), dtype=np.uint8)
        if i % 2 == 0:
            mask[:4, :4] = 1
        with rasterio.open(
            mask_dir / f"patch_{i:06d}.tif",
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

    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    assert len(split["folds"]) == 5
    for fold in split["folds"]:
        assert len(fold["train"]) > 0
        assert len(fold["test"]) > 0
        assert set(fold["train"]) & set(fold["test"]) == set()
