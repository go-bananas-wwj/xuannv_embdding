from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from downstreams.scripts.build_patch_meta import build_region_meta


def _write_mask(path: Path, data: np.ndarray, *, crs: str | None, transform: rasterio.Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_build_region_meta_falls_back_to_manifest_when_mask_has_no_crs(tmp_path: Path) -> None:
    """Harbin masks lost their CRS; metadata should be recovered from the manifest patch file."""
    region_root = tmp_path / "harbin"
    patch_dir = region_root / "patches" / "s2"
    mask_dir = region_root / "labels" / "construction" / "masks"
    patch_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"

    # Source patch is georeferenced in UTM zone 52N
    patch_transform = rasterio.Affine.translation(306797.6449433508, 5069852.38061053) * rasterio.Affine.scale(10, -10)
    patch = np.zeros((128, 128), dtype=np.uint8)
    _write_mask(patch_dir / f"s2_20250101_{patch_id}.tif", patch, crs="EPSG:32652", transform=patch_transform)

    # Mask is not georeferenced (the bug)
    mask_transform = rasterio.Affine.identity()
    mask = np.zeros((128, 128), dtype=np.uint8)
    _write_mask(mask_dir / f"{patch_id}_202512.tif", mask, crs=None, transform=mask_transform)

    manifest = [
        {
            "patch_id": patch_id,
            "s2": [f"patches/s2/s2_20250101_{patch_id}.tif"],
        }
    ]
    manifest_path = region_root / "manifest_labeled.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    meta = build_region_meta([mask_dir], manifest_path=manifest_path, region_root=region_root)

    assert patch_id in meta
    assert meta[patch_id]["crs"] == "EPSG:32652"
    bbox = meta[patch_id]["bbox"]
    assert pytest.approx(bbox[0], 1e-6) == 306797.6449433508
    assert pytest.approx(bbox[1], 1e-6) == 5068572.38061053
    assert pytest.approx(bbox[2], 1e-6) == 308077.6449433508
    assert pytest.approx(bbox[3], 1e-6) == 5069852.38061053
    assert meta[patch_id]["shape"] == (128, 128)
