#!/usr/bin/env python3
"""Prepare AlphaEarth Foundations 2025 annual embeddings for local patches.

The public AEF mosaic is stored as int8 quantized annual embeddings in EPSG:4326.
This script crops the 2025 slice for each local patch, dequantizes it, and
warps it onto the existing 128x128 patch grid so downstream probe code can read
the result as ``{month}_embedding_map.pt``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import xarray as xr
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

LOGGER = logging.getLogger(__name__)

AEF_ZARR_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"
AEF_TIME_INDEX_2025 = 8
AEF_NODATA = -128


def dequantize(values: np.ndarray) -> np.ndarray:
    """Convert AEF int8 values to float32 using the official quadratic scale."""
    x = values.astype(np.float32)
    out = ((x / 127.5) ** 2) * np.sign(x)
    out[values == AEF_NODATA] = np.nan
    return out


def load_patch_ids(label_root: Path | None, patch_ids: list[str] | None) -> list[str]:
    if patch_ids:
        return patch_ids
    if label_root is None:
        raise ValueError("Either --patch-ids or --label-root is required.")
    mask_dir = label_root / "masks"
    ids = sorted(path.stem for path in mask_dir.glob("patch_*.tif"))
    if not ids:
        raise FileNotFoundError(f"No patch masks found under {mask_dir}")
    return ids


def find_reference_raster(processed_region_root: Path, patch_id: str) -> Path:
    """Pick a local GeoTIFF that defines the target patch grid."""
    source_dirs = [
        processed_region_root / "patches" / "s2",
        processed_region_root / "patches" / "highres_optical",
        processed_region_root / "patches" / "landsat",
        processed_region_root / "patches" / "s1",
    ]
    for source_dir in source_dirs:
        matches = sorted(source_dir.glob(f"*_{patch_id}.tif"))
        matches = [path for path in matches if not path.name.endswith("_mask.tif")]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No reference GeoTIFF found for {patch_id}")


def _coord_slice(values: np.ndarray, lo: float, hi: float) -> slice:
    if values[0] <= values[-1]:
        return slice(lo, hi)
    return slice(hi, lo)


def aef_transform(data: xr.DataArray) -> rasterio.Affine:
    x = data.coords["x"].values
    y = data.coords["y"].values
    if len(x) < 2 or len(y) < 2:
        raise ValueError("AEF crop is too small to build a transform.")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    minx = float(x.min() - abs(dx) / 2.0)
    maxx = float(x.max() + abs(dx) / 2.0)
    miny = float(y.min() - abs(dy) / 2.0)
    maxy = float(y.max() + abs(dy) / 2.0)
    return from_bounds(minx, miny, maxx, maxy, len(x), len(y))


def read_aef_patch(ds: xr.Dataset, ref_path: Path) -> torch.Tensor:
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)
        bounds = ref.bounds

    wgs84_bounds = transform_bounds(dst_crs, "EPSG:4326", *bounds, densify_pts=21)
    min_lon, min_lat, max_lon, max_lat = wgs84_bounds
    pad_lon = max((max_lon - min_lon) * 0.05, 1e-4)
    pad_lat = max((max_lat - min_lat) * 0.05, 1e-4)

    x_values = ds["embeddings"].coords["x"].values
    y_values = ds["embeddings"].coords["y"].values
    data = (
        ds["embeddings"]
        .isel(time=AEF_TIME_INDEX_2025)
        .sel(
            x=_coord_slice(x_values, min_lon - pad_lon, max_lon + pad_lon),
            y=_coord_slice(y_values, min_lat - pad_lat, max_lat + pad_lat),
        )
        .compute()
    )
    values = data.values
    if values.ndim != 3:
        raise ValueError(f"Expected AEF crop with 3 dims, got {values.shape}")
    emb = dequantize(values)
    src_transform = aef_transform(data)
    dst = np.full((emb.shape[0], *dst_shape), np.nan, dtype=np.float32)
    reproject(
        source=emb,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=4,
    )
    dst = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(dst.astype(np.float32, copy=False))


def write_meta(out_root: Path, args: argparse.Namespace, patch_ids: list[str]) -> None:
    meta: dict[str, Any] = {
        "source": "AlphaEarth Foundations Satellite Embedding Dataset",
        "aef_zarr_url": args.aef_zarr_url,
        "aef_time_index": AEF_TIME_INDEX_2025,
        "aef_year": 2025,
        "month_alias": args.month,
        "region": args.region,
        "num_patches": len(patch_ids),
        "patch_ids": patch_ids,
        "note": "Annual 2025 AEF embedding saved with a local month alias for downstream compatibility.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aef-zarr-url", default=AEF_ZARR_URL)
    parser.add_argument("--processed-root", type=Path, default=Path("/data/xuannv_embedding/processed"))
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--patch-ids", nargs="*")
    parser.add_argument("--out-root", type=Path, default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"))
    parser.add_argument("--month", default="202512")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    label_root = args.label_root
    if label_root is None:
        label_root = args.processed_root / args.region / "labels" / "building_osm"
    patch_ids = load_patch_ids(label_root, args.patch_ids)

    LOGGER.info("Opening AEF mosaic: %s", args.aef_zarr_url)
    ds = xr.open_zarr(args.aef_zarr_url, storage_options={"anon": True}, consolidated=False)
    LOGGER.info("AEF embeddings shape: %s", ds["embeddings"].shape)

    processed_region_root = args.processed_root / args.region
    region_out = args.out_root / args.region
    region_out.mkdir(parents=True, exist_ok=True)
    write_meta(args.out_root, args, patch_ids)

    for idx, patch_id in enumerate(patch_ids, start=1):
        patch_dir = region_out / patch_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        out_path = patch_dir / f"{args.month}_embedding_map.pt"
        if out_path.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] skip existing %s", idx, len(patch_ids), out_path)
            continue
        ref_path = find_reference_raster(processed_region_root, patch_id)
        try:
            emb = read_aef_patch(ds, ref_path)
            torch.save(emb, out_path)
            LOGGER.info("[%d/%d] saved %s shape=%s", idx, len(patch_ids), out_path, tuple(emb.shape))
        except Exception:
            LOGGER.exception("[%d/%d] failed %s using %s", idx, len(patch_ids), patch_id, ref_path)


if __name__ == "__main__":
    main()
