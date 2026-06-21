#!/usr/bin/env python3
"""Download and align AEF 2025 annual embeddings from source.coop for downstream patches.

Reads patch bboxes from metadata JSON, fetches the corresponding 2025 annual
embedding from the AEF Zarr mosaic, de-quantizes, resamples to the patch's
128x128 grid, and saves as PyTorch tensors.
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
from rasterio.warp import reproject, Resampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AEF_ZARR_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"
AEF_TIME_INDEX_2025 = 8  # 2017=0, ..., 2025=8


def dequantize(aef_int8: np.ndarray) -> np.ndarray:
    """Convert AEF int8 quantized embeddings to float32."""
    x = aef_int8.astype(np.float32)
    return ((x / 127.5) ** 2) * np.sign(x)


def read_aef_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Read (64, H, W) float32 AEF embedding for a WGS84 bbox.

    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    data = (
        ds["embeddings"]
        .isel(time=AEF_TIME_INDEX_2025)
        .sel(x=slice(min_lon, max_lon), y=slice(max_lat, min_lat))
        .compute()
    )
    emb = data.values  # (band, y, x)
    return dequantize(emb)


def aef_transform(data: xr.DataArray) -> rasterio.Affine:
    """Build a geotransform from the x/y coordinates of a sliced DataArray."""
    lon = data.x.values
    lat = data.y.values
    dx = lon[1] - lon[0]
    dy = lat[1] - lat[0]
    return rasterio.Affine.translation(lon[0], lat[0]) * rasterio.Affine.scale(dx, dy)


def resample_to_patch(
    emb: np.ndarray,
    src_transform: rasterio.Affine,
    patch_bounds: tuple[float, float, float, float],
    patch_shape: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Resample AEF embedding (64, H, W) to the patch grid."""
    minx, miny, maxx, maxy = patch_bounds
    dst_transform = from_bounds(minx, miny, maxx, maxy, patch_shape[1], patch_shape[0])
    dst = np.empty((emb.shape[0], *patch_shape), dtype=np.float32)
    reproject(
        source=emb,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
        num_threads=4,
    )
    return dst


def process_region(
    region: str,
    patch_meta_path: Path,
    out_root: Path,
    ds: xr.Dataset,
) -> None:
    """Download and align AEF embeddings for all patches in a region."""
    out_dir = out_root / region
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(patch_meta_path, encoding="utf-8") as f:
        patches: dict[str, dict[str, Any]] = json.load(f)

    # Use the first patch's CRS for the region (all patches share CRS)
    sample_crs = next(iter(patches.values()))["crs"]
    transformer = Transformer.from_crs(sample_crs, "EPSG:4326", always_xy=True)

    for idx, (patch_id, meta) in enumerate(patches.items(), 1):
        patch_bounds = meta["bbox"]  # in patch CRS (e.g. UTM)
        patch_dir = out_dir / patch_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        out_path = patch_dir / "202512_embedding_map.pt"
        if out_path.exists():
            logger.info("[%s %d/%d] %s already exists, skipping", region, idx, len(patches), patch_id)
            continue

        try:
            # Reproject patch bounds to EPSG:4326 for slicing AEF mosaic
            min_lon, min_lat = transformer.transform(patch_bounds[0], patch_bounds[1])
            max_lon, max_lat = transformer.transform(patch_bounds[2], patch_bounds[3])
            wgs84_bbox = (min_lon, min_lat, max_lon, max_lat)

            data = (
                ds["embeddings"]
                .isel(time=AEF_TIME_INDEX_2025)
                .sel(x=slice(wgs84_bbox[0], wgs84_bbox[2]), y=slice(wgs84_bbox[3], wgs84_bbox[1]))
            )
            data = data.compute()
            emb = dequantize(data.values)
            src_transform = aef_transform(data)
            patch_emb = resample_to_patch(emb, src_transform, wgs84_bbox, patch_shape=(128, 128))
            torch.save(torch.from_numpy(patch_emb), out_path)
            logger.info("[%s %d/%d] Saved %s", region, idx, len(patches), out_path)
        except Exception as e:
            logger.error("[%s %d/%d] Failed %s: %s", region, idx, len(patches), patch_id, e)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare AEF 2025 embeddings for downstream patches")
    p.add_argument(
        "--aef-zarr-url",
        default=AEF_ZARR_URL,
        help="URL to AEF Zarr mosaic on source.coop",
    )
    p.add_argument(
        "--patch-meta-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
        help="Directory containing <region>_patch_meta.json files",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
        help="Output directory for embeddings",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=["harbin", "haidian"],
        help="Regions to process",
    )
    args = p.parse_args()

    logger.info("Opening AEF Zarr mosaic: %s", args.aef_zarr_url)
    ds = xr.open_zarr(args.aef_zarr_url, storage_options={"anon": True}, consolidated=False)
    logger.info("Dataset shape: %s", ds["embeddings"].shape)

    for region in args.regions:
        meta_path = args.patch_meta_root / f"{region}_patch_meta.json"
        if not meta_path.exists():
            logger.warning("Metadata not found: %s", meta_path)
            continue
        process_region(region, meta_path, args.out_root, ds)

    logger.info("Done")


if __name__ == "__main__":
    main()
