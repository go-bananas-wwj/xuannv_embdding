#!/usr/bin/env python3
"""Prepare AlphaEarth Foundations 2025 annual embeddings for local patches.

The public AEF release provides annual embedding COGs indexed by a GeoParquet
file. This script finds the 2025 COG intersecting each local patch, streams the
needed pixels from Source Cooperative, dequantizes them, and warps them onto the
existing 128x128 patch grid so downstream probe code can read the result as
``{month}_embedding_map.pt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import s3fs
import torch
from rasterio.env import Env
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

LOGGER = logging.getLogger(__name__)

AEF_INDEX_URL = (
    "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"
    "aef_index_stac_geoparquet.parquet"
)
AEF_YEAR = 2025
AEF_NODATA = -128


def dequantize(values: np.ndarray) -> np.ndarray:
    """Convert AEF int8 values to float32 using the official quadratic scale."""
    x = values.astype(np.float32)
    out = ((x / 127.5) ** 2) * np.sign(x)
    out[values == AEF_NODATA] = np.nan
    return out


def s3_to_vsis3(path: str) -> str:
    if path.startswith("s3://"):
        return "/vsis3/" + path.removeprefix("s3://")
    return path


def cache_cog(cog_path: str, cache_dir: Path | None) -> str:
    if cache_dir is None or not cog_path.startswith("s3://"):
        return cog_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(cog_path).name
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    LOGGER.info("Downloading AEF COG to local cache: %s -> %s", cog_path, local_path)
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(cog_path, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
    tmp_path.replace(local_path)
    return str(local_path)


def bounds_to_window(src: rasterio.io.DatasetReader, bounds: Any) -> Window:
    """Build a robust window for north-up or south-up source transforms."""
    corners = [
        (bounds.left, bounds.top),
        (bounds.right, bounds.top),
        (bounds.left, bounds.bottom),
        (bounds.right, bounds.bottom),
    ]
    rows: list[int] = []
    cols: list[int] = []
    for x, y in corners:
        row, col = src.index(x, y)
        rows.append(row)
        cols.append(col)
    row_start = max(min(rows), 0)
    row_stop = min(max(rows), src.height)
    col_start = max(min(cols), 0)
    col_stop = min(max(cols), src.width)
    return Window(
        col_start,
        row_start,
        max(col_stop - col_start, 1),
        max(row_stop - row_start, 1),
    )


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


def load_aef_index(index_path: str) -> gpd.GeoDataFrame:
    LOGGER.info("Loading AEF COG index: %s", index_path)
    index = gpd.read_parquet(index_path, storage_options={"anon": True})
    if "datetime" in index.columns:
        index = index[index["datetime"].dt.year == AEF_YEAR].copy()
    elif "year" in index.columns:
        index = index[index["year"] == AEF_YEAR].copy()
    if index.empty:
        raise RuntimeError(f"No AEF records found for year {AEF_YEAR}")
    if index.crs is None:
        index = index.set_crs("OGC:CRS84")
    return index


def patch_wgs84_geometry(ref_path: Path) -> Any:
    with rasterio.open(ref_path) as ref:
        min_lon, min_lat, max_lon, max_lat = transform_bounds(
            ref.crs, "OGC:CRS84", *ref.bounds, densify_pts=21
        )
    return box(min_lon, min_lat, max_lon, max_lat)


def find_aef_cog(index: gpd.GeoDataFrame, ref_path: Path) -> str:
    geom = patch_wgs84_geometry(ref_path)
    matches = index[index.intersects(geom)]
    if matches.empty:
        raise FileNotFoundError(f"No AEF 2025 COG intersects {ref_path}")
    row = matches.iloc[0]
    assets = row.get("assets")
    if isinstance(assets, dict) and "data" in assets:
        return str(assets["data"]["href"])
    if "path" in matches.columns:
        return str(row["path"])
    raise KeyError("AEF index row has neither assets.data.href nor path")


def build_patch_cog_map(
    index: gpd.GeoDataFrame,
    processed_region_root: Path,
    patch_ids: list[str],
    cache_dir: Path | None,
) -> dict[str, str]:
    patch_to_cog: dict[str, str] = {}
    for patch_id in patch_ids:
        ref_path = find_reference_raster(processed_region_root, patch_id)
        patch_to_cog[patch_id] = find_aef_cog(index, ref_path)
    unique_cogs = sorted(set(patch_to_cog.values()))
    LOGGER.info("AEF COGs intersecting requested patches: %d", len(unique_cogs))
    cog_to_local = {cog: cache_cog(cog, cache_dir) for cog in unique_cogs}
    return {patch_id: cog_to_local[cog] for patch_id, cog in patch_to_cog.items()}


def read_aef_patch_with_validity(cog_path: str, ref_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Read one AEF map and retain its destination-pixel validity mask.

    The legacy caller historically replaced nodata with zero.  Paper-grade exporters
    need to inspect validity *before* that replacement, so this helper is deliberately
    lossless with respect to nodata and returns a boolean ``(H, W)`` coverage mask.
    """
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_shape = (ref.height, ref.width)

    with Env(AWS_NO_SIGN_REQUEST="YES"):
        open_path = s3_to_vsis3(cog_path) if cog_path.startswith("s3://") else cog_path
        with rasterio.open(open_path) as src:
            if src.nodata != AEF_NODATA:
                raise ValueError(
                    f"AEF source nodata must be declared as {AEF_NODATA}, "
                    f"got {src.nodata}: {cog_path}"
                )
            dst = np.full((src.count, *dst_shape), np.nan, dtype=np.float32)
            for band_idx in range(1, src.count + 1):
                band = np.full(dst_shape, np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=AEF_NODATA,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    num_threads=4,
                )
                dst[band_idx - 1] = band

    dst = dequantize(dst)
    valid = np.isfinite(dst).all(axis=0)
    return torch.from_numpy(dst.astype(np.float32, copy=False)), valid


def read_aef_patch(cog_path: str, ref_path: Path) -> torch.Tensor:
    """Read an AEF map using the historical zero-filled nodata compatibility policy."""
    dst, _ = read_aef_patch_with_validity(cog_path, ref_path)
    return torch.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)


def write_meta(out_root: Path, args: argparse.Namespace, patch_ids: list[str]) -> None:
    meta: dict[str, Any] = {
        "source": "AlphaEarth Foundations Satellite Embedding Dataset",
        "aef_index_path": args.aef_index_path,
        "aef_year": AEF_YEAR,
        "month_alias": args.month,
        "region": args.region,
        "num_patches": len(patch_ids),
        "patch_ids": patch_ids,
        "note": (
            "Annual 2025 AEF embedding saved with a local month alias for downstream compatibility."
        ),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aef-index-path", default=AEF_INDEX_URL)
    parser.add_argument(
        "--processed-root", type=Path, default=Path("/data/xuannv_embedding/processed")
    )
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--patch-ids", nargs="*")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
    )
    parser.add_argument(
        "--cog-cache-dir",
        type=Path,
        default=None,
        help="Optional local cache for official AEF COGs before patch extraction.",
    )
    parser.add_argument("--month", default="202512")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    label_root = args.label_root
    if label_root is None:
        label_root = args.processed_root / args.region / "labels" / "building_osm"
    patch_ids = load_patch_ids(label_root, args.patch_ids)

    index = load_aef_index(args.aef_index_path)
    LOGGER.info("AEF index records for %d: %d", AEF_YEAR, len(index))

    processed_region_root = args.processed_root / args.region
    region_out = args.out_root / args.region
    region_out.mkdir(parents=True, exist_ok=True)
    write_meta(args.out_root, args, patch_ids)
    patch_to_cog = build_patch_cog_map(
        index,
        processed_region_root,
        patch_ids,
        args.cog_cache_dir,
    )

    for idx, patch_id in enumerate(patch_ids, start=1):
        patch_dir = region_out / patch_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        out_path = patch_dir / f"{args.month}_embedding_map.pt"
        if out_path.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] skip existing %s", idx, len(patch_ids), out_path)
            continue
        ref_path = find_reference_raster(processed_region_root, patch_id)
        try:
            cog_path = patch_to_cog[patch_id]
            emb = read_aef_patch(cog_path, ref_path)
            torch.save(emb, out_path)
            LOGGER.info(
                "[%d/%d] saved %s shape=%s cog=%s",
                idx,
                len(patch_ids),
                out_path,
                tuple(emb.shape),
                cog_path,
            )
        except Exception:
            LOGGER.exception("[%d/%d] failed %s using %s", idx, len(patch_ids), patch_id, ref_path)


if __name__ == "__main__":
    main()
