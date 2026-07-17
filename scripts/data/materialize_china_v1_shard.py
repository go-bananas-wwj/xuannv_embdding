#!/usr/bin/env python3
"""Materialize China V1 COG windows into one restartable Zarr shard.

The script reads only a 1280 m by 1280 m window from signed Planetary
Computer COGs.  It never downloads complete satellite scenes.  A shard is
written to a temporary directory and atomically renamed only after its Zarr
metadata, per-patch quality records, and array shapes have been verified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "10")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_CONCURRENCY", "4")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.jp2")

import numpy as np
import planetary_computer
import rasterio
import zarr
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds


LOGGER = logging.getLogger("china_v1.materialize")
CHIP_SIDE_METERS = 1280.0
CHIP_PIXELS = 128
S2_VALID_SCL = {4, 5, 6, 7, 11}
SOURCES: dict[str, dict[str, Any]] = {
    "s2": {"assets": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"], "categorical": {"SCL"}},
    "s1": {"assets": ["vv", "vh"], "categorical": set()},
    "landsat": {"assets": ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"], "categorical": {"qa_pixel"}},
}


@dataclass(frozen=True)
class Patch:
    patch_id: str
    epsg: int
    bounds: tuple[float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float]


class CatalogIndex:
    """In-memory one-degree bbox index for one source/month STAC catalogue."""

    def __init__(self, items_path: Path) -> None:
        self.items = list(_read_jsonl(items_path))
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, item in enumerate(self.items):
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            for lon in range(int(np.floor(bbox[0])), int(np.ceil(bbox[2])) + 1):
                for lat in range(int(np.floor(bbox[1])), int(np.ceil(bbox[3])) + 1):
                    self.cells[(lon, lat)].append(index)

    def query(self, bounds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
        seen: set[int] = set()
        matches: list[dict[str, Any]] = []
        for lon in range(int(np.floor(bounds[0])), int(np.ceil(bounds[2])) + 1):
            for lat in range(int(np.floor(bounds[1])), int(np.ceil(bounds[3])) + 1):
                for index in self.cells.get((lon, lat), []):
                    if index in seen:
                        continue
                    seen.add(index)
                    item = self.items[index]
                    if item.get("bbox") and _intersects(bounds, item["bbox"]):
                        matches.append(item)
        return matches


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _patch_from_record(record: dict[str, Any]) -> Patch:
    required = ("patch_id", "grid_epsg", "grid_col", "grid_row")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"sample record lacks {missing}: {record.get('patch_id', '<unknown>')}")
    left = float(record["grid_col"]) * CHIP_SIDE_METERS
    bottom = float(record["grid_row"]) * CHIP_SIDE_METERS
    bounds = (left, bottom, left + CHIP_SIDE_METERS, bottom + CHIP_SIDE_METERS)
    transformer = Transformer.from_crs(CRS.from_epsg(int(record["grid_epsg"])), "EPSG:4326", always_xy=True)
    x0, y0 = transformer.transform(bounds[0], bounds[1])
    x1, y1 = transformer.transform(bounds[2], bounds[3])
    return Patch(str(record["patch_id"]), int(record["grid_epsg"]), bounds, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


def _intersects(left: tuple[float, float, float, float], right: list[float] | tuple[float, float, float, float]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _datetime_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    properties = item.get("properties", {})
    cloud = properties.get("eo:cloud_cover")
    return (float(cloud) if isinstance(cloud, (int, float)) else 1000.0, str(properties.get("datetime", "")))


def _select_items(catalog: CatalogIndex, patch: Patch, source: str, candidate_limit: int) -> list[dict[str, Any]]:
    config = SOURCES[source]
    candidates = [
        item for item in catalog.query(patch.wgs84_bounds)
        if all(asset in item.get("assets", {}) for asset in config["assets"])
    ]
    candidates.sort(key=_datetime_sort_key)
    return candidates[:candidate_limit]


def _read_asset_to_patch(href: str, patch: Patch, categorical: bool) -> np.ndarray:
    """Read and reproject one remote COG band without fetching a whole scene."""
    with rasterio.open(planetary_computer.sign(href)) as src:
        if src.crs is None:
            raise ValueError(f"asset without CRS: {href}")
        source_bounds = transform_bounds(CRS.from_epsg(patch.epsg), src.crs, *patch.bounds, densify_pts=21)
        requested = window_from_bounds(*source_bounds, transform=src.transform).round_offsets().round_lengths()
        overlap = requested.intersection(Window(0, 0, src.width, src.height))
        if overlap.width <= 0 or overlap.height <= 0:
            raise ValueError("patch does not overlap selected raster")
        source = src.read(1, window=overlap)
        output = np.full((CHIP_PIXELS, CHIP_PIXELS), np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=output,
            src_transform=src.window_transform(overlap),
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=from_bounds(*patch.bounds, CHIP_PIXELS, CHIP_PIXELS),
            dst_crs=CRS.from_epsg(patch.epsg),
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
    return output


def _scene_valid_mask(source: str, stack: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack[0]) & (stack[0] != 0)
    if source == "s2":
        valid &= np.isin(np.nan_to_num(stack[-1], nan=0.0).astype(np.uint8), list(S2_VALID_SCL))
    elif source == "landsat":
        qa = np.nan_to_num(stack[-1], nan=0.0).astype(np.uint16)
        valid &= (qa & 0b11111) == 0
    return valid


def _composite(source: str, scenes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if not scenes:
        bands = len(SOURCES[source]["assets"])
        return np.full((bands, CHIP_PIXELS, CHIP_PIXELS), np.nan, dtype=np.float32), np.zeros((CHIP_PIXELS, CHIP_PIXELS), dtype=np.uint8), []
    validity = [_scene_valid_mask(source, scene) for scene in scenes]
    fractions = [round(float(mask.mean()), 6) for mask in validity]
    best_index = int(np.argmax(fractions))
    valid = np.logical_or.reduce(validity)
    if source == "s1":
        output = np.nanmean(np.stack(scenes, axis=0), axis=0).astype(np.float32)
    else:
        continuous = []
        for scene, scene_mask in zip(scenes, validity):
            continuous.append(np.where(scene_mask[None, :, :], scene, np.nan))
        output = np.nanmedian(np.stack(continuous, axis=0), axis=0).astype(np.float32)
        output[-1] = scenes[best_index][-1]
    output[:, ~valid] = np.nan
    return output, valid.astype(np.uint8), fractions


def _load_scene(source: str, item: dict[str, Any], patch: Patch) -> np.ndarray:
    config = SOURCES[source]
    arrays = [
        _read_asset_to_patch(item["assets"][asset]["href"], patch, asset in config["categorical"])
        for asset in config["assets"]
    ]
    scene = np.stack(arrays).astype(np.float32)
    if source == "landsat":
        reflectance = scene[:-1]
        usable = np.isfinite(reflectance) & (reflectance != 0)
        reflectance[usable] = reflectance[usable] * 0.0000275 - 0.2
    return scene


def _load_available_scenes(
    source: str,
    candidates: list[dict[str, Any]],
    patch: Patch,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, str]]]:
    """Keep reading alternatives when a STAC bbox overstates raster coverage."""
    accepted: list[dict[str, Any]] = []
    scenes: list[np.ndarray] = []
    rejected: list[dict[str, str]] = []
    for item in candidates:
        try:
            scene = _load_scene(source, item, patch)
        except Exception as exc:
            rejected.append({"item_id": str(item.get("id", "unknown")), "error": f"{type(exc).__name__}: {exc}"})
            continue
        accepted.append(item)
        scenes.append(scene)
        if len(scenes) >= top_k:
            break
    return accepted, scenes, rejected


def _create_arrays(group: zarr.Group, source: str, patch_count: int, months: list[str]) -> tuple[Any, Any]:
    band_count = len(SOURCES[source]["assets"])
    image = group.create_dataset(
        "image", shape=(patch_count, len(months), band_count, CHIP_PIXELS, CHIP_PIXELS),
        chunks=(1, 1, band_count, CHIP_PIXELS, CHIP_PIXELS), dtype="f4", fill_value=np.nan,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
    )
    mask = group.create_dataset(
        "valid_mask", shape=(patch_count, len(months), CHIP_PIXELS, CHIP_PIXELS),
        chunks=(1, 1, CHIP_PIXELS, CHIP_PIXELS), dtype="u1", fill_value=0,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
    )
    group.attrs.update({"assets": SOURCES[source]["assets"], "source": source})
    return image, mask


def materialize(points: list[Patch], catalog_root: Path, output: Path, months: list[str], top_k: int) -> dict[str, Any]:
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(temporary), mode="w")
    group.attrs.update({
        "schema_version": "china_v1_zarr_shard_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "patch_ids": [patch.patch_id for patch in points], "months": months, "chip_pixels": CHIP_PIXELS,
        "chip_side_meters": CHIP_SIDE_METERS,
    })
    catalogs = {
        (source, month): CatalogIndex(catalog_root / source / month / "items.jsonl")
        for source in SOURCES for month in months
    }
    arrays = {source: _create_arrays(group.create_group(source), source, len(points), months) for source in SOURCES}
    quality_path = temporary / "quality.jsonl"
    failures = 0
    with quality_path.open("w", encoding="utf-8") as quality_handle:
        for patch_index, patch in enumerate(points):
            for month_index, month in enumerate(months):
                for source in SOURCES:
                    record: dict[str, Any] = {"patch_id": patch.patch_id, "month": month, "source": source, "selected_items": [], "valid_fraction": 0.0, "status": "missing"}
                    try:
                        candidates = _select_items(catalogs[(source, month)], patch, source, top_k * 4)
                        selected, scenes, rejected = _load_available_scenes(source, candidates, patch, top_k)
                        record["selected_items"] = [item["id"] for item in selected]
                        if rejected:
                            record["rejected_items"] = rejected
                        image, mask, scene_fractions = _composite(source, scenes)
                        arrays[source][0][patch_index, month_index] = image
                        arrays[source][1][patch_index, month_index] = mask
                        record.update({"status": "ok" if selected else "missing", "valid_fraction": round(float(mask.mean()), 6), "scene_valid_fractions": scene_fractions})
                    except Exception as exc:  # Keep independent sources/patches running and record retryable cause.
                        failures += 1
                        record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                        LOGGER.exception("failed %s %s %s", patch.patch_id, month, source)
                    quality_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    quality_handle.flush()
    zarr.consolidate_metadata(str(temporary))
    reopened = zarr.open_consolidated(str(temporary), mode="r")
    for source in SOURCES:
        expected = (len(points), len(months), len(SOURCES[source]["assets"]), CHIP_PIXELS, CHIP_PIXELS)
        if reopened[source]["image"].shape != expected:
            raise ValueError(f"Zarr shape check failed for {source}: {reopened[source]['image'].shape} != {expected}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {output}")
    temporary.replace(output)
    return {"output": str(output), "patches": len(points), "months": len(months), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True, help="Candidate point JSONL; first --limit are materialized.")
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if args.limit <= 0 or args.top_k <= 0:
        raise ValueError("--limit and --top-k must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    points = [_patch_from_record(record) for _, record in zip(range(args.limit), _read_jsonl(args.points))]
    required_catalogs = [args.catalog_root / source / month / "items.jsonl" for source in SOURCES for month in args.months]
    missing = [str(path) for path in required_catalogs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"catalog cache incomplete; missing {missing[:3]}")
    started = time.monotonic()
    report = materialize(points, args.catalog_root, args.output, args.months, args.top_k)
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
