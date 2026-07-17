#!/usr/bin/env python3
"""Materialize China V1 COG windows into one restartable Zarr shard.

The script reads only a 1280 m by 1280 m window from signed Planetary
Computer COGs.  It never downloads complete satellite scenes.  A shard is
written to a temporary directory and atomically renamed only after its Zarr
metadata, per-patch quality records, and array shapes have been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# A failed range request from the other side of the world must not stall an
# entire shard for ten minutes.  Python-level retries below reopen the COG
# between attempts, which is more reliable than a long-lived GDAL retry loop.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1024")
os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "20")
os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
# Azure Blob occasionally returns a valid 206 response that GDAL's multi-range
# combiner misparses on this cross-border route.  Single range reads were
# verified against real Sentinel-2 COG windows and complete reliably.
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "NO")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "NO")
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
    "s2": {
        "assets": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"],
        "categorical": {"SCL"}, "quality_asset": "SCL", "min_clear_fraction": 0.90,
    },
    "s1": {"assets": ["vv", "vh"], "categorical": set(), "quality_asset": "vv", "min_clear_fraction": 0.95},
    "landsat": {
        "assets": ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"],
        "categorical": {"qa_pixel"}, "quality_asset": "qa_pixel", "min_clear_fraction": 0.90,
    },
}


@dataclass(frozen=True)
class Patch:
    patch_id: str
    epsg: int
    bounds: tuple[float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float]


class AssetReaderCache:
    """Bounded per-worker COG reader cache for spatially adjacent patch jobs."""

    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("asset cache size must be positive")
        self.max_entries = max_entries
        self._readers: OrderedDict[str, rasterio.io.DatasetReader] = OrderedDict()

    def get(self, href: str) -> rasterio.io.DatasetReader:
        reader = self._readers.pop(href, None)
        if reader is None:
            reader = rasterio.open(planetary_computer.sign(href))
        self._readers[href] = reader
        while len(self._readers) > self.max_entries:
            _, stale = self._readers.popitem(last=False)
            stale.close()
        return reader

    def close(self) -> None:
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()

    def invalidate(self, href: str) -> None:
        """Close a reader after a truncated HTTP range response."""
        reader = self._readers.pop(href, None)
        if reader is not None:
            reader.close()


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
    return candidates if candidate_limit <= 0 else candidates[:candidate_limit]


def _reproject_open_asset(src: rasterio.io.DatasetReader, href: str, patch: Patch, categorical: bool) -> np.ndarray:
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


def _read_asset_to_patch(
    href: str,
    patch: Patch,
    categorical: bool,
    reader_cache: AssetReaderCache | None = None,
    attempts: int = 3,
) -> np.ndarray:
    """Read one remote COG band with bounded reconnect retries.

    Cached GDAL readers are deliberately used only by sequential asset reads.
    GDAL DatasetReader objects are not safe to share between concurrent
    threads, and doing so produced truncated COG tiles in the first full run.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if reader_cache is not None:
                return _reproject_open_asset(reader_cache.get(href), href, patch, categorical)
            with rasterio.open(planetary_computer.sign(href)) as src:
                return _reproject_open_asset(src, href, patch, categorical)
        except Exception as exc:
            last_error = exc
            if reader_cache is not None:
                reader_cache.invalidate(href)
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


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
    valid_stack = np.stack(validity, axis=0)
    valid = np.logical_or.reduce(valid_stack)
    stacked = np.stack(scenes, axis=0)
    # Invalid pixels must never contribute finite values to any source composite.
    masked = np.where(valid_stack[:, None, :, :], stacked, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output = (
            np.nanmean(masked, axis=0) if source == "s1" else np.nanmedian(masked, axis=0)
        ).astype(np.float32)
    if SOURCES[source]["categorical"]:
        # The categorical QA/SCL value comes from the first valid scene at each
        # pixel, rather than from a globally selected scene with different cloud.
        first_valid = np.argmax(valid_stack, axis=0)
        output[-1] = np.take_along_axis(stacked[:, -1], first_valid[None, :, :], axis=0)[0]
    output[:, ~valid] = np.nan
    return output, valid.astype(np.uint8), fractions


def _load_scene(
    source: str,
    item: dict[str, Any],
    patch: Patch,
    asset_workers: int = 1,
    reader_cache: AssetReaderCache | None = None,
    quality_array: np.ndarray | None = None,
) -> np.ndarray:
    config = SOURCES[source]
    quality_asset = str(config["quality_asset"])
    assets = [asset for asset in config["assets"] if quality_array is None or asset != quality_asset]

    def read(asset: str) -> np.ndarray:
        return _read_asset_to_patch(item["assets"][asset]["href"], patch, asset in config["categorical"], reader_cache)
    if asset_workers == 1 or reader_cache is not None:
        arrays_by_asset = {asset: read(asset) for asset in assets}
    else:
        with ThreadPoolExecutor(max_workers=asset_workers, thread_name_prefix="cog-read") as executor:
            arrays_by_asset = dict(zip(assets, executor.map(read, assets)))
    if quality_array is not None:
        arrays_by_asset[quality_asset] = quality_array
    scene = np.stack([arrays_by_asset[asset] for asset in config["assets"]]).astype(np.float32)
    if source == "landsat":
        reflectance = scene[:-1]
        usable = np.isfinite(reflectance) & (reflectance != 0)
        reflectance[usable] = reflectance[usable] * 0.0000275 - 0.2
    return scene


def _quality_mask(source: str, quality_array: np.ndarray) -> np.ndarray:
    if source == "s2":
        return np.isin(np.nan_to_num(quality_array, nan=0.0).astype(np.uint8), list(S2_VALID_SCL))
    if source == "landsat":
        qa = np.nan_to_num(quality_array, nan=0.0).astype(np.uint16)
        return (qa & 0b11111) == 0
    return np.isfinite(quality_array) & (quality_array != 0)


def _load_available_scenes(
    source: str,
    candidates: list[dict[str, Any]],
    patch: Patch,
    max_clean_scenes: int,
    asset_workers: int,
    reader_cache: AssetReaderCache | None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    """Read every locally clean scene; reject cloud/haze-contaminated windows first."""
    accepted: list[dict[str, Any]] = []
    scenes: list[np.ndarray] = []
    rejected: list[dict[str, Any]] = []
    config = SOURCES[source]
    quality_asset = str(config["quality_asset"])
    min_clear_fraction = float(config["min_clear_fraction"])
    for item in candidates:
        try:
            quality_array = _read_asset_to_patch(
                item["assets"][quality_asset]["href"], patch,
                quality_asset in config["categorical"], reader_cache,
            )
            clear_fraction = float(_quality_mask(source, quality_array).mean())
        except Exception as exc:
            rejected.append({"item_id": str(item.get("id", "unknown")), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if clear_fraction < min_clear_fraction:
            rejected.append({"item_id": str(item.get("id", "unknown")), "reason": "quality_rejected", "clear_fraction": round(clear_fraction, 6)})
            continue
        try:
            scene = _load_scene(source, item, patch, asset_workers, reader_cache, quality_array)
        except Exception as exc:
            rejected.append({"item_id": str(item.get("id", "unknown")), "error": f"{type(exc).__name__}: {exc}"})
            continue
        accepted.append(item)
        scenes.append(scene)
        if max_clean_scenes > 0 and len(scenes) >= max_clean_scenes:
            break
    return accepted, scenes, rejected


def _create_arrays(group: zarr.Group, source: str, patch_count: int, months: list[str]) -> tuple[Any, Any]:
    band_count = len(SOURCES[source]["assets"])
    image = group.create_dataset(
        "image", shape=(patch_count, len(months), band_count, CHIP_PIXELS, CHIP_PIXELS),
        chunks=(min(8, patch_count), 1, band_count, CHIP_PIXELS, CHIP_PIXELS), dtype="f4", fill_value=np.nan,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
    )
    mask = group.create_dataset(
        "valid_mask", shape=(patch_count, len(months), CHIP_PIXELS, CHIP_PIXELS),
        chunks=(min(8, patch_count), 1, CHIP_PIXELS, CHIP_PIXELS), dtype="u1", fill_value=0,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
    )
    group.attrs.update({"assets": SOURCES[source]["assets"], "source": source})
    return image, mask


def _fingerprint(points: list[Patch], months: list[str], max_clean_scenes: int) -> str:
    payload = {
        "patch_ids": [patch.patch_id for patch in points], "months": months,
        "max_clean_scenes": max_clean_scenes,
        "quality_thresholds": {source: SOURCES[source]["min_clear_fraction"] for source in SOURCES},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return str(record["patch_id"]), str(record["month"]), str(record["source"])


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {_record_key(record): record for record in _read_jsonl(path)} if path.exists() else {}


def load_catalogs(catalog_root: Path, months: list[str]) -> dict[tuple[str, str], CatalogIndex]:
    """Load every requested source/month index once per long-lived worker."""
    return {
        (source, month): CatalogIndex(catalog_root / source / month / "items.jsonl")
        for source in SOURCES for month in months
    }


def materialize(
    points: list[Patch],
    catalog_root: Path,
    output: Path,
    months: list[str],
    max_clean_scenes: int,
    catalogs: dict[tuple[str, str], CatalogIndex] | None = None,
    asset_workers: int = 1,
    reader_cache: AssetReaderCache | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {output}")
    temporary = output.with_name(output.name + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    lock_path = temporary.with_name(temporary.name + ".lock")
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"shard is already owned by another worker: {lock_path}") from exc
    fingerprint = _fingerprint(points, months, max_clean_scenes)
    try:
        creating = not temporary.exists()
        group = zarr.open_group(str(temporary), mode="w" if creating else "a")
        if creating:
            group.attrs.update({
                "schema_version": "china_v1_zarr_shard_v3", "created_at": datetime.now(timezone.utc).isoformat(),
                "patch_ids": [patch.patch_id for patch in points], "months": months, "chip_pixels": CHIP_PIXELS,
                "chip_side_meters": CHIP_SIDE_METERS, "input_fingerprint": fingerprint,
                "max_clean_scenes": max_clean_scenes,
                "quality_thresholds": {source: SOURCES[source]["min_clear_fraction"] for source in SOURCES},
            })
        elif group.attrs.get("input_fingerprint") != fingerprint:
            raise ValueError("partial shard fingerprint differs from requested inputs")
        catalogs = catalogs or load_catalogs(catalog_root, months)
        arrays = {
            source: (group[source]["image"], group[source]["valid_mask"])
            if source in group else _create_arrays(group.create_group(source), source, len(points), months)
            for source in SOURCES
        }
        done = group["done"] if "done" in group else group.create_dataset(
            "done", shape=(len(points), len(months), len(SOURCES)), chunks=(min(8, len(points)), 1, len(SOURCES)), dtype="u1", fill_value=0,
        )
        quality_path = temporary / "quality.jsonl"
        records = _load_records(quality_path)
        failures = 0
        for patch_index, patch in enumerate(points):
            for month_index, month in enumerate(months):
                for source_index, source in enumerate(SOURCES):
                    if done[patch_index, month_index, source_index]:
                        continue
                    record: dict[str, Any] = {"patch_id": patch.patch_id, "month": month, "source": source, "selected_items": [], "valid_fraction": 0.0, "status": "missing"}
                    try:
                        candidates = _select_items(catalogs[(source, month)], patch, source, candidate_limit=0)
                        selected, scenes, rejected = _load_available_scenes(
                            source, candidates, patch, max_clean_scenes, asset_workers, reader_cache,
                        )
                        record["selected_items"] = [item["id"] for item in selected]
                        record["candidate_count"] = len(candidates)
                        if rejected:
                            record["rejected_items"] = rejected
                        image, mask, scene_fractions = _composite(source, scenes)
                        arrays[source][0][patch_index, month_index] = image
                        arrays[source][1][patch_index, month_index] = mask
                        valid_fraction = round(float(mask.mean()), 6)
                        record.update({
                            "status": "ok" if valid_fraction > 0 else ("no_valid_pixels" if selected else "no_candidate"),
                            "valid_fraction": valid_fraction, "scene_valid_fractions": scene_fractions,
                        })
                    except Exception as exc:
                        failures += 1
                        record.update({"status": "retryable_error", "error": f"{type(exc).__name__}: {exc}"})
                        LOGGER.exception("failed %s %s %s", patch.patch_id, month, source)
                        records[_record_key(record)] = record
                        _append_record(quality_path, record)
                        continue
                    records[_record_key(record)] = record
                    _append_record(quality_path, record)
                    done[patch_index, month_index, source_index] = 1
        if np.count_nonzero(done[:]) != done.size:
            raise RuntimeError("shard has retryable errors; retain partial output for a later resume")
        expected_records = len(points) * len(months) * len(SOURCES)
        if len(records) != expected_records:
            raise RuntimeError(f"quality record check failed: {len(records)} != {expected_records}")
        zarr.consolidate_metadata(str(temporary))
        reopened = zarr.open_consolidated(str(temporary), mode="r")
        for source in SOURCES:
            expected = (len(points), len(months), len(SOURCES[source]["assets"]), CHIP_PIXELS, CHIP_PIXELS)
            if reopened[source]["image"].shape != expected:
                raise ValueError(f"Zarr shape check failed for {source}: {reopened[source]['image'].shape} != {expected}")
        report = {"output": str(output), "patches": len(points), "months": len(months), "failures": failures, "records": expected_records, "input_fingerprint": fingerprint}
        (temporary / "shard_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        return report
    finally:
        lock_path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True, help="Candidate point JSONL; first --limit are materialized.")
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-clean-scenes", type=int, default=0, help="Maximum locally clean scenes per source/month; 0 keeps all of them.")
    parser.add_argument("--asset-workers", type=int, default=2, help="Concurrent COG assets per process; bound the global total across workers.")
    parser.add_argument("--asset-cache-size", type=int, default=256)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_clean_scenes < 0 or args.asset_workers <= 0 or args.asset_cache_size <= 0:
        raise ValueError("--limit, --max-clean-scenes, --asset-workers and --asset-cache-size must be valid")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    points = [_patch_from_record(record) for _, record in zip(range(args.limit), _read_jsonl(args.points))]
    required_catalogs = [args.catalog_root / source / month / "items.jsonl" for source in SOURCES for month in args.months]
    missing = [str(path) for path in required_catalogs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"catalog cache incomplete; missing {missing[:3]}")
    started = time.monotonic()
    cache = AssetReaderCache(args.asset_cache_size)
    try:
        report = materialize(points, args.catalog_root, args.output, args.months, args.max_clean_scenes, asset_workers=args.asset_workers, reader_cache=cache)
    finally:
        cache.close()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
