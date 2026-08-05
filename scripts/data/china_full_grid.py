#!/usr/bin/env python3
"""Enumerate the canonical nationwide 1,280 m parent grid."""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pyproj import Transformer
from shapely import normalize as normalize_geometry
from shapely import set_precision
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry

PARENT_SIDE_METERS = 1280
CHINA_OWNER_EPSGS = frozenset(range(32643, 32654))
SHAPEFILE_MAX_BYTES = 1_800_000_000
SHAPEFILE_SAFE_FRACTION = 0.95
SHAPEFILE_ROW_BLOCK_ROWS = 1_000
SHAPEFILE_COMPONENT_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")


@dataclass(frozen=True)
class GridSpec:
    side_m: int = PARENT_SIDE_METERS
    macro_side_patches: int = 10
    boundary_version: str = "geoBoundaries-CHN-ADM0-frozen-20260726"
    atlas_version: str = "china-full-1280m-v1-20260805"

    def __post_init__(self) -> None:
        if self.side_m != PARENT_SIDE_METERS:
            raise ValueError(f"side_m must be exactly {PARENT_SIDE_METERS}")


@dataclass
class ZoneWriteSummary:
    """Counts and GeoParquet parts emitted for one UTM grid zone."""

    grid_id: str
    all_count: int = 0
    sampled_count: int = 0
    unsampled_count: int = 0
    parquet_parts: list[str] = field(default_factory=list)
    shapefile_parts: list[str] = field(default_factory=list)
    batch_count: int = 0


def parent_key(grid_epsg: int, grid_col: int, grid_row: int) -> str:
    """Return the stable identity for one parent grid cell."""
    return f"{grid_epsg}:{grid_col}:{grid_row}"


def utm_owner_epsg(longitude: float, latitude: float) -> int:
    """Return the EPSG code for the unique half-open UTM zone at a point."""
    zone = max(1, min(60, math.floor((float(longitude) + 180.0) / 6.0) + 1))
    return (32600 if float(latitude) >= 0 else 32700) + zone


def _validate_china_owner_epsg(grid_epsg: int) -> None:
    if grid_epsg not in CHINA_OWNER_EPSGS:
        raise ValueError("grid_epsg must be within China owner EPSGs 32643 through 32653")


def build_patch_record(
    macro: Mapping[str, Any],
    grid_col: int,
    grid_row: int,
    longitude: float,
    latitude: float,
    spec: GridSpec,
) -> dict[str, Any]:
    """Build one canonical parent-grid record from its integer coordinates."""
    grid_epsg = int(macro["grid_epsg"])
    macro_col = int(macro["macro_col"])
    macro_row = int(macro["macro_row"])
    key = parent_key(grid_epsg, grid_col, grid_row)
    return {
        "schema_version": "china_full_1280m_parent_grid_v1",
        "atlas_version": spec.atlas_version,
        "boundary_version": spec.boundary_version,
        "patch_id": f"parent_{key}",
        "parent_key": key,
        "macro_id": str(macro.get("macro_id", f"{macro['grid_id']}_c{macro_col}_r{macro_row}")),
        "macro_local_col": grid_col - macro_col * spec.macro_side_patches,
        "macro_local_row": grid_row - macro_row * spec.macro_side_patches,
        "grid_id": str(macro["grid_id"]),
        "grid_epsg": grid_epsg,
        "grid_col": grid_col,
        "grid_row": grid_row,
        "utm_bounds": [
            grid_col * spec.side_m,
            grid_row * spec.side_m,
            (grid_col + 1) * spec.side_m,
            (grid_row + 1) * spec.side_m,
        ],
        "longitude": longitude,
        "latitude": latitude,
    }


def enumerate_macro_patch_records(
    macro: Mapping[str, Any], boundary_wgs84: Any, spec: GridSpec
) -> Iterator[dict[str, Any]]:
    """Yield owner-zone-valid parent cells whose centers are covered by ADM0."""
    grid_epsg = int(macro["grid_epsg"])
    _validate_china_owner_epsg(grid_epsg)
    macro_col = int(macro["macro_col"])
    macro_row = int(macro["macro_row"])
    to_wgs84 = Transformer.from_crs(grid_epsg, 4326, always_xy=True)
    half_side = spec.side_m / 2

    for local_col in range(spec.macro_side_patches):
        for local_row in range(spec.macro_side_patches):
            grid_col = macro_col * spec.macro_side_patches + local_col
            grid_row = macro_row * spec.macro_side_patches + local_row
            minx = grid_col * spec.side_m
            miny = grid_row * spec.side_m
            longitude, latitude = to_wgs84.transform(minx + half_side, miny + half_side)
            if utm_owner_epsg(longitude, latitude) != grid_epsg:
                continue
            if not boundary_wgs84.covers(Point(longitude, latitude)):
                continue
            yield build_patch_record(macro, grid_col, grid_row, longitude, latitude, spec)


def validate_patch_record(record: Mapping[str, Any], spec: GridSpec) -> None:
    """Raise ValueError when a record violates the canonical parent-grid contract."""
    required = (
        "patch_id",
        "parent_key",
        "macro_id",
        "macro_local_col",
        "macro_local_row",
        "grid_id",
        "grid_epsg",
        "grid_col",
        "grid_row",
        "utm_bounds",
        "longitude",
        "latitude",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"patch record missing required fields: {missing}")

    grid_epsg = record["grid_epsg"]
    grid_col = record["grid_col"]
    grid_row = record["grid_row"]
    if not all(isinstance(value, int) for value in (grid_epsg, grid_col, grid_row)):
        raise ValueError("grid_epsg, grid_col, and grid_row must be integers")
    _validate_china_owner_epsg(grid_epsg)
    expected_key = parent_key(grid_epsg, grid_col, grid_row)
    if record["parent_key"] != expected_key or record["patch_id"] != f"parent_{expected_key}":
        raise ValueError("parent_key and patch_id must match integer grid coordinates")

    expected_bounds = [
        grid_col * spec.side_m,
        grid_row * spec.side_m,
        (grid_col + 1) * spec.side_m,
        (grid_row + 1) * spec.side_m,
    ]
    if record["utm_bounds"] != expected_bounds:
        raise ValueError("utm_bounds must be exact integer-aligned parent-cell bounds")

    for local_field in ("macro_local_col", "macro_local_row"):
        local_value = record[local_field]
        if not isinstance(local_value, int) or not 0 <= local_value < spec.macro_side_patches:
            raise ValueError(f"{local_field} must be within the macrocell")

    longitude = record["longitude"]
    latitude = record["latitude"]
    if not isinstance(longitude, (int, float)) or not isinstance(latitude, (int, float)):
        raise ValueError("longitude and latitude must be numeric")
    if utm_owner_epsg(longitude, latitude) != grid_epsg:
        raise ValueError("grid_epsg must own the WGS84 center point")


def _wgs84_geometry(record: Mapping[str, Any]):
    minx, miny, maxx, maxy = record["utm_bounds"]
    to_wgs84 = Transformer.from_crs(int(record["grid_epsg"]), 4326, always_xy=True)
    return transform_geometry(to_wgs84.transform, box(minx, miny, maxx, maxy))


def _geoparquet_records(records: Iterable[Mapping[str, Any]], sampled_keys: set[str]):
    import geopandas as gpd

    rows = []
    for record in records:
        row = dict(record)
        geometry = _wgs84_geometry(record)
        identity = (
            f"{record['atlas_version']}:{record['grid_epsg']}:"
            f"{record['grid_col']}:{record['grid_row']}"
        )
        row["sampled"] = str(record["parent_key"]) in sampled_keys
        row["wgs84_bounds"] = list(geometry.bounds)
        row["identity_hash"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        row["footprint_hash"] = hashlib.sha256(
            normalize_geometry(set_precision(geometry, 1e-9)).wkb
        ).hexdigest()
        row["geometry"] = geometry
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _shapefile_records(records: Iterable[Mapping[str, Any]], sampled_keys: set[str]):
    import geopandas as gpd

    rows = []
    for record in records:
        rows.append(
            {
                "PATCH_ID": record["patch_id"],
                "UTM_EPSG": record["grid_epsg"],
                "GRID_COL": record["grid_col"],
                "GRID_ROW": record["grid_row"],
                "MACRO_ID": record["macro_id"],
                "SAMPLED": int(str(record["parent_key"]) in sampled_keys),
                "geometry": _wgs84_geometry(record),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _shapefile_component_sizes(path: Path) -> dict[str, int]:
    return {
        suffix: path.with_suffix(suffix).stat().st_size
        for suffix in SHAPEFILE_COMPONENT_SUFFIXES
        if path.with_suffix(suffix).exists()
    }


def _remove_shapefile(path: Path) -> None:
    for suffix in SHAPEFILE_COMPONENT_SUFFIXES:
        path.with_suffix(suffix).unlink(missing_ok=True)


def _shapefile_component_limit() -> int:
    return max(1, math.floor(SHAPEFILE_MAX_BYTES * SHAPEFILE_SAFE_FRACTION))


def _fits_shapefile_component_cap(
    records: list[Mapping[str, Any]],
    sampled_keys: set[str],
    probe_path: Path,
) -> bool:
    frame = _shapefile_records(records, sampled_keys)
    try:
        frame.to_file(probe_path, index=False)
        sizes = _shapefile_component_sizes(probe_path).values()
        return all(size < _shapefile_component_limit() for size in sizes)
    finally:
        _remove_shapefile(probe_path)


def _split_shapefile_records(
    records: list[Mapping[str, Any]], sampled_keys: set[str], staging_dir: Path
) -> list[list[Mapping[str, Any]]]:
    probe_path = staging_dir / f".cap-probe-{uuid.uuid4().hex}.shp"
    if _fits_shapefile_component_cap(records, sampled_keys, probe_path):
        return [records]
    if len(records) == 1:
        raise ValueError(
            f"one Shapefile feature exceeds the {SHAPEFILE_MAX_BYTES} byte component cap"
        )
    midpoint = len(records) // 2
    first_half = _split_shapefile_records(records[:midpoint], sampled_keys, staging_dir)
    second_half = _split_shapefile_records(records[midpoint:], sampled_keys, staging_dir)
    return first_half + second_half


def _stage_shapefile_partition(
    records: list[Mapping[str, Any]],
    sampled_keys: set[str],
    staging_root: Path,
    grid_id: str,
    partition: str,
    batch_index: int,
) -> list[Path]:
    records_by_row_block: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(
        records,
        key=lambda item: (int(item["grid_row"]), int(item["grid_col"]), str(item["parent_key"])),
    ):
        records_by_row_block[int(record["grid_row"]) // SHAPEFILE_ROW_BLOCK_ROWS].append(record)

    staging_dir = staging_root / partition / grid_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row_block, block_records in sorted(records_by_row_block.items()):
        for part_index, part_records in enumerate(
            _split_shapefile_records(block_records, sampled_keys, staging_dir)
        ):
            path = staging_dir / (
                f"{grid_id}_{partition}_rowblock-{row_block:06d}_part-{batch_index:05d}-{part_index:03d}.shp"
            )
            _shapefile_records(part_records, sampled_keys).to_file(path, index=False)
            sizes = _shapefile_component_sizes(path).values()
            if not all(size < _shapefile_component_limit() for size in sizes):
                raise ValueError(f"Shapefile cap check failed for {path}")
            paths.append(path)
    return paths


def _publish_staged_files(staging_root: Path, output_root: Path) -> list[Path]:
    staged_files = sorted(path for path in staging_root.rglob("*") if path.is_file())
    published = []
    try:
        for staged_path in staged_files:
            destination = output_root / staged_path.relative_to(staging_root)
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(destination)
            published.append(destination)
    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    return published


def write_zone_batch(
    batch: list[Mapping[str, Any]],
    sampled_keys: set[str],
    output_root: str | Path,
    summary: ZoneWriteSummary,
) -> None:
    """Write one bounded batch into all, sampled, and unsampled partitions."""
    if not batch:
        return
    output_root = Path(output_root)
    for record in batch:
        if str(record["grid_id"]) != summary.grid_id:
            raise ValueError("each write_zone_records call must contain one grid_id")
        validate_patch_record(record, GridSpec())

    partitions = {
        "all": batch,
        "sampled": [record for record in batch if str(record["parent_key"]) in sampled_keys],
        "unsampled": [record for record in batch if str(record["parent_key"]) not in sampled_keys],
    }
    staging_root = output_root.parent / f".{output_root.name}.batch-{uuid.uuid4().hex}"
    staged_parquet_paths: list[Path] = []
    staged_shapefile_paths: list[Path] = []
    try:
        for partition, partition_records in partitions.items():
            if not partition_records:
                continue
            partition_dir = staging_root / partition / summary.grid_id
            partition_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = partition_dir / f"part-{summary.batch_count:05d}.parquet"
            _geoparquet_records(partition_records, sampled_keys).to_parquet(
                parquet_path, index=False, compression="zstd"
            )
            staged_parquet_paths.append(parquet_path)
            staged_shapefile_paths.extend(
                _stage_shapefile_partition(
                    partition_records,
                    sampled_keys,
                    staging_root,
                    summary.grid_id,
                    partition,
                    summary.batch_count,
                )
            )
        _publish_staged_files(staging_root, output_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    summary.all_count += len(batch)
    summary.sampled_count += len(partitions["sampled"])
    summary.unsampled_count += len(partitions["unsampled"])
    summary.parquet_parts.extend(
        str(output_root / path.relative_to(staging_root)) for path in staged_parquet_paths
    )
    summary.shapefile_parts.extend(
        str(output_root / path.relative_to(staging_root)) for path in staged_shapefile_paths
    )
    summary.batch_count += 1


def write_zone_records(
    records: Iterable[Mapping[str, Any]],
    sampled_keys: set[str],
    output_root: str | Path,
    batch_size: int = 100_000,
) -> ZoneWriteSummary:
    """Stream one UTM zone's parent records to disjoint output partitions."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(records)
    first = next(iterator)
    summary = ZoneWriteSummary(grid_id=str(first["grid_id"]))
    batch = [first]
    for record in iterator:
        batch.append(record)
        if len(batch) == batch_size:
            write_zone_batch(batch, sampled_keys, output_root, summary)
            batch.clear()
    if batch:
        write_zone_batch(batch, sampled_keys, output_root, summary)
    if summary.all_count != summary.sampled_count + summary.unsampled_count:
        raise ValueError(f"partition mismatch for {summary.grid_id}: {summary}")
    return summary
