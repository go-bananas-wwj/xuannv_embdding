#!/usr/bin/env python3
"""Enumerate the canonical nationwide 1,280 m parent grid."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry

PARENT_SIDE_METERS = 1280
CHINA_OWNER_EPSGS = frozenset(range(32643, 32654))
SHAPEFILE_MAX_BYTES = 1_800_000_000


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
        row["sampled"] = str(record["parent_key"]) in sampled_keys
        row["geometry"] = _wgs84_geometry(record)
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


def _shapefile_size(path: Path) -> int:
    return sum(
        path.with_suffix(suffix).stat().st_size
        for suffix in (".shp", ".shx", ".dbf")
        if path.with_suffix(suffix).exists()
    )


def _remove_shapefile(path: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        path.with_suffix(suffix).unlink(missing_ok=True)


def _write_shapefile_partition(
    records: list[Mapping[str, Any]],
    sampled_keys: set[str],
    output_root: Path,
    grid_id: str,
    partition: str,
) -> None:
    if not records:
        return
    path = output_root / partition / grid_id / f"{grid_id}_{partition}.shp"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _shapefile_records(records, sampled_keys)
    staged_path = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.shp")
    try:
        frame.to_file(staged_path, index=False)
        projected_size = _shapefile_size(path) + _shapefile_size(staged_path)
        if projected_size > SHAPEFILE_MAX_BYTES:
            raise ValueError(
                f"Shapefile size cap exceeded for {grid_id}/{partition}: "
                f"{projected_size} bytes > {SHAPEFILE_MAX_BYTES} bytes"
            )
        if path.exists():
            frame.to_file(path, index=False, mode="a")
        else:
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                staged_file = staged_path.with_suffix(suffix)
                if staged_file.exists():
                    staged_file.replace(path.with_suffix(suffix))
    finally:
        _remove_shapefile(staged_path)


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
    part_index = summary.batch_count
    for partition, records in partitions.items():
        if not records:
            continue
        partition_dir = output_root / partition / summary.grid_id
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / f"part-{part_index:05d}.parquet"
        _geoparquet_records(records, sampled_keys).to_parquet(parquet_path, index=False)
        summary.parquet_parts.append(str(parquet_path))
        _write_shapefile_partition(records, sampled_keys, output_root, summary.grid_id, partition)

    summary.all_count += len(batch)
    summary.sampled_count += len(partitions["sampled"])
    summary.unsampled_count += len(partitions["unsampled"])
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
