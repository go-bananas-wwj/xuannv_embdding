from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer
from shapely import normalize, set_precision
from shapely.geometry import box

MODULE_PATH = Path(__file__).parents[1] / "scripts/data/china_full_grid.py"
SPEC = importlib.util.spec_from_file_location("china_full_grid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _macro() -> dict[str, int | str]:
    return {
        "grid_epsg": 32650,
        "grid_id": "utm50n",
        "macro_col": 34,
        "macro_row": 344,
    }


def _boundary():
    return box(116.28, 39.83, 116.31, 39.86)


def synthetic_parent_records(count: int) -> list[dict[str, object]]:
    """Return canonical records spanning two macro cells for writer tests."""
    spec = MODULE.GridSpec(boundary_version="test")
    first_macro = _macro()
    second_macro = {**first_macro, "macro_col": first_macro["macro_col"] + 1}
    records: list[dict[str, object]] = []
    for index in range(count):
        macro = first_macro if index < count // 2 else second_macro
        grid_col = int(macro["macro_col"]) * spec.macro_side_patches + index % 10
        grid_row = int(macro["macro_row"]) * spec.macro_side_patches + index // 10
        records.append(
            MODULE.build_patch_record(
                macro,
                grid_col,
                grid_row,
                116.28 + index * 0.001,
                39.83 + index * 0.001,
                spec,
            )
        )
    return records


def test_writer_partitions_all_sampled_and_unsampled(tmp_path: Path) -> None:
    records = synthetic_parent_records(count=12)
    sampled = {str(records[1]["parent_key"]), str(records[7]["parent_key"])}

    summary = MODULE.write_zone_records(iter(records), sampled, tmp_path, batch_size=5)

    assert summary.all_count == 12
    assert summary.sampled_count == 2
    assert summary.unsampled_count == 10
    assert summary.sampled_count + summary.unsampled_count == summary.all_count

    def read_partition(name: str) -> gpd.GeoDataFrame:
        parts = sorted((tmp_path / name / "utm50n").glob("*.parquet"))
        frames = [gpd.read_parquet(part) for part in parts]
        return gpd.GeoDataFrame(pd.concat(frames), crs="EPSG:4326")

    all_records = read_partition("all")
    sampled_records = read_partition("sampled")
    unsampled_records = read_partition("unsampled")

    assert all_records.crs.to_epsg() == 4326
    assert set(sampled_records["parent_key"]).isdisjoint(unsampled_records["parent_key"])
    assert set(sampled_records["parent_key"]) == sampled
    assert len(all_records) == len(sampled_records) + len(unsampled_records)
    assert {"patch_id", "parent_key", "schema_version", "atlas_version", "boundary_version"} <= set(
        all_records.columns
    )
    assert all_records.geometry.geom_type.eq("Polygon").all()

    shapefile_parts = sorted((tmp_path / "all" / "utm50n").glob("*.shp"))
    shapefile_records = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(part) for part in shapefile_parts]), crs="EPSG:4326"
    )
    assert shapefile_records.crs.to_epsg() == 4326
    assert len(shapefile_records) == 12
    assert {"PATCH_ID", "UTM_EPSG", "GRID_COL", "GRID_ROW", "MACRO_ID", "SAMPLED"} <= set(
        shapefile_records.columns
    )


def test_writer_rejects_shapefile_larger_than_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MODULE, "SHAPEFILE_MAX_BYTES", 1)

    with pytest.raises(ValueError, match="one Shapefile feature exceeds"):
        MODULE.write_zone_records(synthetic_parent_records(count=1), set(), tmp_path, batch_size=1)

    assert not list(tmp_path.rglob("*.shp"))
    assert not list(tmp_path.rglob("*.parquet"))


def test_writer_splits_shapefiles_at_component_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = synthetic_parent_records(count=12)
    probe = tmp_path / "probe.shp"
    MODULE._shapefile_records(records[:1], set()).to_file(probe, index=False)
    one_record_component = max(
        probe.with_suffix(suffix).stat().st_size for suffix in (".shp", ".shx", ".dbf")
    )
    component_cap = math.ceil(one_record_component / 0.9)
    monkeypatch.setattr(MODULE, "SHAPEFILE_MAX_BYTES", component_cap)

    MODULE.write_zone_records(records, set(), tmp_path / "output", batch_size=12)

    parts = sorted((tmp_path / "output" / "all" / "utm50n").glob("*.shp"))
    assert len(parts) > 1
    assert all("rowblock-" in part.name and "part-" in part.name for part in parts)
    for part in parts:
        assert all(
            part.with_suffix(suffix).stat().st_size < component_cap
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
            if part.with_suffix(suffix).exists()
        )


def test_geoparquet_has_canonical_fields_and_zstd_metadata(tmp_path: Path) -> None:
    record = synthetic_parent_records(count=1)[0]
    MODULE.write_zone_records([record], set(), tmp_path, batch_size=1)
    parquet_path = next((tmp_path / "all" / "utm50n").glob("*.parquet"))
    result = gpd.read_parquet(parquet_path)
    row = result.iloc[0]

    assert {"wgs84_bounds", "identity_hash", "footprint_hash"} <= set(result.columns)
    assert tuple(row["wgs84_bounds"]) == pytest.approx(row.geometry.bounds)
    expected_identity = (
        f"{row['atlas_version']}:{row['grid_epsg']}:{row['grid_col']}:{row['grid_row']}"
    )
    assert row["identity_hash"] == hashlib.sha256(expected_identity.encode("utf-8")).hexdigest()
    expected_footprint = normalize(set_precision(row.geometry, 1e-9)).wkb
    assert row["footprint_hash"] == hashlib.sha256(expected_footprint).hexdigest()

    metadata = pq.ParquetFile(parquet_path).metadata
    assert {
        metadata.row_group(0).column(column).compression
        for column in range(metadata.row_group(0).num_columns)
    } == {"ZSTD"}


def test_writer_keeps_output_empty_when_a_later_partition_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = synthetic_parent_records(count=2)
    sampled = {str(records[0]["parent_key"])}
    original_to_parquet = gpd.GeoDataFrame.to_parquet
    calls = 0

    def fail_second_parquet_write(self, path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second partition failure")
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail_second_parquet_write)

    with pytest.raises(OSError, match="synthetic second partition failure"):
        MODULE.write_zone_records(records, sampled, tmp_path, batch_size=2)

    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.shp"))


def test_cli_writes_requested_zone_from_small_local_inputs(tmp_path: Path) -> None:
    record = next(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), MODULE.GridSpec()))
    boundary_path = tmp_path / "boundary.geojson"
    macro_path = tmp_path / "macros.json"
    sampled_path = tmp_path / "sampled.json"
    output_root = tmp_path / "output"
    boundary_frame = gpd.GeoDataFrame(geometry=[_boundary()], crs="EPSG:4326")
    boundary_frame.to_file(boundary_path, driver="GeoJSON")
    macro_path.write_text(
        '[{"grid_epsg": 32650, "grid_id": "utm50n", "macro_col": 34, "macro_row": 344}]',
        encoding="utf-8",
    )
    sampled_key = record["parent_key"]
    sampled_path.write_text(f'["{sampled_key}"]', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/build_china_full_grid.py",
            "--boundary",
            str(boundary_path),
            "--macrocells",
            str(macro_path),
            "--sampled-parent-keys",
            str(sampled_path),
            "--output-root",
            str(output_root),
            "--grid-id",
            "utm50n",
            "--batch-size",
            "100",
        ],
        cwd=MODULE_PATH.parents[2],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"sampled_count": 1' in result.stdout
    assert list((output_root / "all" / "utm50n").glob("*.shp"))


def test_grid_spec_rejects_noncanonical_parent_cell_size() -> None:
    with pytest.raises(ValueError, match="side_m must be exactly 1280"):
        MODULE.GridSpec(side_m=640)


def test_enumerator_emits_exact_1280m_parent_cells() -> None:
    spec = MODULE.GridSpec(side_m=1280, macro_side_patches=10, boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert records
    for record in records:
        minx, miny, maxx, maxy = record["utm_bounds"]
        assert maxx - minx == 1280
        assert maxy - miny == 1280
        assert minx == record["grid_col"] * 1280
        assert miny == record["grid_row"] * 1280
        assert record["parent_key"] == (
            f'{record["grid_epsg"]}:{record["grid_col"]}:{record["grid_row"]}'
        )


def test_enumerator_records_have_deterministic_ids() -> None:
    spec = MODULE.GridSpec(boundary_version="test")

    first = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))
    second = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert [record["patch_id"] for record in first] == [record["patch_id"] for record in second]
    assert all(record["patch_id"] == f"parent_{record['parent_key']}" for record in first)


def test_enumerator_gives_each_record_a_unique_macro_local_key() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    keys = {
        (record["macro_id"], record["macro_local_col"], record["macro_local_row"])
        for record in records
    }
    assert len(keys) == len(records)
    assert all(0 <= record["macro_local_col"] < 10 for record in records)
    assert all(0 <= record["macro_local_row"] < 10 for record in records)


def test_enumerator_filters_cells_owned_by_another_utm_zone() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    to_utm49 = Transformer.from_crs(4326, 32649, always_xy=True)
    easting, northing = to_utm49.transform(117.0, 39.84)
    macro = {
        "grid_epsg": 32649,
        "grid_id": "utm49n",
        "macro_col": math.floor(easting / 12800),
        "macro_row": math.floor(northing / 12800),
    }
    boundary = box(116.9, 39.7, 117.1, 40.0)

    records = list(MODULE.enumerate_macro_patch_records(macro, boundary, spec))

    assert records == []


def test_enumerator_rejects_epsg_outside_china_owner_range() -> None:
    macro = {**_macro(), "grid_epsg": 32642, "grid_id": "utm42n"}

    with pytest.raises(ValueError, match="32643 through 32653"):
        list(MODULE.enumerate_macro_patch_records(macro, _boundary(), MODULE.GridSpec()))


def test_enumerator_preserves_wgs84_longitude_latitude_axis_order() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    records = list(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))

    assert records
    for record in records:
        assert 116.28 <= record["longitude"] <= 116.31
        assert 39.83 <= record["latitude"] <= 39.86


def test_validate_patch_record_rejects_misaligned_bounds() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    record = next(MODULE.enumerate_macro_patch_records(_macro(), _boundary(), spec))
    record["utm_bounds"][0] += 1

    with pytest.raises(ValueError, match="utm_bounds"):
        MODULE.validate_patch_record(record, spec)


def test_validate_patch_record_rejects_epsg_outside_china_owner_range() -> None:
    spec = MODULE.GridSpec(boundary_version="test")
    to_utm42 = Transformer.from_crs(4326, 32642, always_xy=True)
    easting, northing = to_utm42.transform(69.0, 30.0)
    grid_col = math.floor(easting / 1280)
    grid_row = math.floor(northing / 1280)
    macro = {
        "grid_epsg": 32642,
        "grid_id": "utm42n",
        "macro_col": grid_col // 10,
        "macro_row": grid_row // 10,
    }
    record = MODULE.build_patch_record(macro, grid_col, grid_row, 69.0, 30.0, spec)

    with pytest.raises(ValueError, match="32643 through 32653"):
        MODULE.validate_patch_record(record, spec)
