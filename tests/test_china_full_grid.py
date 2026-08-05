from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from pyproj import Transformer
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
