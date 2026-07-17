from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/build_national_macrocell_inventory.py"
SPEC = importlib.util.spec_from_file_location("national_macrocell_inventory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_inventory_marks_candidate_counts_as_estimates(tmp_path: Path) -> None:
    adm0 = tmp_path / "adm0.geojson"
    adm1 = tmp_path / "adm1.geojson"
    geometry = box(116.0, 39.0, 116.2, 39.2)
    gpd.GeoDataFrame({"shapeName": ["test"], "geometry": [geometry]}, crs="EPSG:4326").to_file(adm0, driver="GeoJSON")
    gpd.GeoDataFrame({"shapeName": ["test-province"], "geometry": [geometry]}, crs="EPSG:4326").to_file(adm1, driver="GeoJSON")

    records, summary = MODULE.build_inventory(adm0, adm1)
    assert records
    assert all(record["grid_epsg"] == 32650 for record in records)
    assert all(record["candidate_count_status"] == "estimate_only_requires_exact_quality_atlas" for record in records)
    assert summary["estimated_one_percent_base_samples"] > 0
