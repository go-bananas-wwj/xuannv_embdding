from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/data/export_china_full_grid_qgis_shapes.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("export_china_full_grid_qgis_shapes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dissolved_boundaries_follow_cells_not_bounding_boxes() -> None:
    """A concave shard boundary must preserve the grid-cell footprint, not become a bbox."""
    module = load_script()
    cells = gpd.GeoDataFrame(
        {"shard_id": [1, 1, 1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
        ],
        crs="EPSG:4326",
    )

    regions, national = module.dissolve_region_boundaries(cells)

    shard_one = regions.loc[regions["shard_id"] == 1].geometry.iloc[0]
    assert len(regions) == 2
    assert shard_one.area == 3
    assert not shard_one.equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert national.geometry.iloc[0].area == 4
    assert national.crs == cells.crs


def test_write_grid_cells_shapefile_keeps_exact_cell_features(tmp_path: Path) -> None:
    """QGIS export must retain one feature per 1280 m grid cell and its parent key."""
    module = load_script()
    cells = gpd.GeoDataFrame(
        {
            "parent_key": ["32648:1:2", "32648:1:3"],
            "shard_id": [1, 1],
            "grid_id": ["utm48n", "utm48n"],
            "grid_col": [1, 1],
            "grid_row": [2, 3],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        ],
        crs="EPSG:4326",
    )
    destination = tmp_path / "shard_01_grid_cells.shp"

    module.write_qgis_shapefile(cells, destination)

    result = gpd.read_file(destination)
    assert len(result) == 2
    assert result["parent_key"].tolist() == ["32648:1:2", "32648:1:3"]
    assert result.geometry.iloc[0].equals(cells.geometry.iloc[0])
