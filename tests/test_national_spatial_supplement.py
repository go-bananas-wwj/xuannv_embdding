from __future__ import annotations

import importlib.util
from pathlib import Path

from shapely.geometry import box


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/build_national_spatial_supplement.py"
SPEC = importlib.util.spec_from_file_location("national_spatial_supplement", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_spatial_supplement_reaches_target_without_reusing_base_macro() -> None:
    inventory = []
    for col in range(3):
        inventory.append({
            "macro_id": f"utm50n_c{col}_r0", "grid_id": "utm50n", "grid_epsg": 32650,
            "macro_col": 32 + col, "macro_row": 330,
            "utm_bounds": [409600.0 + col * 12800, 4224000.0, 422400.0 + col * 12800, 4236800.0],
            "estimated_patch_count": 100.0, "admin1": "test",
        })
    base = [{"macro_id": "utm50n_c0_r0", "grid_id": "utm50n", "longitude": 116.0, "latitude": 39.0}]
    supplement, allocation = MODULE.build_supplement(inventory, base, box(115.0, 38.0, 118.0, 40.0), 7, 3)
    assert len(supplement) == 2
    assert allocation == {"utm50n": 2}
    assert {item["macro_id"] for item in supplement}.isdisjoint({"utm50n_c0_r0"})


def test_allocate_distributes_more_than_one_residual_round() -> None:
    groups = {"a": [{}, {}, {}, {}], "b": [{}, {}, {}, {}]}
    allocation = MODULE._allocate(5, groups)
    assert sum(allocation.values()) == 5
