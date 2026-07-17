from __future__ import annotations

import importlib.util
from pathlib import Path

from shapely.geometry import box


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/visualize_national_sampling.py"
SPEC = importlib.util.spec_from_file_location("national_sampling_preview", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preview_selection_is_deterministic_and_inside_boundary() -> None:
    macro = {
        "macro_id": "utm50n_c0_r0", "grid_id": "utm50n", "grid_epsg": 32650,
        "macro_col": 32, "macro_row": 330, "utm_bounds": [409600.0, 4224000.0, 422400.0, 4236800.0],
        "estimated_patch_count": 100.0, "admin1": "test",
    }
    country = box(115.0, 38.0, 117.0, 40.0)
    first = MODULE.select_static_preview([macro], country, seed=7)
    second = MODULE.select_static_preview([macro], country, seed=7)
    assert first == second
    assert len(first) == 1
    assert country.covers(MODULE.Point(first[0]["longitude"], first[0]["latitude"]))
