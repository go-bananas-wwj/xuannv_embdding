from __future__ import annotations

import importlib.util
from pathlib import Path

from shapely.geometry import box


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/build_national_coastal_supplement.py"
SPEC = importlib.util.spec_from_file_location("national_coastal_supplement", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coastal_candidate_requires_ocean_intersection() -> None:
    macro = {
        "grid_epsg": 32650, "utm_bounds": [409600.0, 4224000.0, 422400.0, 4236800.0],
        "land_fraction": 0.5,
    }
    macro_geometry = MODULE.macro_wgs84(macro)
    assert MODULE._coastal_candidates([macro], macro_geometry.buffer(0.01)) == [macro]
    assert MODULE._coastal_candidates([macro], box(70.0, 10.0, 71.0, 11.0)) == []
