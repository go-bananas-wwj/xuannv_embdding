from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/audit_osm_weak_semantics.py"
SPEC = importlib.util.spec_from_file_location("osm_weak_semantics_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_categories_merge_related_osm_values_without_background_assumption() -> None:
    assert {"road"} <= MODULE.categories_for_tags({"highway": "residential"})
    assert {"road", "construction"} <= MODULE.categories_for_tags({"highway": "construction"})
    assert {"education"} <= MODULE.categories_for_tags({"amenity": "university"})
    assert {"sports"} <= MODULE.categories_for_tags({"leisure": "pitch"})
    assert MODULE.categories_for_tags({}) == set()
