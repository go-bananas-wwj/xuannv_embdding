from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/select_china_v1_shard_points.py"
SPEC = importlib.util.spec_from_file_location("china_v1_shard_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selection_is_reproducible_and_unique(tmp_path) -> None:
    input_path = tmp_path / "points.jsonl"
    input_path.write_text("".join(json.dumps({"patch_id": f"p{index}"}) + "\n" for index in range(10)), encoding="utf-8")
    first = MODULE.select_points(input_path, 4, 7)
    second = MODULE.select_points(input_path, 4, 7)
    assert first == second
    assert len({point["patch_id"] for point in first}) == 4


def test_spatial_order_groups_nearby_grid_cells() -> None:
    points = [{"patch_id": "b", "grid_id": "utm48n", "grid_row": 3, "grid_col": 1}, {"patch_id": "a", "grid_id": "utm47n", "grid_row": 9, "grid_col": 1}]
    assert [point["patch_id"] for point in MODULE._spatial_order(points)] == ["a", "b"]
