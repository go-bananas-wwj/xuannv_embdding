from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/materialize_china_v1_shard.py"
SPEC = importlib.util.spec_from_file_location("china_v1_materialize", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_patch_bounds_reconstruct_the_grid_cell() -> None:
    patch = MODULE._patch_from_record({"patch_id": "p", "grid_epsg": 32643, "grid_col": 312, "grid_row": 3346})
    assert patch.bounds == (399360.0, 4282880.0, 400640.0, 4284160.0)


def test_s2_cloud_mask_rejects_cloud_and_keeps_land() -> None:
    stack = np.ones((12, 2, 2), dtype=np.float32)
    stack[-1] = np.array([[4, 8], [6, 3]], dtype=np.float32)
    assert MODULE._scene_valid_mask("s2", stack).tolist() == [[True, False], [True, False]]


def test_landsat_qa_mask_uses_bits_without_rescaling() -> None:
    stack = np.ones((7, 2, 2), dtype=np.float32)
    stack[-1] = np.array([[0, 1], [16, 32]], dtype=np.float32)
    assert MODULE._scene_valid_mask("landsat", stack).tolist() == [[True, False], [False, True]]


def test_load_available_scenes_skips_bad_candidate(monkeypatch) -> None:
    candidates = [{"id": "bad"}, {"id": "first-good"}, {"id": "second-good"}]
    def fake_load(_source, item, _patch):
        if item["id"] == "bad":
            raise ValueError("bbox edge")
        return np.ones((2, 128, 128), dtype=np.float32)
    monkeypatch.setattr(MODULE, "_load_scene", fake_load)
    selected, scenes, rejected = MODULE._load_available_scenes("s1", candidates, object(), 2)
    assert [item["id"] for item in selected] == ["first-good", "second-good"]
    assert len(scenes) == 2
    assert rejected[0]["item_id"] == "bad"
