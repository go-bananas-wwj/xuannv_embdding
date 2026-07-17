from __future__ import annotations

import importlib.util
import json
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


def test_s1_composite_excludes_invalid_scene_values() -> None:
    invalid = np.zeros((2, 2, 2), dtype=np.float32)
    valid = np.full((2, 2, 2), 2.0, dtype=np.float32)
    image, mask, fractions = MODULE._composite("s1", [invalid, valid])
    assert fractions == [0.0, 1.0]
    assert mask.tolist() == [[1, 1], [1, 1]]
    assert np.all(image == 2.0)


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


def test_catalog_index_returns_only_intersecting_items(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    rows = [
        {"id": "near", "bbox": [100.0, 30.0, 101.0, 31.0], "assets": {}},
        {"id": "far", "bbox": [110.0, 30.0, 111.0, 31.0], "assets": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    index = MODULE.CatalogIndex(path)
    assert [item["id"] for item in index.query((100.4, 30.4, 100.6, 30.6))] == ["near"]


def test_select_items_filters_missing_assets(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    required = {name: {"href": "https://example.test/x.tif"} for name in MODULE.SOURCES["s1"]["assets"]}
    rows = [
        {"id": "usable", "bbox": [100.0, 30.0, 101.0, 31.0], "assets": required, "properties": {}},
        {"id": "incomplete", "bbox": [100.0, 30.0, 101.0, 31.0], "assets": {"vv": {}}, "properties": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    patch = MODULE.Patch("p", 32647, (0, 0, 1, 1), (100.4, 30.4, 100.6, 30.6))
    assert [item["id"] for item in MODULE._select_items(MODULE.CatalogIndex(path), patch, "s1", 4)] == ["usable"]


def test_load_catalogs_builds_requested_source_month_pairs(tmp_path) -> None:
    for source in MODULE.SOURCES:
        path = tmp_path / source / "2025-04"
        path.mkdir(parents=True)
        (path / "items.jsonl").write_text("", encoding="utf-8")
    catalogs = MODULE.load_catalogs(tmp_path, ["2025-04"])
    assert set(catalogs) == {(source, "2025-04") for source in MODULE.SOURCES}
