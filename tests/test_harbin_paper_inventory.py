from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import geopandas as gpd
from shapely.geometry import Point


SCRIPT_PATH = Path(__file__).parent.parent / "scripts/eval/build_harbin_paper_inventory.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("harbin_paper_inventory", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {SCRIPT_PATH}")
    module = types.ModuleType(spec.name)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_map_grid_ids_to_manifest_ids_preserves_geometry(tmp_path: Path) -> None:
    """空间 split 必须使用 embedding/标签实际使用的 harbin_patch ID。"""
    module = _load_module()
    grid = tmp_path / "grid.json"
    manifest = tmp_path / "manifest.json"
    grid.write_text(
        json.dumps([{"patch_id": "patch_000001", "bounds": [0, 0, 1280, 1280]}]),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps([{"patch_id": "harbin_patch_000001", "source_patch_id": "patch_000001"}]),
        encoding="utf-8",
    )

    mapped = module.map_grid_ids_to_manifest_ids(grid, manifest)

    assert mapped == [{"patch_id": "harbin_patch_000001", "bounds": [0, 0, 1280, 1280]}]


def test_map_grid_ids_to_manifest_ids_rejects_missing_geometry_mapping(tmp_path: Path) -> None:
    """缺失 source_patch_id 映射时必须停止，不能混用两套 patch ID。"""
    module = _load_module()
    grid = tmp_path / "grid.json"
    manifest = tmp_path / "manifest.json"
    grid.write_text(json.dumps([{"patch_id": "patch_000001", "bounds": [0, 0, 1, 1]}]), encoding="utf-8")
    manifest.write_text(json.dumps([{"patch_id": "harbin_patch_000002", "source_patch_id": "patch_000002"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="无法映射"):
        module.map_grid_ids_to_manifest_ids(grid, manifest)


def test_local_aef_index_does_not_receive_remote_storage_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """冻结到本地的 AEF 索引必须可读取，不能把 S3 参数传给本地路径。"""
    path = Path(__file__).parent.parent / "scripts/data/prepare_aef_2025_embeddings.py"
    spec = importlib.util.spec_from_file_location("prepare_aef", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calls: list[dict[str, object]] = []

    def read_parquet(path: str, **kwargs: object) -> gpd.GeoDataFrame:
        calls.append(kwargs)
        return gpd.GeoDataFrame({"year": [2025]}, geometry=[Point(0, 0)], crs="OGC:CRS84")

    monkeypatch.setattr(module.gpd, "read_parquet", read_parquet)
    index = module.load_aef_index("/data/frozen/aef_index.parquet")

    assert len(index) == 1
    assert calls == [{}]
