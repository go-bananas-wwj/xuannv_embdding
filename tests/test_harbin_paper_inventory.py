from __future__ import annotations

import importlib.util
import json
import os
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


def test_aef_s3_uri_has_a_stable_official_https_fallback() -> None:
    """S3 SDK 故障时仍只能落到同一 Source Cooperative 官方对象。"""
    path = Path(__file__).parent.parent / "scripts/data/prepare_aef_2025_embeddings.py"
    spec = importlib.util.spec_from_file_location("prepare_aef_https", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.s3_to_official_https(
        "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/file.tiff"
    ) == "https://data.source.coop/tge-labs/aef/v1/annual/file.tiff"


def test_aef_https_fallback_sanitizes_cann_loader_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """系统 curl 不能继承 CANN 的动态库路径。"""
    path = Path(__file__).parent.parent / "scripts/data/prepare_aef_2025_embeddings.py"
    spec = importlib.util.spec_from_file_location("prepare_aef_curl_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/lib")
    monkeypatch.setenv("LD_PRELOAD", "/usr/local/Ascend/lib/libfake.so")

    env = module.clean_curl_environment()

    assert "LD_LIBRARY_PATH" not in env
    assert "LD_PRELOAD" not in env
    assert env["PATH"] == os.environ["PATH"]


def test_build_coverage_inventory_binds_target_ids_to_source_labels_and_reference_grid(
    tmp_path: Path,
) -> None:
    """论文清单必须同时保存 embedding ID、标签 ID 和 label-free AEF 来源。"""
    module = _load_module()
    raster = tmp_path / "s2_20260401_patch_000001.tif"
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=1,
        dtype="float32",
        crs="EPSG:32652",
        transform=from_origin(500000, 5100000, 10, 10),
    ) as dst:
        dst.write(np.zeros((1, 128, 128), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "patch_id": "harbin_patch_000001",
                    "source_patch_id": "patch_000001",
                    "s2": [raster.name],
                }
            ]
        ),
        encoding="utf-8",
    )

    inventory = module.build_aef_coverage_inventory(
        manifest,
        ["harbin_patch_000001"],
        resolver=lambda _: "s3://official/2025/52N/aef.tiff",
    )

    assert inventory["region"] == "harbin"
    assert inventory["records"][0]["patch_id"] == "harbin_patch_000001"
    assert inventory["records"][0]["source_patch_id"] == "patch_000001"
    assert inventory["records"][0]["aef_2025_uri"] == "s3://official/2025/52N/aef.tiff"
    assert inventory["records"][0]["reference_grid"]["shape"] == [128, 128]
