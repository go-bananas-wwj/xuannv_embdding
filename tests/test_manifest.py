from __future__ import annotations

import json
from pathlib import Path

from xuannv_embedding.utils.manifest import generate_manifest, load_manifest, save_manifest


def test_generate_manifest_with_missing_source(tmp_path: Path) -> None:
    """多源 manifest 生成：部分 source 缺失时对应字段为 None。"""
    processed = tmp_path / "processed"
    (processed / "s2").mkdir(parents=True)
    (processed / "s1").mkdir(parents=True)

    # 基准 source s2 有两个 patch
    (processed / "s2" / "s2_20250129_p000_r000.tif").write_text("s2")
    (processed / "s2" / "s2_20250129_p000_r001.tif").write_text("s2")
    # s1 只匹配到其中一个
    (processed / "s1" / "s1_20250129_p000_r000.tif").write_text("s1")

    manifest = generate_manifest(processed, ["s2", "s1"])

    assert len(manifest) == 2
    entry_by_id = {entry["patch_id"]: entry for entry in manifest}
    assert set(entry_by_id.keys()) == {"20250129_p000_r000", "20250129_p000_r001"}

    entry0 = entry_by_id["20250129_p000_r000"]
    assert entry0["s2"] == [Path("s2/s2_20250129_p000_r000.tif")]
    assert entry0["s1"] == [Path("s1/s1_20250129_p000_r000.tif")]

    entry1 = entry_by_id["20250129_p000_r001"]
    assert entry1["s2"] == [Path("s2/s2_20250129_p000_r001.tif")]
    assert entry1["s1"] is None


def test_generate_manifest_cross_source_matching(tmp_path: Path) -> None:
    """不同 source 下具有相同 patch_id 的文件应被归到同一条目。"""
    processed = tmp_path / "processed"
    (processed / "s2").mkdir(parents=True)
    (processed / "s1").mkdir(parents=True)

    (processed / "s2" / "s2_20250215_p001_r001.tif").write_text("s2")
    (processed / "s1" / "s1_20250215_p001_r001.tif").write_text("s1")

    manifest = generate_manifest(processed, ["s2", "s1"])
    assert len(manifest) == 1
    assert manifest[0]["patch_id"] == "20250215_p001_r001"
    assert manifest[0]["s2"] == [Path("s2/s2_20250215_p001_r001.tif")]
    assert manifest[0]["s1"] == [Path("s1/s1_20250215_p001_r001.tif")]


def test_save_and_load_manifest(tmp_path: Path) -> None:
    """manifest 保存为 JSON 后，加载应将路径字符串还原为 Path。"""
    processed = tmp_path / "processed"
    (processed / "s2").mkdir(parents=True)
    (processed / "s2" / "s2_20250129_p000_r000.tif").write_text("s2")

    manifest = generate_manifest(processed, ["s2"])
    output_path = tmp_path / "manifest.json"
    save_manifest(manifest, output_path)

    assert output_path.exists()
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw[0]["patch_id"] == "20250129_p000_r000"
    assert raw[0]["s2"] == ["s2/s2_20250129_p000_r000.tif"]

    loaded = load_manifest(output_path)
    assert loaded[0]["patch_id"] == "20250129_p000_r000"
    assert loaded[0]["s2"] == [Path("s2/s2_20250129_p000_r000.tif")]


def test_generate_manifest_empty_sources(tmp_path: Path) -> None:
    """空 sources 列表应返回空 manifest。"""
    manifest = generate_manifest(tmp_path / "processed", [])
    assert manifest == []


def test_generate_manifest_missing_base_dir(tmp_path: Path) -> None:
    """基准 source 目录不存在时应返回空列表且不抛出异常。"""
    manifest = generate_manifest(tmp_path / "processed", ["s2"])
    assert manifest == []


def test_generate_manifest_missing_source_dir(tmp_path: Path) -> None:
    """某个非基准 source 目录缺失时，对应字段应为 None。"""
    processed = tmp_path / "processed"
    (processed / "s2").mkdir(parents=True)
    (processed / "s2" / "s2_20250129_p000_r000.tif").write_text("s2")

    manifest = generate_manifest(processed, ["s2", "landsat"])
    assert len(manifest) == 1
    assert manifest[0]["s2"] == [Path("s2/s2_20250129_p000_r000.tif")]
    assert manifest[0]["landsat"] is None


def test_generate_manifest_different_dates_same_grid(tmp_path: Path) -> None:
    """不同 source 的观测日期不一致时，应通过网格编号归到同一条目。"""
    processed = tmp_path / "processed"
    (processed / "s2").mkdir(parents=True)
    (processed / "worldcover").mkdir(parents=True)

    (processed / "s2" / "s2_20250129_p000_r000.tif").write_text("s2")
    (processed / "worldcover" / "worldcover_20230101_p000_r000.tif").write_text("wc")

    manifest = generate_manifest(
        processed,
        ["s2", "worldcover"],
        source_dirs={"s2": "s2", "worldcover": "worldcover"},
    )
    assert len(manifest) == 1
    entry = manifest[0]
    assert entry["patch_id"] == "20250129_p000_r000"
    assert entry["s2"] == [Path("s2/s2_20250129_p000_r000.tif")]
    assert entry["worldcover"] == [Path("worldcover/worldcover_20230101_p000_r000.tif")]
