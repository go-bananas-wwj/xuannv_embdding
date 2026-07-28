#!/usr/bin/env python3
"""冻结哈尔滨论文空间网格，并把原始网格 ID 对齐到训练 manifest ID。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import rasterio


def _load_split_builder() -> Any:
    """按仓库文件路径加载空间划分器，避免 downstreams/scripts 同名包冲突。"""
    script_path = Path(__file__).resolve().parents[1] / "data" / "build_block_paper_split.py"
    spec = importlib.util.spec_from_file_location("xuannv_block_paper_split", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载空间划分器: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_aef_helpers() -> Any:
    """按仓库文件路径加载 AEF helper，避免同名包受 PYTHONPATH 影响。"""
    script_path = Path(__file__).resolve().parents[1] / "data" / "prepare_aef_2025_embeddings.py"
    spec = importlib.util.spec_from_file_location("xuannv_prepare_aef", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 AEF helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"{path} 必须是对象记录列表")
    return payload


def map_grid_ids_to_manifest_ids(grid_path: str | Path, manifest_path: str | Path) -> list[dict[str, Any]]:
    """将原始网格 ``source_patch_id`` 映射为哈尔滨训练/标签使用的 ``patch_id``。"""
    grid_records = _load_records(Path(grid_path))
    manifest_records = _load_records(Path(manifest_path))
    target_by_source: dict[str, str] = {}
    for record in manifest_records:
        source_patch_id = record.get("source_patch_id")
        patch_id = record.get("patch_id")
        if not isinstance(source_patch_id, str) or not isinstance(patch_id, str):
            raise ValueError("哈尔滨 manifest 每条记录必须有字符串 source_patch_id 和 patch_id")
        if source_patch_id in target_by_source:
            raise ValueError(f"manifest source_patch_id 重复: {source_patch_id}")
        target_by_source[source_patch_id] = patch_id

    mapped: list[dict[str, Any]] = []
    for record in grid_records:
        source_patch_id = record.get("patch_id")
        bounds = record.get("bounds")
        if not isinstance(source_patch_id, str) or not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError("原始哈尔滨网格记录缺少 patch_id 或 bounds")
        target_patch_id = target_by_source.get(source_patch_id)
        if target_patch_id is None:
            raise ValueError(f"无法映射原始网格 patch ID: {source_patch_id}")
        mapped.append({"patch_id": target_patch_id, "bounds": bounds})

    if len({record["patch_id"] for record in mapped}) != len(mapped):
        raise ValueError("映射后哈尔滨 patch_id 不唯一")
    return mapped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_grid_record(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        return {
            "path": str(path),
            "sha256": _sha256_file(path),
            "crs": str(dataset.crs),
            "transform": list(dataset.transform)[:6],
            "bounds": [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ],
            "shape": [dataset.height, dataset.width],
        }


def build_aef_coverage_inventory(
    manifest_path: str | Path,
    patch_ids: list[str],
    resolver: Callable[[Path], str],
) -> dict[str, Any]:
    """构建不读取任何标签的哈尔滨 AEF 2025 覆盖清单。"""
    manifest_path = Path(manifest_path)
    manifest_records = _load_records(manifest_path)
    by_patch = {str(record.get("patch_id")): record for record in manifest_records}
    if len(by_patch) != len(manifest_records):
        raise ValueError("哈尔滨 manifest patch_id 不唯一")

    records: list[dict[str, Any]] = []
    for patch_id in sorted(patch_ids):
        manifest_record = by_patch.get(patch_id)
        if manifest_record is None:
            raise ValueError(f"评测 patch 不在哈尔滨 manifest 中: {patch_id}")
        source_patch_id = manifest_record.get("source_patch_id")
        s2_paths = manifest_record.get("s2")
        if not isinstance(source_patch_id, str) or not isinstance(s2_paths, list) or not s2_paths:
            raise ValueError(f"{patch_id} 缺少 source_patch_id 或 S2 参考栅格")
        reference_path = manifest_path.parent / str(s2_paths[0])
        if not reference_path.is_file():
            raise FileNotFoundError(f"{patch_id} S2 参考栅格不存在: {reference_path}")
        records.append(
            {
                "patch_id": patch_id,
                "source_patch_id": source_patch_id,
                "reference_grid": _reference_grid_record(reference_path),
                "aef_2025_uri": resolver(reference_path),
            }
        )
    return {
        "schema_version": 1,
        "kind": "aef_harbin_annual_2025_coverage_inventory",
        "region": "harbin",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapped-grid-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--aef-index", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    args = parser.parse_args()

    mapped = map_grid_ids_to_manifest_ids(args.grid, args.manifest)
    args.mapped_grid_output.parent.mkdir(parents=True, exist_ok=True)
    args.mapped_grid_output.write_text(json.dumps(mapped, indent=2) + "\n", encoding="utf-8")
    split = _load_split_builder().build_split(mapped)
    split["patch_metadata_path"] = str(args.mapped_grid_output.resolve())
    split["manifest_path"] = str(args.manifest.resolve())
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    patch_ids = sorted(
        set().union(*(set(fold[key]) for fold in split["folds"] for key in ("train", "val", "test", "buffer")))
    )
    aef = _load_aef_helpers()
    index = aef.load_aef_index(str(args.aef_index))
    inventory = build_aef_coverage_inventory(
        args.manifest,
        patch_ids,
        resolver=lambda reference_path: aef.find_aef_cog(index, reference_path),
    )
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
