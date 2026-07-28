#!/usr/bin/env python3
"""冻结哈尔滨论文空间网格，并把原始网格 ID 对齐到训练 manifest ID。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_split_builder() -> Any:
    """按仓库文件路径加载空间划分器，避免 downstreams/scripts 同名包冲突。"""
    script_path = Path(__file__).resolve().parents[1] / "data" / "build_block_paper_split.py"
    spec = importlib.util.spec_from_file_location("xuannv_block_paper_split", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载空间划分器: {script_path}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapped-grid-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()

    mapped = map_grid_ids_to_manifest_ids(args.grid, args.manifest)
    args.mapped_grid_output.parent.mkdir(parents=True, exist_ok=True)
    args.mapped_grid_output.write_text(json.dumps(mapped, indent=2) + "\n", encoding="utf-8")
    split = _load_split_builder().build_split(mapped)
    split["patch_metadata_path"] = str(args.mapped_grid_output.resolve())
    split["manifest_path"] = str(args.manifest.resolve())
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
