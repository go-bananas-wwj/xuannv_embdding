#!/usr/bin/env python3
"""将实验配置 YAML 中的 `_base_` 继承展开为自包含的完整配置。

遍历给定目录下的所有 `.yaml` / `.yml` 文件；如果文件顶层包含 `_base_` 字段，
则递归加载基配置并合并，最后将合并后的完整配置写回原文件，同时移除 `_base_` 字段。

用法示例：
    python tools/flatten_configs.py configs downstreams/configs \
        .worktrees/feat-multitask-downstream/configs \
        .worktrees/feat-multitask-downstream/downstreams/configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base；保留 base 中原有的键顺序，新键追加到末尾。"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_resolved(path: Path, visited: set[str] | None = None) -> dict:
    """加载 YAML 并递归解析 `_base_` 继承。"""
    if visited is None:
        visited = set()

    abs_path = path.resolve()
    key = str(abs_path)
    if key in visited:
        raise ValueError(f"检测到循环继承: {path}")
    visited.add(key)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")

    base_name = data.pop("_base_", None)
    if base_name is None:
        return data

    base_path = path.parent / base_name
    if not base_path.is_file():
        raise FileNotFoundError(f"基配置文件不存在: {base_path} (由 {path} 引用)")

    base_data = load_resolved(base_path, visited)
    merged = deep_merge(base_data, data)
    return merged


def flatten_file(path: Path, dry_run: bool = False) -> bool:
    """如果文件包含 `_base_`，则展开并写回。返回是否发生修改。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict) or "_base_" not in raw:
        return False

    resolved = load_resolved(path)

    if not dry_run:
        # 原子写入：先写临时文件，再替换原文件
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(resolved, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        tmp.replace(path)

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将实验配置 YAML 中的 _base_ 继承展开")
    parser.add_argument("directories", nargs="+", type=Path, help="要处理的配置目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印会修改的文件，不实际写入")
    args = parser.parse_args(argv)

    modified: list[Path] = []
    for directory in args.directories:
        if not directory.is_dir():
            print(f"跳过非目录项: {directory}", file=sys.stderr)
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in (".yaml", ".yml"):
                continue
            try:
                changed = flatten_file(path, dry_run=args.dry_run)
            except Exception as exc:
                print(f"处理失败 {path}: {exc}", file=sys.stderr)
                continue
            if changed:
                action = "[dry-run] 将展开" if args.dry_run else "已展开"
                print(f"{action}: {path}")
                modified.append(path)

    print(f"\n总计: {len(modified)} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
