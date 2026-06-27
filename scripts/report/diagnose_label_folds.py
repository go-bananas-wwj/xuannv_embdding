#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

MONTH_SUFFIX_RE = re.compile(r"^(?P<patch>.+)_(?P<month>\d{6})$")
WORLDCOVER_RE = re.compile(r"^worldcover_\d{8}_(?P<patch>.+)$")

DEFAULT_TASKS = [
    (
        "construction",
        Path("/data/xuannv_embedding/processed/haidian/labels/construction"),
        "binary",
    ),
    (
        "building_change",
        Path("/data/xuannv_embedding/processed/harbin/labels/building_change"),
        "binary",
    ),
    (
        "farm_change",
        Path("/data/xuannv_embedding/processed/harbin/labels/farm_change"),
        "binary",
    ),
    (
        "rubbish",
        Path("/data/xuannv_embedding/processed/harbin/labels/rubbish"),
        "binary",
    ),
    (
        "construction_joint",
        Path("/data/xuannv_embedding/processed/construction_joint_v2"),
        "binary",
    ),
    (
        "haidian_worldcover",
        Path("/data/xuannv_embedding/processed/haidian/labels/worldcover"),
        "multiclass",
    ),
    (
        "harbin_worldcover",
        Path("/data/xuannv_embedding/processed/harbin/labels/worldcover"),
        "multiclass",
    ),
]


def valid_yyyymm(value: str) -> bool:
    year = int(value[:4])
    month = int(value[4:])
    return 1900 <= year <= 2100 and 1 <= month <= 12


def mask_dir_for(root: Path) -> Path:
    masks = root / "masks"
    return masks if masks.exists() else root


def patch_id_from_file(path: Path, kind: str) -> str:
    stem = path.stem
    if kind == "multiclass":
        match = WORLDCOVER_RE.match(stem)
        if match is not None:
            return match.group("patch")
    match = MONTH_SUFFIX_RE.match(stem)
    if match is not None and valid_yyyymm(match.group("month")):
        return match.group("patch")
    return stem


def resolve_mask_path(mask_dir: Path, patch_id: str) -> Path | None:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        wc_candidates = sorted(mask_dir.glob(f"worldcover_*_{patch_id}.tif"))
        return wc_candidates[-1] if wc_candidates else None

    def month_key(path: Path) -> int:
        match = MONTH_SUFFIX_RE.match(path.stem)
        if match is None or not valid_yyyymm(match.group("month")):
            return -1
        return int(match.group("month"))

    return max(candidates, key=month_key)


def read_mask_stats(path: Path, kind: str) -> dict[str, Any]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    valid = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        valid &= arr != nodata
    total_pixels = int(valid.sum())
    if kind == "binary":
        positive_pixels = int(((arr > 0) & valid).sum())
        return {
            "total_pixels": total_pixels,
            "positive_pixels": positive_pixels,
            "positive_ratio": positive_pixels / total_pixels if total_pixels else 0.0,
            "class_histogram": {},
        }
    values, counts = np.unique(arr[valid], return_counts=True)
    hist = {str(int(v)): int(c) for v, c in zip(values, counts)}
    nonzero_pixels = int(sum(c for v, c in hist.items() if v != "0"))
    return {
        "total_pixels": total_pixels,
        "positive_pixels": nonzero_pixels,
        "positive_ratio": nonzero_pixels / total_pixels if total_pixels else 0.0,
        "class_histogram": hist,
    }


def load_split(root: Path) -> dict[str, Any] | None:
    split_path = root / "split_5fold.json"
    if not split_path.exists():
        return None
    return json.loads(split_path.read_text(encoding="utf-8"))


def summarize_split(
    task_name: str,
    split: dict[str, Any] | None,
    patch_stats_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if split is None:
        return []
    rows = []
    for fold_info in split.get("folds", []):
        fold = int(fold_info["fold"])
        for split_name in ("train", "val", "test"):
            patch_ids = list(fold_info.get(split_name, []))
            stats = [patch_stats_by_id.get(pid) for pid in patch_ids]
            missing = sum(1 for s in stats if s is None)
            present = [s for s in stats if s is not None]
            total_pixels = sum(int(s["total_pixels"]) for s in present)
            positive_pixels = sum(int(s["positive_pixels"]) for s in present)
            positive_patches = sum(1 for s in present if int(s["positive_pixels"]) > 0)
            rows.append(
                {
                    "task": task_name,
                    "fold": fold,
                    "split": split_name,
                    "patches": len(patch_ids),
                    "present_masks": len(present),
                    "missing_masks": missing,
                    "positive_patches": positive_patches,
                    "positive_pixels": positive_pixels,
                    "total_pixels": total_pixels,
                    "positive_patch_ratio": positive_patches / len(patch_ids)
                    if patch_ids
                    else 0.0,
                    "positive_pixel_ratio": positive_pixels / total_pixels
                    if total_pixels
                    else 0.0,
                }
            )
    return rows


def parse_task(value: str) -> tuple[str, Path, str]:
    try:
        name, rest = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Task must be name=/path[:binary|multiclass]"
        ) from exc
    if ":" in rest:
        raw_path, kind = rest.rsplit(":", 1)
    else:
        raw_path, kind = rest, "binary"
    if kind not in {"binary", "multiclass"}:
        raise argparse.ArgumentTypeError("Task kind must be binary or multiclass")
    return name, Path(raw_path), kind


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown(
    path: Path,
    task_summaries: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
) -> None:
    lines = ["# P1 Label/Fold Diagnostics", ""]
    lines.append("## Task Availability")
    lines.append("")
    lines.append("| task | kind | root | masks | split | status |")
    lines.append("|---|---:|---|---:|---:|---|")
    for row in availability_rows:
        lines.append(
            f"| {row['task']} | {row['kind']} | `{row['root']}` | "
            f"{row['mask_count']} | {row['has_split']} | {row['status']} |"
        )
    lines.append("")
    lines.append("## Task Summary")
    lines.append("")
    lines.append(
        "| task | kind | patches | positive patches | positive pixels | total pixels | positive pixel ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in task_summaries:
        lines.append(
            f"| {row['task']} | {row['kind']} | {row['patches']} | "
            f"{row['positive_patches']} | {row['positive_pixels']} | "
            f"{row['total_pixels']} | {row['positive_pixel_ratio']:.8f} |"
        )
    if fold_rows:
        lines.append("")
        lines.append("## Fold Summary")
        lines.append("")
        lines.append(
            "| task | fold | split | patches | present | missing | positive patches | positive pixels | positive pixel ratio |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in fold_rows:
            lines.append(
                f"| {row['task']} | {row['fold']} | {row['split']} | "
                f"{row['patches']} | {row['present_masks']} | {row['missing_masks']} | "
                f"{row['positive_patches']} | {row['positive_pixels']} | "
                f"{row['positive_pixel_ratio']:.8f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        action="append",
        type=parse_task,
        help="Override tasks. Format: name=/path[:binary|multiclass]. May repeat.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v2_202512_202605/diagnostics"),
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or "p1_label_fold_diagnostics_" + datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.task if args.task else DEFAULT_TASKS
    availability_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    class_counter_by_task: dict[str, Counter[str]] = defaultdict(Counter)

    for task_name, root, kind in tasks:
        mask_dir = mask_dir_for(root)
        mask_paths = sorted(mask_dir.glob("*.tif")) if mask_dir.exists() else []
        split = load_split(root)
        status = "ok" if mask_paths else "missing_masks"
        availability_rows.append(
            {
                "task": task_name,
                "kind": kind,
                "root": str(root),
                "mask_dir": str(mask_dir),
                "mask_count": len(mask_paths),
                "has_split": split is not None,
                "status": status,
            }
        )
        patch_stats_by_id: dict[str, dict[str, Any]] = {}
        for mask_path in mask_paths:
            patch_id = patch_id_from_file(mask_path, kind)
            stats = read_mask_stats(mask_path, kind)
            row = {
                "task": task_name,
                "kind": kind,
                "patch_id": patch_id,
                "mask_path": str(mask_path),
                **stats,
            }
            patch_rows.append({k: v for k, v in row.items() if k != "class_histogram"})
            patch_stats_by_id[patch_id] = row
            class_counter_by_task[task_name].update(stats["class_histogram"])

        fold_rows.extend(summarize_split(task_name, split, patch_stats_by_id))
        total_pixels = sum(int(s["total_pixels"]) for s in patch_stats_by_id.values())
        positive_pixels = sum(
            int(s["positive_pixels"]) for s in patch_stats_by_id.values()
        )
        positive_patches = sum(
            1 for s in patch_stats_by_id.values() if int(s["positive_pixels"]) > 0
        )
        task_summaries.append(
            {
                "task": task_name,
                "kind": kind,
                "patches": len(patch_stats_by_id),
                "positive_patches": positive_patches,
                "positive_pixels": positive_pixels,
                "total_pixels": total_pixels,
                "positive_pixel_ratio": positive_pixels / total_pixels
                if total_pixels
                else 0.0,
            }
        )

    class_rows = []
    for task_name, counter in sorted(class_counter_by_task.items()):
        total = sum(counter.values())
        for class_id, pixels in sorted(counter.items(), key=lambda kv: int(kv[0])):
            class_rows.append(
                {
                    "task": task_name,
                    "class_id": class_id,
                    "pixels": pixels,
                    "ratio": pixels / total if total else 0.0,
                }
            )

    write_csv(
        out_dir / "availability.csv",
        availability_rows,
        ["task", "kind", "root", "mask_dir", "mask_count", "has_split", "status"],
    )
    write_csv(
        out_dir / "patch_stats.csv",
        patch_rows,
        [
            "task",
            "kind",
            "patch_id",
            "mask_path",
            "total_pixels",
            "positive_pixels",
            "positive_ratio",
        ],
    )
    write_csv(
        out_dir / "fold_summary.csv",
        fold_rows,
        [
            "task",
            "fold",
            "split",
            "patches",
            "present_masks",
            "missing_masks",
            "positive_patches",
            "positive_pixels",
            "total_pixels",
            "positive_patch_ratio",
            "positive_pixel_ratio",
        ],
    )
    write_csv(out_dir / "class_histogram.csv", class_rows, ["task", "class_id", "pixels", "ratio"])
    summary = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "availability": availability_rows,
        "task_summary": task_summaries,
        "fold_summary": fold_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir / "REPORT.md", task_summaries, fold_rows, availability_rows)
    print(out_dir)


if __name__ == "__main__":
    main()
