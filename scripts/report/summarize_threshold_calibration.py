#!/usr/bin/env python3
"""Summarize threshold calibration for downstream benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

TASKS = (
    "construction",
    "building_change",
    "farm_change",
    "rubbish",
    "construction_joint",
)
METRICS = ("f1_0.5", "f1_at_threshold", "f1_best", "val_threshold", "best_threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_task(root: Path, task: str) -> dict[str, Any]:
    summary_path = root / task / "summary.json"
    if not summary_path.exists():
        summary_path = root / task / "summary_5fold.json"
    rows = load_rows(summary_path)
    stats = {metric: _stats(_values(rows, metric)) for metric in METRICS}
    f1_05 = _values(rows, "f1_0.5")
    f1_cal = _values(rows, "f1_at_threshold")
    f1_best = _values(rows, "f1_best")
    calibration_gain = [cal - base for cal, base in zip(f1_cal, f1_05, strict=False)]
    oracle_gap = [best - cal for best, cal in zip(f1_best, f1_cal, strict=False)]
    return {
        "num_folds": len(rows),
        "stats": stats,
        "calibration_gain": _stats(calibration_gain),
        "oracle_gap": _stats(oracle_gap),
        "folds": rows,
    }


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Threshold Calibration Summary",
        "",
        f"Benchmark root: `{report['benchmark_root']}`",
        "",
        "| Task | Folds | F1@0.5 | F1@val_thr | Gain | F1_best | Oracle gap | Val threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in report["tasks"].items():
        stats = item["stats"]
        lines.append(
            "| {task} | {folds} | {f105} | {f1cal} | {gain} | {f1best} | {gap} | {thr} |".format(
                task=task,
                folds=item["num_folds"],
                f105=fmt(stats["f1_0.5"]["mean"]),
                f1cal=fmt(stats["f1_at_threshold"]["mean"]),
                gain=fmt(item["calibration_gain"]["mean"]),
                f1best=fmt(stats["f1_best"]["mean"]),
                gap=fmt(item["oracle_gap"]["mean"]),
                thr=fmt(stats["val_threshold"]["mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `F1@val_thr` is the test F1 measured at the threshold selected on validation.",
            "- `Gain` = `F1@val_thr - F1@0.5`.",
            "- `Oracle gap` = `F1_best - F1@val_thr`; a large gap means "
            "validation threshold selection can still improve.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks = {
        task: summarize_task(args.benchmark_root, task)
        for task in args.tasks
        if (args.benchmark_root / task).exists()
    }
    report = {"benchmark_root": str(args.benchmark_root), "tasks": tasks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)


if __name__ == "__main__":
    main()
