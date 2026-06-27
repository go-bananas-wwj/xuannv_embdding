#!/usr/bin/env python3
"""Compare V2 downstream benchmark summaries against V1.0 acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

V1_BASELINE = {
    "construction": {"auc_roc": 0.7461, "f1_best": 0.2136, "miou": 0.0273},
    "building_change": {"auc_roc": 0.8828, "f1_best": 0.1163, "miou": 0.0327},
    "farm_change": {"auc_roc": 0.8876, "f1_best": 0.1208, "miou": 0.0385},
    "rubbish": {"auc_roc": 0.8876, "f1_best": 0.1208, "miou": 0.0384},
    "construction_joint": {"auc_roc": 0.8123, "f1_best": 0.1904, "miou": 0.0799},
}
PROTECTED_TASKS = ("building_change", "farm_change", "rubbish", "construction_joint")
MACRO_THRESHOLDS = {"auc_roc": 0.8433, "f1_best": 0.1524, "miou": 0.0434}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Directory containing task subdirs with summary.json or legacy summary_5fold.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output comparison JSON path.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of fold metrics")
    return data


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for metric in ("auc_roc", "f1_best", "f1_0.5", "miou"):
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            stats[metric] = {"mean": None, "std": None}
        else:
            stats[metric] = {"mean": mean(values), "std": pstdev(values)}
    stats["num_folds"] = len(rows)
    return stats


def _write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# V2 Acceptance Comparison",
        "",
        f"Overall pass: `{comparison['pass']}`",
        "",
        "| Task | AUC | F1_best | mIoU | Pass |",
        "|---|---:|---:|---:|---|",
    ]
    for task, result in comparison["tasks"].items():
        stats = result["metrics"]
        lines.append(
            "| {task} | {auc:.4f} | {f1:.4f} | {miou:.4f} | {passed} |".format(
                task=task,
                auc=stats["auc_roc"]["mean"],
                f1=stats["f1_best"]["mean"],
                miou=stats["miou"]["mean"],
                passed=result["pass"],
            )
        )
    lines.extend(
        [
            "",
            "## Macro",
            "",
            json.dumps(comparison["macro"], ensure_ascii=False, indent=2),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    tasks: dict[str, Any] = {}
    macro_values = {"auc_roc": [], "f1_best": [], "miou": []}

    for task, baseline in V1_BASELINE.items():
        summary_path = args.benchmark_root / task / "summary.json"
        if not summary_path.exists():
            summary_path = args.benchmark_root / task / "summary_5fold.json"
        if not summary_path.exists():
            tasks[task] = {"pass": False, "error": f"missing {summary_path}"}
            continue
        stats = _task_stats(_load_summary(summary_path))
        task_pass = stats["num_folds"] == 5
        for metric, baseline_value in baseline.items():
            value = stats[metric]["mean"]
            task_pass = task_pass and value is not None
            if task in PROTECTED_TASKS:
                task_pass = task_pass and value >= baseline_value
            if value is not None:
                macro_values[metric].append(value)
        tasks[task] = {"pass": task_pass, "metrics": stats, "baseline": baseline}

    macro = {
        metric: {
            "mean": mean(values) if values else None,
            "threshold": threshold,
            "pass": bool(values) and mean(values) > threshold and len(values) == len(V1_BASELINE),
        }
        for metric, threshold in MACRO_THRESHOLDS.items()
        for values in [macro_values[metric]]
    }
    comparison = {
        "pass": all(task.get("pass", False) for task in tasks.values())
        and all(item["pass"] for item in macro.values()),
        "tasks": tasks,
        "macro": macro,
        "benchmark_root": str(args.benchmark_root),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), comparison)


if __name__ == "__main__":
    main()
