#!/usr/bin/env python3
"""Compare a downstream benchmark directory against the AEF benchmark."""

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
    "water",
    "construction_joint",
)
METRICS = ("auc_roc", "f1_best", "f1_0.5", "miou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--aef-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/aef_benchmark"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_folds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        out[metric] = {
            "mean": mean(values) if values else None,
            "std": pstdev(values) if values else None,
        }
    return out


def task_summary(root: Path, task: str) -> dict[str, Any] | None:
    path = root / task / "summary.json"
    if not path.exists():
        path = root / task / "summary_5fold.json"
    if not path.exists():
        return None
    return summarize(load_rows(path))


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    name = report["name"]
    lines = [
        f"# {name} vs AEF",
        "",
        f"Candidate root: `{report['benchmark_root']}`",
        f"AEF root: `{report['aef_root']}`",
        "",
        "| Task | Metric | Candidate | AEF | Delta | Winner |",
        "|---|---|---:|---:|---:|---|",
    ]
    for task, task_report in report["tasks"].items():
        if "error" in task_report:
            lines.append(f"| {task} | error | NA | NA | NA | {task_report['error']} |")
            continue
        for metric in METRICS:
            candidate = task_report["candidate"][metric]["mean"]
            aef = task_report["aef"][metric]["mean"]
            diff = delta(candidate, aef)
            if diff is None:
                winner = "NA"
            elif diff > 0:
                winner = name
            elif diff < 0:
                winner = "AEF"
            else:
                winner = "tie"
            lines.append(
                "| {task} | {metric} | {candidate} | {aef} | {diff} | {winner} |".format(
                    task=task,
                    metric=metric,
                    candidate=fmt(candidate),
                    aef=fmt(aef),
                    diff=fmt(diff),
                    winner=winner,
                )
            )
    lines.extend(
        [
            "",
            "## Macro",
            "",
            "| Metric | Candidate | AEF | Delta | Winner |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric, values in report["macro"].items():
        diff = delta(values["candidate"], values["aef"])
        if diff is None:
            winner = "NA"
        elif diff > 0:
            winner = name
        elif diff < 0:
            winner = "AEF"
        else:
            winner = "tie"
        lines.append(
            "| {metric} | {candidate} | {aef} | {diff} | {winner} |".format(
                metric=metric,
                candidate=fmt(values["candidate"]),
                aef=fmt(values["aef"]),
                diff=fmt(diff),
                winner=winner,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks: dict[str, Any] = {}
    macro_values: dict[str, dict[str, list[float]]] = {
        metric: {"candidate": [], "aef": []} for metric in METRICS
    }

    for task in args.tasks:
        candidate = task_summary(args.benchmark_root, task)
        aef = task_summary(args.aef_root, task)
        if candidate is None or aef is None:
            tasks[task] = {
                "error": f"missing candidate={candidate is None} aef={aef is None}"
            }
            continue
        tasks[task] = {"candidate": candidate, "aef": aef}
        for metric in METRICS:
            candidate_mean = candidate[metric]["mean"]
            aef_mean = aef[metric]["mean"]
            if candidate_mean is not None and aef_mean is not None:
                macro_values[metric]["candidate"].append(candidate_mean)
                macro_values[metric]["aef"].append(aef_mean)

    macro = {
        metric: {
            "candidate": mean(values["candidate"]) if values["candidate"] else None,
            "aef": mean(values["aef"]) if values["aef"] else None,
        }
        for metric, values in macro_values.items()
    }
    report = {
        "name": args.name,
        "benchmark_root": str(args.benchmark_root),
        "aef_root": str(args.aef_root),
        "tasks": tasks,
        "macro": macro,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)


if __name__ == "__main__":
    main()
