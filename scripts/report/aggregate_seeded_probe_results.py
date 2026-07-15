#!/usr/bin/env python3
"""Aggregate 5-fold downstream probe results across multiple random seeds.

Each task is expected at ``<input-root>/<task>/seed<seed>/summary.json``.
The primary paper statistic is the mean and sample standard deviation of the
per-seed five-fold means.  Pooled 15-run statistics are retained in JSON for
diagnostic use, but are not substituted for seed-level uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("f1_at_threshold", "ap", "auc_roc", "miou", "precision", "recall", "f1_best")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return rows


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def discover_tasks(input_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in input_root.iterdir()
        if path.is_dir() and any(path.glob("seed*/summary.json"))
    )


def summarize_task(task_dir: Path, seeds: list[int]) -> dict[str, Any]:
    seed_rows: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        summary = task_dir / f"seed{seed}" / "summary.json"
        if summary.exists():
            seed_rows[str(seed)] = load_rows(summary)

    if not seed_rows:
        raise FileNotFoundError(f"No seed summaries found in {task_dir}")

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        per_seed = {
            seed: mean(float(row[metric]) for row in rows if metric in row)
            for seed, rows in seed_rows.items()
        }
        pooled = [float(row[metric]) for rows in seed_rows.values() for row in rows if metric in row]
        metrics[metric] = {
            "per_seed_mean": per_seed,
            "seed_level": stats(list(per_seed.values())),
            "pooled_fold_level": stats(pooled),
        }

    return {
        "seeds_present": [int(seed) for seed in seed_rows],
        "folds_per_seed": {seed: len(rows) for seed, rows in seed_rows.items()},
        "metrics": metrics,
    }


def fmt(metric: dict[str, Any]) -> str:
    values = metric["seed_level"]
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


def write_csv(output: Path, report: dict[str, Any]) -> Path:
    path = output.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "metric", "mean", "std", "n_seeds", "pooled_mean", "pooled_std", "n_folds"],
        )
        writer.writeheader()
        for task, task_report in report["tasks"].items():
            for metric, values in task_report["metrics"].items():
                seed_level = values["seed_level"]
                pooled = values["pooled_fold_level"]
                writer.writerow(
                    {
                        "task": task,
                        "metric": metric,
                        "mean": seed_level["mean"],
                        "std": seed_level["std"],
                        "n_seeds": seed_level["n"],
                        "pooled_mean": pooled["mean"],
                        "pooled_std": pooled["std"],
                        "n_folds": pooled["n"],
                    }
                )
    return path


def write_markdown(output: Path, report: dict[str, Any]) -> Path:
    path = output.with_suffix(".md")
    lines = [
        "# Seeded downstream probe summary",
        "",
        "Primary values are mean +/- sample standard deviation across the three per-seed 5-fold means.",
        "The CSV/JSON also retain pooled fold-level values for diagnostics.",
        "",
        "| Task | F1 @ validation threshold | AP | AUC-ROC | mIoU | Precision | Recall | Oracle F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, task_report in report["tasks"].items():
        metrics = task_report["metrics"]
        lines.append(
            "| {task} | {f1} | {ap} | {auc} | {miou} | {precision} | {recall} | {oracle} |".format(
                task=task,
                f1=fmt(metrics["f1_at_threshold"]),
                ap=fmt(metrics["ap"]),
                auc=fmt(metrics["auc_roc"]),
                miou=fmt(metrics["miou"]),
                precision=fmt(metrics["precision"]),
                recall=fmt(metrics["recall"]),
                oracle=fmt(metrics["f1_best"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    tasks = args.tasks or discover_tasks(args.input_root)
    report = {
        "input_root": str(args.input_root),
        "requested_seeds": list(args.seeds),
        "aggregation": "mean/std across per-seed five-fold means",
        "tasks": {
            task: summarize_task(args.input_root / task, list(args.seeds)) for task in tasks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = write_csv(args.output, report)
    markdown_path = write_markdown(args.output, report)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
