#!/usr/bin/env python3
"""Bar plot of per-task AUC-ROC and F1_best from the V1.0 summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--out-path", type=Path, default=None)
    args = p.parse_args()

    summary_path = args.summary or Path(
        "/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json"
    )
    out_path = args.out_path or Path(
        "/data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png"
    )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    tasks = list(summary.keys())
    auc_roc = [summary[t]["auc_roc_mean"] for t in tasks]
    f1_best = [summary[t]["f1_best_mean"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, auc_roc, width, label="AUC-ROC", color="steelblue")
    bars2 = ax.bar(x + width / 2, f1_best, width, label="F1_best", color="coral")

    ax.axhline(0.8, color="green", linestyle="--", linewidth=1.5, label="Target 0.8")
    ax.axhline(0.6, color="orange", linestyle="--", linewidth=1.5, label="Target 0.6")

    ax.set_ylabel("Score")
    ax.set_title("V1.0 Downstream Task Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved task metrics comparison plot to {out_path}")


if __name__ == "__main__":
    main()
