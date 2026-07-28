#!/usr/bin/env python3
"""Render the main P10C sparse-readout comparison figure from per-cell records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = ("building", "road", "water")
FEATURE_ORDER = ("xuannv", "aef", "traditional")
FEATURE_LABELS = {
    "xuannv": "XuannvEarth\nP10C, Apr. 2026",
    "aef": "AEF\nannual 2025",
    "traditional": "Traditional\nApr. 2026 features",
}
FEATURE_COLORS = {"xuannv": "#b2182b", "aef": "#2166ac", "traditional": "#5a5a5a"}
METRICS = (("f1", "F1"), ("auc", "ROC-AUC"), ("ap", "Average precision"))


def load_rows(*roots: Path) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        for result_path in sorted(root.glob("**/results.json")):
            payload = json.loads(result_path.read_text())
            rows.extend(payload["rows"])
    expected = len(TASK_ORDER) * 5 * 3 * len(FEATURE_ORDER)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} result rows, found {len(rows)}")
    return rows


def aggregate(rows: list[dict]) -> dict[tuple[str, str, str], tuple[float, float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric, _ in METRICS:
            values[(row["task"], row["feature"], metric)].append(row["metrics"][metric])
    summary: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, items in values.items():
        if len(items) != 15:
            raise ValueError(f"Expected 15 cells for {key}, found {len(items)}")
        summary[key] = (float(np.mean(items)), float(np.std(items, ddof=1)))
    return summary


def render(summary: dict[tuple[str, str, str], tuple[float, float]], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 4.5), sharey=False)
    x = np.arange(len(TASK_ORDER))
    width = 0.23
    offsets = (-width, 0.0, width)

    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for offset, feature in zip(offsets, FEATURE_ORDER, strict=True):
            means = [summary[(task, feature, metric)][0] for task in TASK_ORDER]
            stds = [summary[(task, feature, metric)][1] for task in TASK_ORDER]
            axis.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=2.5,
                color=FEATURE_COLORS[feature],
                edgecolor="white",
                linewidth=0.6,
                label=FEATURE_LABELS[feature],
            )
        axis.set_title(label, fontweight="bold")
        axis.set_xticks(x, ("Building", "Road", "Water"))
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.10))
    figure.text(
        0.5,
        0.025,
        "Five positive-pixel-ratio-stratified patch folds × three support-polygon selections; "
        "bars are mean ± sample SD over 15 cells. Same PU reader and validation-only threshold rule.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.25, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-root", type=Path, required=True)
    parser.add_argument("--roadwater-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(aggregate(load_rows(args.building_root, args.roadwater_root)), args.output)


if __name__ == "__main__":
    main()
