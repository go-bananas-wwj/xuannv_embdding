#!/usr/bin/env python3
"""Aggregate AEF benchmark results and generate comparison report against V1.0."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["auc_roc", "f1_best", "f1_0.5", "miou"]


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_task(summary_path: Path) -> dict[str, Any]:
    with open(summary_path, encoding="utf-8") as f:
        folds = json.load(f)

    if not folds:
        raise ValueError(f"No fold results in {summary_path}")

    out: dict[str, Any] = {"n_folds": len(folds)}
    for metric in METRICS:
        values = [fold[metric] for fold in folds if metric in fold]
        if len(values) != len(folds):
            raise ValueError(f"Metric {metric} missing in some folds of {summary_path}")
        m, s = mean_std(values)
        out[f"{metric}_mean"] = m
        out[f"{metric}_std"] = s
    return out


def format_delta(v1: float, v2: float) -> str:
    delta = v2 - v1
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}pp"


def generate_report(
    aef_root: Path,
    v10_path: Path,
    output_path: Path,
    plot_path: Path,
) -> None:
    with open(v10_path, encoding="utf-8") as f:
        v10 = json.load(f)

    tasks = ["construction", "building_change", "farm_change", "rubbish", "construction_joint"]
    aef: dict[str, Any] = {}
    for task in tasks:
        summary_path = aef_root / task / "summary_5fold.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing AEF summary: {summary_path}")
        aef[task] = aggregate_task(summary_path)

    # Write JSON summary
    with open(aef_root / "aef_summary.json", "w", encoding="utf-8") as f:
        json.dump(aef, f, ensure_ascii=False, indent=2)

    # Markdown report
    lines: list[str] = []
    lines.append("# AEF 2025 官方 embedding 下游 benchmark 报告")
    lines.append("")
    lines.append("## 说明")
    lines.append("- **V1.0**: 自研模型产出的 embedding（双时相 202512+202605，64 维）。")
    lines.append("- **AEF 2025**: Google / GeoAI Foundation 官方年度 embedding（单时相 2025，64 维）。")
    lines.append("- 指标均为 5-fold 交叉验证的 mean±std。")
    lines.append("")
    lines.append("| 任务 | 对比项 | AUC-ROC | F1-best | F1@0.5 | mIoU |")
    lines.append("|------|--------|---------|---------|--------|------|")

    for task in tasks:
        v = v10[task]
        a = aef[task]
        lines.append(
            f"| {task} | V1.0 | "
            f"{v['auc_roc_mean']:.4f}±{v['auc_roc_std']:.4f} | "
            f"{v['f1_best_mean']:.4f}±{v['f1_best_std']:.4f} | "
            f"{v['f1_0.5_mean']:.4f} | "
            f"{v['miou_mean']:.4f} |"
        )
        lines.append(
            f"| {task} | AEF | "
            f"{a['auc_roc_mean']:.4f}±{a['auc_roc_std']:.4f} | "
            f"{a['f1_best_mean']:.4f}±{a['f1_best_std']:.4f} | "
            f"{a['f1_0.5_mean']:.4f} | "
            f"{a['miou_mean']:.4f} |"
        )
        auc_delta = format_delta(v["auc_roc_mean"], a["auc_roc_mean"])
        f1b_delta = format_delta(v["f1_best_mean"], a["f1_best_mean"])
        f1p5_delta = format_delta(v["f1_0.5_mean"], a["f1_0.5_mean"])
        miou_delta = format_delta(v["miou_mean"], a["miou_mean"])
        lines.append(
            f"| {task} | Δ(AEF−V1.0) | {auc_delta} | {f1b_delta} | {f1p5_delta} | {miou_delta} |"
        )
        lines.append("| | | | | | |")

    lines.append("")
    lines.append("## 结论")
    wins = {m: 0 for m in METRICS}
    for task in tasks:
        for metric in METRICS:
            if aef[task][f"{metric}_mean"] > v10[task][f"{metric}_mean"]:
                wins[metric] += 1
    lines.append(
        f"AEF 在 5 个任务中，AUC-ROC 胜出 {wins['auc_roc']}/5，"
        f"F1-best 胜出 {wins['f1_best']}/5，F1@0.5 胜出 {wins['f1_0.5']}/5，"
        f"mIoU 胜出 {wins['miou']}/5。"
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    # Bar plot
    x = np.arange(len(tasks))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        v_means = [v10[t][f"{metric}_mean"] for t in tasks]
        a_means = [aef[t][f"{metric}_mean"] for t in tasks]
        ax.bar(x - width / 2, v_means, width, label="V1.0", color="#4C78A8")
        ax.bar(x + width / 2, a_means, width, label="AEF 2025", color="#F58518")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate AEF vs V1.0 benchmark report")
    p.add_argument(
        "--aef-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/aef_benchmark"),
        help="Directory containing AEF task result folders",
    )
    p.add_argument(
        "--v10-path",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json"),
        help="V1.0 summary JSON",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md"),
        help="Output markdown report path",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/aef_benchmark/aef_vs_v1.0.png"),
        help="Output comparison plot path",
    )
    args = p.parse_args()

    generate_report(args.aef_root, args.v10_path, args.output, args.plot)
    print(f"Report: {args.output}")
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
