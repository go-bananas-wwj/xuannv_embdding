#!/usr/bin/env python3
"""汇总 V1.2 national 模型与 AEF 2025 的 4-task 5-fold benchmark 结果并生成对比报告。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["auc_roc", "f1_best", "f1_0.5", "miou"]
TASKS = ["construction", "building_change", "farm_change", "rubbish"]

# AEF 2025 的 construction 任务为 haidian+harbin 联合目录，
# 其他任务目录名与 TASKS 一致。
AEF_TASK_DIR = {
    "construction": "construction_joint",
    "building_change": "building_change",
    "farm_change": "farm_change",
    "rubbish": "rubbish",
}


def aggregate_task(summary_path: Path) -> dict[str, Any]:
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    folds = data.get("folds", data) if isinstance(data, dict) else data
    if not folds:
        raise ValueError(f"No fold results in {summary_path}")
    out: dict[str, Any] = {"n_folds": len(folds)}
    for metric in METRICS:
        values = [fold[metric] for fold in folds if metric in fold]
        if len(values) != len(folds):
            raise ValueError(f"Metric {metric} missing in some folds of {summary_path}")
        arr = np.array(values, dtype=np.float64)
        out[f"{metric}_mean"] = float(arr.mean())
        out[f"{metric}_std"] = float(arr.std(ddof=0))
    return out


def format_delta(v1: float, v2: float) -> str:
    delta = v2 - v1
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}pp"


def generate_report(
    aef_root: Path,
    v12_root: Path,
    output_path: Path,
    plot_path: Path,
) -> None:
    aef: dict[str, Any] = {}
    v12: dict[str, Any] = {}
    for task in TASKS:
        aef[task] = aggregate_task(aef_root / AEF_TASK_DIR[task] / "summary_5fold.json")
        v12[task] = aggregate_task(v12_root / task / "summary_5fold.json")

    lines: list[str] = []
    lines.append("# V1.2 National Cross-Modal 模型下游 benchmark 对比报告")
    lines.append("")
    lines.append("## 说明")
    lines.append("- **V1.2**: 本分支训练的 national 低分辨率跨模态 embedding 模型（s2 10m + s2_20m，2025-12 ~ 2026-05）。")
    lines.append("- **AEF 2025**: Google / GeoAI Foundation 官方年度 embedding（单时相 2025，64 维）。")
    lines.append("- 指标均为 haidian + harbin 联合 5-fold 交叉验证的 mean±std。")
    lines.append("")
    lines.append("| 任务 | 对比项 | AUC-ROC | F1-best | F1@0.5 | mIoU |")
    lines.append("|------|--------|---------|---------|--------|------|")

    for task in TASKS:
        v = v12[task]
        a = aef[task]
        lines.append(
            f"| {task} | V1.2 | "
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
        auc_delta = format_delta(a["auc_roc_mean"], v["auc_roc_mean"])
        f1b_delta = format_delta(a["f1_best_mean"], v["f1_best_mean"])
        f1p5_delta = format_delta(a["f1_0.5_mean"], v["f1_0.5_mean"])
        miou_delta = format_delta(a["miou_mean"], v["miou_mean"])
        lines.append(
            f"| {task} | Δ(V1.2−AEF) | {auc_delta} | {f1b_delta} | {f1p5_delta} | {miou_delta} |"
        )
        lines.append("| | | | | | |")

    lines.append("")
    lines.append("## 结论")
    wins = {m: 0 for m in METRICS}
    for task in TASKS:
        for metric in METRICS:
            if v12[task][f"{metric}_mean"] > aef[task][f"{metric}_mean"]:
                wins[metric] += 1
    lines.append(
        f"V1.2 在 4 个任务中，AUC-ROC 胜出 {wins['auc_roc']}/4，"
        f"F1-best 胜出 {wins['f1_best']}/4，F1@0.5 胜出 {wins['f1_0.5']}/4，"
        f"mIoU 胜出 {wins['miou']}/4。"
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    # Bar plot
    x = np.arange(len(TASKS))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        v_means = [v12[t][f"{metric}_mean"] for t in TASKS]
        a_means = [aef[t][f"{metric}_mean"] for t in TASKS]
        ax.bar(x - width / 2, v_means, width, label="V1.2", color="#4C78A8")
        ax.bar(x + width / 2, a_means, width, label="AEF 2025", color="#F58518")
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Report: {output_path}")
    print(f"Plot:   {plot_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate V1.2 vs AEF 2025 benchmark report")
    p.add_argument(
        "--aef-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/aef_benchmark"),
    )
    p.add_argument(
        "--v12-root",
        type=Path,
        default=Path("/data2/xuannv_embedding/national/downstream_benchmark"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/data2/xuannv_embedding/national/V1.2_BENCHMARK_REPORT.md"),
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=Path("/data2/xuannv_embedding/national/v1.2_vs_aef.png"),
    )
    args = p.parse_args()
    generate_report(args.aef_root, args.v12_root, args.output, args.plot)


if __name__ == "__main__":
    main()
