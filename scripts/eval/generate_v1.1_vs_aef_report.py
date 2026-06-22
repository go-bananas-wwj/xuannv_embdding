#!/usr/bin/env python3
"""Generate a presentation-ready comparison figure: V1.1 embedding vs AEF 2025.

Reads:
  - V1.1 5-fold results: /data/xuannv_embedding/experiments/v1.1_long/v1.1_distill_long_stable_50ep_benchmark/*/summary_5fold.json
  - AEF baseline: /root/workspace/report/data/aef_benchmark_summary.json

Writes:
  - /root/workspace/report/v1.1_benchmark/v1.1_vs_aef_metrics.png
  - /root/workspace/report/v1.1_benchmark/v1.1_metrics_summary.json
"""
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
METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "f1_best": "F1-best",
    "f1_0.5": "F1@0.5",
    "miou": "mIoU",
}
TASKS = ["construction", "building_change", "farm_change", "rubbish", "construction_joint"]
TASK_LABELS = {
    "construction": "Construction",
    "building_change": "Building Change",
    "farm_change": "Farm Change",
    "rubbish": "Rubbish",
    "construction_joint": "Construction Joint",
}


def aggregate_folds(summary_path: Path) -> dict[str, Any]:
    with open(summary_path, encoding="utf-8") as f:
        folds = json.load(f)
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


def load_v11(benchmark_root: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for task in TASKS:
        summary_path = benchmark_root / task / "summary_5fold.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing V1.1 summary: {summary_path}")
        data[task] = aggregate_folds(summary_path)
    return data


def load_aef(aef_path: Path) -> dict[str, Any]:
    with open(aef_path, encoding="utf-8") as f:
        raw = json.load(f)
    # aef_benchmark_summary.json contains a list of entries, some with empty task keys.
    entries = raw if isinstance(raw, list) else raw.get("tasks", [])
    data: dict[str, Any] = {}
    for entry in entries:
        task = entry.get("task")
        version = entry.get("version")
        if not task or version != "AEF":
            continue
        data[task] = {
            "n_folds": 5,
            "auc_roc_mean": entry["auc_roc"]["value"],
            "auc_roc_std": entry["auc_roc"].get("std") or 0.0,
            "f1_best_mean": entry["f1_best"]["value"],
            "f1_best_std": entry["f1_best"].get("std") or 0.0,
            "f1_0.5_mean": entry["f1_at_0_5"]["value"],
            "f1_0.5_std": 0.0,
            "miou_mean": entry["miou"]["value"],
            "miou_std": 0.0,
        }
    missing = [t for t in TASKS if t not in data]
    if missing:
        raise ValueError(f"AEF baseline missing tasks: {missing}")
    return data


def format_delta(v11_val: float, aef_val: float) -> str:
    delta = aef_val - v11_val
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta*100:.2f}pp"


def write_report(
    v11: dict[str, Any],
    aef: dict[str, Any],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# V1.1 vs AEF 2025 下游 benchmark 对比报告")
    lines.append("")
    lines.append("## 说明")
    lines.append("- **V1.1**：本次 AEF 蒸馏训练得到的 embedding（双时相 202512+202605，64 维）。")
    lines.append("- **AEF 2025**：Google / GeoAI Foundation 官方年度 embedding（单时相 2025，64 维）。")
    lines.append("- 指标均为 5-fold 交叉验证的 mean±std。")
    lines.append("")
    lines.append("| 任务 | 对比项 | AUC-ROC | F1-best | F1@0.5 | mIoU |")
    lines.append("|------|--------|---------|---------|--------|------|")

    for task in TASKS:
        v = v11[task]
        a = aef[task]
        lines.append(
            f"| {TASK_LABELS[task]} | V1.1 | "
            f"{v['auc_roc_mean']:.4f}±{v['auc_roc_std']:.4f} | "
            f"{v['f1_best_mean']:.4f}±{v['f1_best_std']:.4f} | "
            f"{v['f1_0.5_mean']:.4f} | "
            f"{v['miou_mean']:.4f} |"
        )
        lines.append(
            f"| {TASK_LABELS[task]} | AEF | "
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
            f"| {TASK_LABELS[task]} | Δ(AEF−V1.1) | {auc_delta} | {f1b_delta} | {f1p5_delta} | {miou_delta} |"
        )
        lines.append("| | | | | | |")

    wins = {m: 0 for m in METRICS}
    for task in TASKS:
        for metric in METRICS:
            if v11[task][f"{metric}_mean"] > aef[task][f"{metric}_mean"]:
                wins[metric] += 1

    lines.append("")
    lines.append("## 结论")
    lines.append(
        f"V1.1 在 5 个任务中，AUC-ROC 胜出 {wins['auc_roc']}/5，"
        f"F1-best 胜出 {wins['f1_best']}/5，F1@0.5 胜出 {wins['f1_0.5']}/5，"
        f"mIoU 胜出 {wins['miou']}/5。"
    )
    lines.append("")
    lines.append("![V1.1 vs AEF 2025 指标对比](v1.1_vs_aef_metrics.png)")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(
    v11: dict[str, Any],
    aef: dict[str, Any],
    output_path: Path,
    dpi: int = 200,
) -> None:
    x = np.arange(len(TASKS))
    width = 0.35

    # Use a clean, presentation-friendly palette matching the report.
    color_v11 = "#3498db"  # accent blue
    color_aef = "#f58518"  # AEF orange

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    axes = axes.ravel()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        v_means = np.array([v11[t][f"{metric}_mean"] for t in TASKS])
        v_stds = np.array([v11[t][f"{metric}_std"] for t in TASKS])
        a_means = np.array([aef[t][f"{metric}_mean"] for t in TASKS])
        a_stds = np.array([aef[t][f"{metric}_std"] for t in TASKS])

        bars1 = ax.bar(
            x - width / 2,
            v_means,
            width,
            yerr=v_stds,
            label="V1.1 (Ours)",
            color=color_v11,
            edgecolor="white",
            linewidth=0.8,
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        bars2 = ax.bar(
            x + width / 2,
            a_means,
            width,
            yerr=a_stds,
            label="AEF 2025",
            color=color_aef,
            edgecolor="white",
            linewidth=0.8,
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )

        ax.set_title(METRIC_LABELS[metric], fontsize=13, fontweight="bold", color="#2c3e50")
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Dynamic ylim with a little headroom; keep AUC close to [0,1].
        all_vals = np.concatenate([v_means - v_stds, v_means + v_stds, a_means - a_stds, a_means + a_stds])
        ymin, ymax = max(0.0, all_vals.min() - 0.02), min(1.0, all_vals.max() + 0.05)
        if metric == "auc_roc":
            ymin, ymax = 0.55, 1.02
        ax.set_ylim(ymin, ymax)

        # Annotate bars with values.
        for bar, val in zip(bars1, v_means):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#2c3e50",
            )
        for bar, val in zip(bars2, a_means):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#2c3e50",
            )

    fig.suptitle(
        "V1.1 vs AEF 2025：下游任务 5-fold 指标对比",
        fontsize=16,
        fontweight="bold",
        color="#2c3e50",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate V1.1 vs AEF comparison figure")
    p.add_argument(
        "--v11-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v1.1_long/v1.1_distill_long_stable_50ep_benchmark"),
        help="V1.1 benchmark output root",
    )
    p.add_argument(
        "--aef-summary",
        type=Path,
        default=Path("/root/workspace/report/data/aef_benchmark_summary.json"),
        help="AEF baseline summary JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/workspace/report/v1.1_benchmark"),
        help="Output directory for figure and JSON",
    )
    p.add_argument("--dpi", type=int, default=200, help="Output PNG DPI")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    v11 = load_v11(args.v11_root)
    aef = load_aef(args.aef_summary)

    summary: dict[str, Any] = {
        "v1.1": v11,
        "aef": aef,
    }
    summary_path = args.output_dir / "v1.1_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_path = args.output_dir / "v1.1_vs_aef_metrics.png"
    plot_comparison(v11, aef, plot_path, dpi=args.dpi)

    report_path = args.output_dir / "V1.1_vs_AEF_REPORT.md"
    write_report(v11, aef, report_path)

    print(f"Summary: {summary_path}")
    print(f"Figure: {plot_path}")
    print(f"Report: {report_path}")

    # Update the HTML report index to include this section.
    index_updater = Path("/root/workspace/xuannv/.worktrees/feat-multitask-downstream/scripts/eval/update_report_index.py")
    if index_updater.exists():
        import subprocess
        subprocess.run(
            [
                "python",
                str(index_updater),
                "--report", str(report_path),
                "--figure", str(plot_path),
            ],
            check=True,
        )
        print("Updated /root/workspace/report/index.html")



if __name__ == "__main__":
    main()
