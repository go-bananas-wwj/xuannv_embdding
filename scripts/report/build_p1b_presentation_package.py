#!/usr/bin/env python3
"""Build a presentation package for the P1B downstream benchmark."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_BENCHMARK_ROOT = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/"
    "v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751"
)
DEFAULT_EMBEDDING_ROOT = Path(
    "/data/xuannv_embedding/embeddings/v2_202512_202605/"
    "20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_"
    "v2_p1_sparse_sampler_hardneg_20260627_090500"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/presentation"
)

TASK_NAMES = {
    "construction": "施工地识别",
    "building_change": "建筑变化检测",
    "farm_change": "耕地变化检测",
    "rubbish": "疑似垃圾/堆场检测",
    "construction_joint": "施工地联合检测",
}
METRIC_NAMES = {
    "auc_roc": "AUC",
    "f1_best": "F1_best",
    "f1_0.5": "F1@0.5",
    "miou": "mIoU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--samples-per-task", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def metric_mean(task_data: dict[str, Any], side: str, metric: str) -> float:
    return float(task_data[side][metric]["mean"])


def write_metrics_csv(comparison: dict[str, Any], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "task_cn",
                "metric",
                "p1b_mean",
                "aef_mean",
                "delta",
                "p1b_std",
                "aef_std",
            ]
        )
        for task, task_data in comparison["tasks"].items():
            for metric in METRIC_NAMES:
                p1b = metric_mean(task_data, "candidate", metric)
                aef = metric_mean(task_data, "aef", metric)
                writer.writerow(
                    [
                        task,
                        TASK_NAMES.get(task, task),
                        metric,
                        f"{p1b:.6f}",
                        f"{aef:.6f}",
                        f"{p1b - aef:.6f}",
                        f"{float(task_data['candidate'][metric]['std']):.6f}",
                        f"{float(task_data['aef'][metric]['std']):.6f}",
                    ]
                )


def plot_macro(comparison: dict[str, Any], output: Path) -> None:
    metrics = list(METRIC_NAMES)
    p1b = [float(comparison["macro"][metric]["candidate"]) for metric in metrics]
    aef = [float(comparison["macro"][metric]["aef"]) for metric in metrics]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, p1b, width, label="P1B", color="#2563eb")
    ax.bar(x + width / 2, aef, width, label="AEF", color="#94a3b8")
    ax.set_xticks(x, [METRIC_NAMES[m] for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("P1B vs AEF - Macro Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for xpos, value in zip(x - width / 2, p1b, strict=True):
        ax.text(xpos, value + 0.015, fmt(value), ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, aef, strict=True):
        ax.text(xpos, value + 0.015, fmt(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_task_metric(comparison: dict[str, Any], metric: str, output: Path) -> None:
    tasks = list(comparison["tasks"])
    labels = tasks
    p1b = [metric_mean(comparison["tasks"][task], "candidate", metric) for task in tasks]
    aef = [metric_mean(comparison["tasks"][task], "aef", metric) for task in tasks]
    x = np.arange(len(tasks))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x - width / 2, p1b, width, label="P1B", color="#0f766e")
    ax.bar(x + width / 2, aef, width, label="AEF", color="#cbd5e1")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, max(max(p1b), max(aef), 0.1) * 1.25)
    ax.set_ylabel(METRIC_NAMES[metric])
    ax.set_title(f"P1B vs AEF - {METRIC_NAMES[metric]} by Task")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def pick_visual_records(metadata: list[dict[str, Any]], samples_per_task: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in metadata:
        if any(record.get("missing", {}).values()):
            continue
        grouped.setdefault(record["task"], []).append(record)

    selected: list[dict[str, Any]] = []
    for task in TASK_NAMES:
        records = sorted(
            grouped.get(task, []),
            key=lambda item: int(item.get("gt_positive_pixels", 0)),
            reverse=True,
        )
        selected.extend(records[:samples_per_task])
    return selected


def copy_visuals(records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for record in records:
        src = Path(record["figure"])
        dst = output_dir / record["task"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        new_record = dict(record)
        new_record["package_figure"] = str(dst)
        copied.append(new_record)
    return copied


def write_readme(
    output_root: Path,
    comparison: dict[str, Any],
    threshold: dict[str, Any],
    visuals: list[dict[str, Any]],
    benchmark_root: Path,
    embedding_root: Path,
) -> None:
    macro = comparison["macro"]
    lines = [
        "# P1B 下游任务汇报包",
        "",
        "## 1. 汇报结论",
        "",
        (
            "P1B embedding 在完整 5 折下游测评中，macro 指标整体超过 AEF："
            f"AUC `{fmt(float(macro['auc_roc']['candidate']))}` vs "
            f"`{fmt(float(macro['auc_roc']['aef']))}`，"
            f"F1_best `{fmt(float(macro['f1_best']['candidate']))}` vs "
            f"`{fmt(float(macro['f1_best']['aef']))}`，"
            f"mIoU `{fmt(float(macro['miou']['candidate']))}` vs "
            f"`{fmt(float(macro['miou']['aef']))}`。"
        ),
        "",
        "最适合作为汇报亮点的任务是 `rubbish`、`building_change` 和 `construction`。"
        "`construction_joint` 的 AUC 超过 AEF，但 F1 和 mIoU 仍需后续校准和 head/loss 优化。",
        "",
        "## 2. 产物目录",
        "",
        f"- P1B benchmark：`{benchmark_root}`",
        f"- P1B embedding：`{embedding_root}`",
        "- 指标 CSV：`tables/p1b_vs_aef_metrics.csv`",
        "- macro 对比图：`charts/macro_metrics_vs_aef.png`",
        "- 分任务 F1_best 对比图：`charts/task_f1_best_vs_aef.png`",
        "- 分任务 mIoU 对比图：`charts/task_miou_vs_aef.png`",
        "- 代表性可视化：`visuals/`",
        "",
        "## 3. 分任务指标",
        "",
        "| 任务 | AUC | AEF AUC | F1_best | AEF F1_best | F1@0.5 | AEF F1@0.5 | mIoU | AEF mIoU | 说明 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, task_data in comparison["tasks"].items():
        p1b_auc = metric_mean(task_data, "candidate", "auc_roc")
        aef_auc = metric_mean(task_data, "aef", "auc_roc")
        p1b_f1 = metric_mean(task_data, "candidate", "f1_best")
        aef_f1 = metric_mean(task_data, "aef", "f1_best")
        p1b_f105 = metric_mean(task_data, "candidate", "f1_0.5")
        aef_f105 = metric_mean(task_data, "aef", "f1_0.5")
        p1b_miou = metric_mean(task_data, "candidate", "miou")
        aef_miou = metric_mean(task_data, "aef", "miou")
        note = "优于 AEF" if p1b_f1 >= aef_f1 and p1b_miou >= aef_miou else "部分指标仍需优化"
        lines.append(
            f"| {TASK_NAMES.get(task, task)} | {fmt(p1b_auc)} | {fmt(aef_auc)} | "
            f"{fmt(p1b_f1)} | {fmt(aef_f1)} | {fmt(p1b_f105)} | {fmt(aef_f105)} | "
            f"{fmt(p1b_miou)} | {fmt(aef_miou)} | {note} |"
        )

    lines.extend(
        [
            "",
            "## 4. 阈值校准提示",
            "",
            "默认 `0.5` 阈值不是当前模型的最佳工作点。验证集阈值可以显著提升 F1：",
            "",
            "| 任务 | F1@0.5 | F1@val_thr | 提升 | 平均阈值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, task_data in threshold["tasks"].items():
        stats = task_data["stats"]
        lines.append(
            f"| {TASK_NAMES.get(task, task)} | "
            f"{fmt(float(stats['f1_0.5']['mean']))} | "
            f"{fmt(float(stats['f1_at_threshold']['mean']))} | "
            f"{fmt(float(task_data['calibration_gain']['mean']))} | "
            f"{fmt(float(stats['val_threshold']['mean']))} |"
        )

    lines.extend(
        [
            "",
            "## 5. 可视化说明",
            "",
            "每张条带图从左到右依次是：变化前高分影像、变化后高分影像、变化前 embedding PCA、变化后 embedding PCA、embedding 差分 PCA、模型预测概率、真实 GT 标签。",
            "",
        ]
    )
    for record in visuals:
        fig = Path(record["package_figure"])
        rel = fig.relative_to(output_root)
        lines.extend(
            [
                f"### {TASK_NAMES.get(record['task'], record['task'])} / {record['patch_id']}",
                "",
                f"- GT 正样本像素：`{record['gt_positive_pixels']}`",
                f"- 原始 GT：`{record['gt_mask']}`",
                "",
                f"![{fig.name}]({rel.as_posix()})",
                "",
            ]
        )

    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"p1b_downstream_showcase_{dt.datetime.utcnow():%Y%m%d_%H%M%S}"
    output_root = args.output_base / run_name
    tables_dir = output_root / "tables"
    charts_dir = output_root / "charts"
    visuals_dir = output_root / "visuals"
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    comparison = read_json(args.benchmark_root / "comparison_vs_aef.json")
    threshold = read_json(args.benchmark_root / "threshold_calibration.json")
    metadata = read_json(args.benchmark_root / "visualizations" / "metadata.json")

    write_metrics_csv(comparison, tables_dir / "p1b_vs_aef_metrics.csv")
    plot_macro(comparison, charts_dir / "macro_metrics_vs_aef.png")
    plot_task_metric(comparison, "f1_best", charts_dir / "task_f1_best_vs_aef.png")
    plot_task_metric(comparison, "miou", charts_dir / "task_miou_vs_aef.png")

    selected = pick_visual_records(metadata, args.samples_per_task)
    copied = copy_visuals(selected, visuals_dir)
    (output_root / "selected_visualizations.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(output_root, comparison, threshold, copied, args.benchmark_root, args.embedding_root)
    print(f"presentation package: {output_root}")


if __name__ == "__main__":
    main()
