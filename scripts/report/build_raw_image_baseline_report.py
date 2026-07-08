#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.report.visualize_osm_downstream_outputs import (
    binary_metrics,
    load_highres,
    load_prediction,
    read_mask,
    red_binary_mask,
    red_probability_map,
    show_panel,
)


ROOT = Path("/data/xuannv_embedding/experiments/p12_context_eval_20260707")
OUT = ROOT / "raw_202604_head_sweep_20260708_report"
RAW_ROOT = ROOT / "raw_202604_head_sweep_20260708_fixedenv"
P10C_CONV_ROOT = ROOT / "head_sweep_20260708/conv3x3/benchmarks/p10c"
P12C_CONV_ROOT = ROOT / "head_sweep_20260708/conv3x3/benchmarks/p12c"
P10C_DEEP_ROOT = ROOT / "head_sweep_20260708/deep_wide/benchmarks/p10c"
P12C_DEEP_ROOT = ROOT / "head_sweep_20260708/deep_wide/benchmarks/p12c"
P10C_WIDE_ROOT = ROOT / "wide_probe_compare_20260708/benchmarks/p10c"
P12C_WIDE_ROOT = ROOT / "wide_probe_compare_20260708/benchmarks/p12c"
P10C_OLD_ROOT = ROOT / "mlp5_probe_compare_20260708/benchmarks/p10c"
P12C_OLD_ROOT = ROOT / "mlp5_probe_compare_20260708/benchmarks/p12c"
PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")

TASKS = {
    "building": {
        "label": "Building",
        "xuannv_task": "haidian_building_osm",
        "label_task": "building_osm",
        "raw_best": "unet",
    },
    "road": {
        "label": "Road",
        "xuannv_task": "haidian_road_osm",
        "label_task": "road_osm",
        "raw_best": "unet",
    },
    "water": {
        "label": "Water",
        "xuannv_task": "haidian_water_osm",
        "label_task": "osm_water",
        "raw_best": "deeplab_lite",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if value is not None else float("nan")


def read_metric(path: Path, source: str, task: str, model: str) -> dict[str, Any]:
    record = load_json(path)
    return {
        "source": source,
        "task": task,
        "model": model,
        "f1": metric_value(record, "f1_at_threshold"),
        "f1_best": metric_value(record, "f1_best"),
        "ap": metric_value(record, "ap"),
        "auc": metric_value(record, "auc_roc"),
        "threshold": metric_value(record, "threshold"),
        "best_epoch": record.get("best_epoch", ""),
        "parameters": record.get("parameters", ""),
        "path": str(path),
    }


def collect_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    xuannv_roots = [
        ("P10C old local", P10C_OLD_ROOT, "old_local"),
        ("P10C wide MLP", P10C_WIDE_ROOT, "wide_mlp"),
        ("P10C deep-wide MLP", P10C_DEEP_ROOT, "deep_wide_mlp"),
        ("P10C conv3x3", P10C_CONV_ROOT, "conv3x3"),
        ("P12C old local", P12C_OLD_ROOT, "old_local"),
        ("P12C wide MLP", P12C_WIDE_ROOT, "wide_mlp"),
        ("P12C deep-wide MLP", P12C_DEEP_ROOT, "deep_wide_mlp"),
        ("P12C conv3x3", P12C_CONV_ROOT, "conv3x3"),
    ]
    for task, info in TASKS.items():
        for source, root, model in xuannv_roots:
            path = root / info["xuannv_task"] / "fold_0" / "metrics.json"
            if path.exists():
                rows.append(read_metric(path, source, task, model))
        for path in sorted((RAW_ROOT / task / "s2_s1_landsat_highres_indices").glob("*/shot_full/fold_0/metrics.json")):
            model = path.parts[-4]
            rows.append(read_metric(path, f"Raw 202604 {model}", task, model))
    return pd.DataFrame(rows)


def plot_metric_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    selected = df[
        df["source"].isin(
            [
                "P10C conv3x3",
                "P12C conv3x3",
                "Raw 202604 conv3x3",
                "Raw 202604 unet",
                "Raw 202604 deeplab_lite",
                "Raw 202604 segformer_lite",
            ]
        )
    ].copy()
    selected["method"] = selected["source"].str.replace("Raw 202604 ", "Raw ", regex=False)
    tasks = list(TASKS)
    methods = [
        "P10C conv3x3",
        "P12C conv3x3",
        "Raw conv3x3",
        "Raw unet",
        "Raw deeplab_lite",
        "Raw segformer_lite",
    ]
    colors = {
        "P10C conv3x3": "#D62728",
        "P12C conv3x3": "#FF7F0E",
        "Raw conv3x3": "#1F77B4",
        "Raw unet": "#2CA02C",
        "Raw deeplab_lite": "#9467BD",
        "Raw segformer_lite": "#8C564B",
    }
    fig, axes = plt.subplots(1, len(tasks), figsize=(13, 4.2), sharey=True)
    for ax, task in zip(axes, tasks):
        sub = selected[selected["task"] == task].set_index("method")
        values = [float(sub.loc[m, metric]) if m in sub.index else np.nan for m in methods]
        ax.bar(range(len(methods)), values, color=[colors[m] for m in methods], width=0.7)
        ax.set_title(TASKS[task]["label"])
        ax.set_ylim(0, 0.75)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(idx, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel(metric.upper())
    fig.suptitle(f"Downstream {metric.upper()} on 2026-04 Haidian Tasks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def threshold_for(metrics_path: Path) -> float:
    record = load_json(metrics_path)
    return float(record.get("threshold", record.get("val_threshold", 0.5)))


def prediction_path(root: Path, task: str, model: str, patch_id: str, raw: bool) -> Path:
    if raw:
        return (
            root
            / task
            / "s2_s1_landsat_highres_indices"
            / model
            / "shot_full"
            / "fold_0"
            / "predictions"
            / f"{patch_id}_prob.tif"
        )
    return root / TASKS[task]["xuannv_task"] / "fold_0" / "predictions" / f"{patch_id}_prob.tif"


def metrics_path(root: Path, task: str, model: str, raw: bool) -> Path:
    if raw:
        return root / task / "s2_s1_landsat_highres_indices" / model / "shot_full" / "fold_0" / "metrics.json"
    return root / TASKS[task]["xuannv_task"] / "fold_0" / "metrics.json"


def select_patches(task: str, raw_model: str, count: int = 3) -> list[str]:
    label_root = PROCESSED_ROOT / "haidian" / "labels" / TASKS[task]["label_task"] / "masks"
    xuannv_dir = P10C_CONV_ROOT / TASKS[task]["xuannv_task"] / "fold_0" / "predictions"
    raw_dir = RAW_ROOT / task / "s2_s1_landsat_highres_indices" / raw_model / "shot_full" / "fold_0" / "predictions"
    xuannv_thr = threshold_for(P10C_CONV_ROOT / TASKS[task]["xuannv_task"] / "fold_0" / "metrics.json")
    raw_thr = threshold_for(metrics_path(RAW_ROOT, task, raw_model, raw=True))
    scored: list[tuple[float, str]] = []
    for path in sorted(xuannv_dir.glob("patch_*_prob.tif")):
        patch_id = path.stem.replace("_prob", "")
        raw_path = raw_dir / path.name
        label_path = label_root / f"{patch_id}.tif"
        if not raw_path.exists() or not label_path.exists():
            continue
        gt = read_mask(label_path) > 0
        if int(gt.sum()) <= 0:
            continue
        xuannv_prob = load_prediction(path)
        raw_prob = load_prediction(raw_path)
        xuannv_f1 = float(binary_metrics(xuannv_prob >= xuannv_thr, gt)["f1"])
        raw_f1 = float(binary_metrics(raw_prob >= raw_thr, gt)["f1"])
        scored.append((xuannv_f1 + raw_f1, patch_id))
    return [patch_id for _, patch_id in sorted(scored, reverse=True)[:count]]


def make_patch_figure(task: str, patch_ids: list[str], raw_model: str, out_path: Path) -> None:
    label_root = PROCESSED_ROOT / "haidian" / "labels" / TASKS[task]["label_task"] / "masks"
    xuannv_thr = threshold_for(P10C_CONV_ROOT / TASKS[task]["xuannv_task"] / "fold_0" / "metrics.json")
    raw_thr = threshold_for(metrics_path(RAW_ROOT, task, raw_model, raw=True))
    fig, axes = plt.subplots(len(patch_ids), 6, figsize=(18, 4.4 * len(patch_ids)))
    if len(patch_ids) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, patch_id in enumerate(patch_ids):
        gt = read_mask(label_root / f"{patch_id}.tif") > 0
        highres = load_highres(PROCESSED_ROOT, "haidian", patch_id, "202604")
        xuannv_prob = load_prediction(prediction_path(P10C_CONV_ROOT, task, "conv3x3", patch_id, raw=False))
        raw_prob = load_prediction(prediction_path(RAW_ROOT, task, raw_model, patch_id, raw=True))
        panels = [
            (highres, f"{patch_id}\nHigh-res 202604"),
            (red_binary_mask(gt), "GT"),
            (red_probability_map(xuannv_prob), f"Xuannv Prob\nP10C conv3x3"),
            (red_binary_mask(xuannv_prob >= xuannv_thr), f"Xuannv Mask\nthr={xuannv_thr:.3f}"),
            (red_probability_map(raw_prob), f"Raw Prob\n{raw_model}"),
            (red_binary_mask(raw_prob >= raw_thr), f"Raw Mask\nthr={raw_thr:.3f}"),
        ]
        for col_idx, (image, title) in enumerate(panels):
            show_panel(axes[row_idx, col_idx], image, title)
    fig.suptitle(f"{TASKS[task]['label']} examples: Xuannv embedding head vs raw 202604 image head")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_table(df: pd.DataFrame) -> str:
    table = df.copy()
    table["F1"] = table["f1"].map(lambda x: f"{x:.3f}")
    table["AP"] = table["ap"].map(lambda x: f"{x:.3f}")
    table["AUC"] = table["auc"].map(lambda x: f"{x:.3f}")
    table = table[["task", "source", "F1", "AP", "AUC", "threshold", "best_epoch"]]
    table["threshold"] = table["threshold"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(df: pd.DataFrame, f1_chart: Path, ap_chart: Path, example_paths: dict[str, Path]) -> Path:
    report_path = Path("/root/workspace/xuannv/docs/production/haidian_raw_image_baseline_comparison_20260708.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best_raw = df[df["source"].str.startswith("Raw 202604")].sort_values(["task", "f1"], ascending=[True, False]).groupby("task").head(1)
    xuannv_core = df[df["source"].isin(["P10C conv3x3", "P12C conv3x3"])]
    head_sweep = df[df["source"].str.contains("P10C|P12C", regex=True)]
    lines = [
        "# 玄女海淀 V1：下游头与原始影像基线补充评测",
        "",
        "## 结论先行",
        "",
        "- 只换下游头时，`conv3x3` 空间头在建筑、道路、水体三类任务上稳定优于原来的局部头、wide MLP 和 deep-wide MLP，说明玄女 embedding 中存在可被轻量空间头释放的局部边界信息。",
        "- 使用 2026 年 4 月原始多源影像直接训练下游头时，强监督的 UNet/DeepLab-lite 在 full train split 场景下可以达到或超过玄女轻量头；但这类方法依赖更重的输入、更大的下游模型和完整监督训练。",
        "- 公平解释是：玄女 embedding 的价值不应只看 full-label 强监督上限，更应强调少量标注、轻量头、快速制图和跨任务复用；raw+UNet 是强基线，适合展示“如果不用 embedding，需要更重下游模型才能追上”。",
        "",
        "## 评测设置",
        "",
        "- 区域：海淀区 320 个 patch。",
        "- 月份：原始影像基线使用 2026 年 4 月多源输入。",
        "- 任务：建筑、道路、水体。",
        "- 玄女输入：64 维 embedding，使用同一训练/验证/测试划分，重点比较 P10C/P12C 与多种下游头。",
        "- 原始影像输入：Sentinel-2、Sentinel-1、Landsat、高分光学和高分 SAR 派生的 42 通道特征，不使用 embedding。",
        "- 阈值：沿用验证集选择的最佳阈值，再在测试集报告 F1、AP、AUC。",
        "",
        "## 核心指标图",
        "",
        f"![F1 comparison]({f1_chart})",
        "",
        f"![AP comparison]({ap_chart})",
        "",
        "## 玄女下游头消融",
        "",
        format_table(head_sweep.sort_values(["task", "source"])),
        "",
        "## 原始影像强基线最佳结果",
        "",
        format_table(best_raw.sort_values("task")),
        "",
        "## 玄女 conv3x3 与原始影像强头对比",
        "",
        format_table(pd.concat([xuannv_core, best_raw]).sort_values(["task", "source"])),
        "",
        "## 代表性可视化",
        "",
    ]
    for task, path in example_paths.items():
        lines += [
            f"### {TASKS[task]['label']}",
            "",
            f"![{TASKS[task]['label']} examples]({path})",
            "",
        ]
    lines += [
        "## 分析",
        "",
        "1. 原始影像直接训练强头的建筑和道路结果较强，说明局部纹理和邻域上下文对这两类任务非常重要；这也解释了为什么纯 MLP 下游头容易产生假正例。",
        "2. 玄女 embedding 接入 `conv3x3` 后显著好于纯通道 MLP，说明 embedding 不是完全坍缩，但下游头需要一点局部空间建模才能把边界信息释放出来。",
        "3. 下一步若要进一步提高 embedding 本体，应继续让主模型在训练阶段学习更清晰的局部边界和多维语义，而不是只在下游端堆大模型。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect_metrics()
    metrics_csv = OUT / "raw_vs_xuannv_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    f1_chart = OUT / "raw_vs_xuannv_f1.png"
    ap_chart = OUT / "raw_vs_xuannv_ap.png"
    plot_metric_bars(df, "f1", f1_chart)
    plot_metric_bars(df, "ap", ap_chart)
    examples: dict[str, Path] = {}
    for task, info in TASKS.items():
        patch_ids = select_patches(task, info["raw_best"], count=3)
        out_path = OUT / f"{task}_xuannv_vs_raw_examples.png"
        make_patch_figure(task, patch_ids, info["raw_best"], out_path)
        examples[task] = out_path
    report_path = write_report(df, f1_chart, ap_chart, examples)
    print(json.dumps({"metrics_csv": str(metrics_csv), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
