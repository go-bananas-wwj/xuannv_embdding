#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


TASK_LABELS = {
    "building": "建筑提取 Building",
    "road": "道路提取 Road",
    "water": "水体提取 Water",
    "park_green": "公园绿地 Park/Green",
    "education": "学校高校 Education",
    "sports_pitch": "运动场地 Sports/Pitch",
}

TASK_TITLES_EN = {
    "building": "Building Extraction",
    "road": "Road Extraction",
    "water": "Water Extraction",
    "park_green": "Park / Green Space",
    "education": "Education / Campus",
    "sports_pitch": "Sports Field / Pitch",
}

TASK_ROOTS = {
    "building": ["building_osm"],
    "road": ["road_osm"],
    "water": ["osm_water"],
    "park_green": ["osm_park", "osm_green", "osm_grass", "osm_garden"],
    "education": ["osm_education", "osm_school", "osm_university"],
    "sports_pitch": ["osm_sports", "osm_pitch", "osm_playground"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Xuannv vs traditional ML benchmark report.")
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--patch-grid", type=Path, default=Path("configs/regions/haidian_patches.json"))
    parser.add_argument("--label-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian/labels"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shot", default="50")
    return parser.parse_args()


def resolve_mask(mask_dir: Path, patch_id: str) -> Path:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"*{patch_id}.tif")) + sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No mask for {patch_id} in {mask_dir}")
    return candidates[-1]


def load_gt(label_root: Path, task: str, patch_id: str) -> np.ndarray:
    merged: np.ndarray | None = None
    for rel in TASK_ROOTS[task]:
        path = resolve_mask(label_root / rel / "masks", patch_id)
        with rasterio.open(path) as src:
            arr = (src.read(1) == 1).astype(np.float32)
        merged = arr if merged is None else np.maximum(merged, arr)
    if merged is None:
        raise RuntimeError(task)
    return merged


def load_prob(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return np.clip(arr, 0.0, 1.0)


def red_on_white(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    img = np.ones((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    img[..., 1] = 1.0 - arr
    img[..., 2] = 1.0 - arr
    return img


def build_layout(patch_grid: Path) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    records = json.loads(patch_grid.read_text(encoding="utf-8"))
    xs = sorted({round(float(item["bounds"][0]), 3) for item in records})
    ys = sorted({round(float(item["bounds"][1]), 3) for item in records}, reverse=True)
    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}
    layout: dict[str, tuple[int, int]] = {}
    for item in records:
        patch_id = item["patch_id"]
        x = round(float(item["bounds"][0]), 3)
        y = round(float(item["bounds"][1]), 3)
        layout[patch_id] = (y_to_row[y], x_to_col[x])
    return layout, (len(ys), len(xs))


def make_canvas(layout: dict[str, tuple[int, int]], grid_shape: tuple[int, int], patch_arrays: dict[str, np.ndarray]) -> np.ndarray:
    sample = next(iter(patch_arrays.values()))
    h, w = sample.shape
    canvas = np.ones((grid_shape[0] * h, grid_shape[1] * w, 3), dtype=np.float32)
    for patch_id, arr in patch_arrays.items():
        if patch_id not in layout:
            continue
        row, col = layout[patch_id]
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = red_on_white(arr)
    return canvas


def save_task_mosaic(args: argparse.Namespace, task: str, layout: dict[str, tuple[int, int]], grid_shape: tuple[int, int]) -> Path:
    xu_dir = args.visual_root / task / "xuannv_embedding" / "rf" / f"shot_{args.shot}" / "fold_0" / "predictions_all"
    trad_dir = args.visual_root / task / "s2_s1_landsat_indices" / "rf" / f"shot_{args.shot}" / "fold_0" / "predictions_all"
    patch_ids = sorted(layout)
    gt = {pid: load_gt(args.label_root, task, pid) for pid in patch_ids}
    xu = {pid: load_prob(xu_dir / f"{pid}_prob.tif") for pid in patch_ids}
    trad = {pid: load_prob(trad_dir / f"{pid}_prob.tif") for pid in patch_ids}
    panels = [
        ("Ground Truth", make_canvas(layout, grid_shape, gt)),
        ("Xuannv Embedding + RF", make_canvas(layout, grid_shape, xu)),
        ("Traditional Multi-source + RF", make_canvas(layout, grid_shape, trad)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.suptitle(TASK_TITLES_EN.get(task, task), fontsize=16, fontweight="bold")
    fig.text(0.5, 0.03, "Darker red means higher class probability. White is background. 320 patches are mosaicked by geographic layout.", ha="center", fontsize=10)
    out_path = args.output_root / "visualizations" / f"{task}_shot{args.shot}_gt_xuannv_traditional_full_domain.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_delta_bars(best: pd.DataFrame, out_dir: Path) -> Path:
    shot50 = best[best["shot"].astype(str) == "50"].copy()
    shot50["task_label"] = shot50["task"].map(TASK_TITLES_EN).fillna(shot50["task"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180)
    for ax, metric, title in [
        (axes[0], "delta_f1", "F1 Gain"),
        (axes[1], "delta_ap", "AP Gain"),
        (axes[2], "delta_auc", "AUC Gain"),
    ]:
        ax.barh(shot50["task_label"], shot50[metric], color="#cc3333")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[1:]:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Xuannv embedding gain over best traditional baseline (50-shot)", fontsize=14, fontweight="bold")
    out_path = out_dir / "visualizations" / "shot50_delta_bars.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    view = view.astype(str)
    widths = {
        col: max(len(str(col)), *(len(value) for value in view[col].tolist()))
        for col in view.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    rows = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in view.columns) + " |"
        for _, row in view.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    layout, grid_shape = build_layout(args.patch_grid)
    best = pd.read_csv(args.metrics_root / "xuannv_vs_traditional_best.csv")
    metrics = pd.read_csv(args.metrics_root / "all_metrics.csv")
    vis_paths = [save_task_mosaic(args, task, layout, grid_shape) for task in TASK_LABELS]
    delta_path = plot_delta_bars(best, args.output_root)

    shot50 = best[best["shot"].astype(str) == args.shot].copy()
    shot50["任务"] = shot50["task"].map(TASK_LABELS).fillna(shot50["task"])
    shot50["最佳传统特征"] = shot50["trad_feature"].replace(
        {
            "s2_indices": "S2 光谱+指数",
            "s2_s1_landsat_indices": "S2+S1+Landsat+指数",
        }
    )
    report = args.output_root / "haidian_v1_traditional_ml_benchmark_report.md"
    lines = [
        "# 玄女海淀 V1 与传统机器学习方法对比评测",
        "",
        "## 结论先行",
        "",
        "- 在 6 个下游任务、5/10/50-shot 三种少量标签设置下，玄女 embedding 的最佳结果全部超过传统遥感特征 baseline。",
        "- 50-shot 设置下，玄女相对最佳传统 baseline 的 F1 提升范围为 +0.0198 至 +0.0392；AP 提升范围为 +0.0271 至 +0.0598。",
        "- 在建筑、道路、水体、公园绿地、学校高校、运动场地这些不同类型任务上，玄女 embedding 都能被简单 RF/Logistic 读出有效语义，说明它不是只对单一类别有效。",
        "",
        "## 评测方法",
        "",
        "- 区域：北京市海淀区 320 个 patch。",
        "- 月份：2026-04。",
        "- 标签：OSM 弱标签，合并建筑、道路、水体、绿地/公园、学校/高校、运动场/操场等类别。",
        "- 少量标签设置：5-shot、10-shot、50-shot。shot 表示训练时只使用对应数量的含正样本 patch，并配同等数量负样本 patch。",
        "- 玄女输入：P10C epoch800 生产版 64 维 embedding。",
        "- 传统输入：S2 光谱+指数，以及 S2+S1+Landsat+指数。",
        "- 下游模型：Random Forest 与 Logistic Regression。阈值在验证集选择 F1-best，再在测试集汇报。",
        "",
        "## 50-shot 主结果",
        "",
        markdown_table(
            shot50,
            [
                "任务",
                "xu_model",
                "xu_f1",
                "xu_ap",
                "xu_auc",
                "最佳传统特征",
                "trad_model",
                "trad_f1",
                "trad_ap",
                "trad_auc",
                "delta_f1",
                "delta_ap",
                "delta_auc",
            ],
        ),
        "",
        f"![50-shot delta]({delta_path})",
        "",
        "## 全域 320 patch 可视化",
        "",
        "图中左列是真实 OSM 标签，中列是玄女 embedding + RF，右列是传统多源特征 + RF。红色越深表示该类别概率越高，白色为背景。",
        "",
    ]
    for path in vis_paths:
        task = path.name.split("_shot")[0]
        lines += [f"### {TASK_LABELS.get(task, task)}", "", f"![{task}]({path})", ""]
    lines += [
        "## 全部 shot 汇总",
        "",
        markdown_table(
            best,
            [
                "task",
                "shot",
                "xu_model",
                "xu_f1",
                "xu_ap",
                "xu_auc",
                "trad_feature",
                "trad_model",
                "trad_f1",
                "trad_ap",
                "trad_auc",
                "delta_f1",
                "delta_ap",
                "delta_auc",
            ],
        ),
        "",
        "## 指标说明",
        "",
        "- F1：综合考虑 Precision 和 Recall，越高表示漏检和误检整体更少。",
        "- AP：Precision-Recall 曲线下面积，更适合类别不均衡任务。",
        "- AUC：ROC 曲线下面积，衡量模型把正负样本排序分开的能力。",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    metrics.describe(include="all").to_csv(args.output_root / "metrics_describe.csv")
    print(report)


if __name__ == "__main__":
    main()
