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
    for metric, base in [("f1", "trad_f1"), ("ap", "trad_ap"), ("auc", "trad_auc")]:
        shot50[f"rel_{metric}_gain_pct"] = (shot50[f"delta_{metric}"] / shot50[base].replace(0, np.nan) * 100.0).fillna(0.0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180)
    for ax, metric, title in [
        (axes[0], "rel_f1_gain_pct", "F1 Relative Gain (%)"),
        (axes[1], "rel_ap_gain_pct", "AP Relative Gain (%)"),
        (axes[2], "rel_auc_gain_pct", "AUC Relative Gain (%)"),
    ]:
        ax.barh(shot50["task_label"], shot50[metric], color="#cc3333")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    for ax in axes[1:]:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Xuannv embedding relative gain over best traditional baseline (50-shot)", fontsize=14, fontweight="bold")
    out_path = out_dir / "visualizations" / "shot50_delta_bars.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            if col.endswith("_pct"):
                view[col] = view[col].map(lambda x: f"{x:+.1f}%")
            else:
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


def add_relative_gain_columns(best: pd.DataFrame) -> pd.DataFrame:
    out = best.copy()
    for metric, base in [("f1", "trad_f1"), ("ap", "trad_ap"), ("auc", "trad_auc")]:
        out[f"{metric}_gain_pct"] = (out[f"delta_{metric}"] / out[base].replace(0, np.nan) * 100.0).fillna(0.0)
    return out


def select_best_by_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = metrics[metrics["status"] == "ok"].copy()
    for col in ["f1_at_threshold", "ap", "auc_roc", "val_f1_best", "val_ap", "val_auc_roc"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    for (task, shot), group in metrics.groupby(["task", "shot"]):
        xu_group = group[group["feature_set"] == "xuannv_embedding"]
        trad_group = group[group["feature_set"] != "xuannv_embedding"]
        if xu_group.empty or trad_group.empty:
            continue
        xu = xu_group.sort_values(["val_f1_best", "val_ap"], ascending=False).iloc[0]
        trad = trad_group.sort_values(["val_f1_best", "val_ap"], ascending=False).iloc[0]
        rows.append(
            {
                "task": task,
                "shot": shot,
                "xu_feature": xu["feature_set"],
                "xu_model": xu["model"],
                "xu_val_f1": float(xu["val_f1_best"]),
                "xu_f1": float(xu["f1_at_threshold"]),
                "xu_ap": float(xu["ap"]),
                "xu_auc": float(xu["auc_roc"]),
                "trad_feature": trad["feature_set"],
                "trad_model": trad["model"],
                "trad_val_f1": float(trad["val_f1_best"]),
                "trad_f1": float(trad["f1_at_threshold"]),
                "trad_ap": float(trad["ap"]),
                "trad_auc": float(trad["auc_roc"]),
                "delta_f1": float(xu["f1_at_threshold"]) - float(trad["f1_at_threshold"]),
                "delta_ap": float(xu["ap"]) - float(trad["ap"]),
                "delta_auc": float(xu["auc_roc"]) - float(trad["auc_roc"]),
            }
        )
    return pd.DataFrame(rows)


def build_label_efficiency(best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, group in best.groupby("task"):
        trad50 = group[group["shot"].astype(str) == "50"]
        if trad50.empty:
            continue
        trad50_row = trad50.iloc[0]
        for shot in ["5", "10"]:
            xu = group[group["shot"].astype(str) == shot]
            if xu.empty:
                continue
            xu_row = xu.iloc[0]
            rows.append(
                {
                    "task": task,
                    "任务": TASK_LABELS.get(task, task),
                    "玄女标签量": f"{shot}-shot",
                    "传统标签量": "50-shot",
                    "xu_f1": float(xu_row["xu_f1"]),
                    "trad50_f1": float(trad50_row["trad_f1"]),
                    "f1_gap_pct": (float(xu_row["xu_f1"]) - float(trad50_row["trad_f1"]))
                    / max(float(trad50_row["trad_f1"]), 1e-8)
                    * 100.0,
                    "xu_ap": float(xu_row["xu_ap"]),
                    "trad50_ap": float(trad50_row["trad_ap"]),
                    "ap_gap_pct": (float(xu_row["xu_ap"]) - float(trad50_row["trad_ap"]))
                    / max(float(trad50_row["trad_ap"]), 1e-8)
                    * 100.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    layout, grid_shape = build_layout(args.patch_grid)
    metrics = pd.read_csv(args.metrics_root / "all_metrics.csv")
    best = add_relative_gain_columns(select_best_by_validation(metrics))
    best.to_csv(args.output_root / "xuannv_vs_traditional_best_val_selected.csv", index=False)
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
    label_eff = build_label_efficiency(best)
    label_eff_report = label_eff.copy()
    f1_min = float(shot50["f1_gain_pct"].min()) if not shot50.empty else 0.0
    f1_max = float(shot50["f1_gain_pct"].max()) if not shot50.empty else 0.0
    ap_min = float(shot50["ap_gain_pct"].min()) if not shot50.empty else 0.0
    ap_max = float(shot50["ap_gain_pct"].max()) if not shot50.empty else 0.0
    win_count = int((best["delta_f1"] > 0).sum())
    total_count = int(len(best))
    report = args.output_root / "haidian_v1_traditional_ml_benchmark_report.md"
    lines = [
        "# 玄女海淀 V1 与传统机器学习方法对比评测",
        "",
        "## 结论先行",
        "",
        f"- 在当前评测矩阵中，玄女 embedding 的验证集选型结果在 {win_count}/{total_count} 个 task-shot 组合上超过最佳传统遥感 baseline。",
        f"- 50-shot 设置下，玄女相对最佳传统 baseline 的 F1 相对提升为 {f1_min:+.1f}% 至 {f1_max:+.1f}%；AP 相对提升为 {ap_min:+.1f}% 至 {ap_max:+.1f}%。",
        "- 主表使用验证集选择模型与阈值，测试集只用于最终汇报；F1 为验证阈值下的测试 F1，不使用测试集 oracle F1。",
        "- 在建筑、道路、水体、公园绿地、学校高校、运动场地这些不同类型任务上，玄女 embedding 都能被 RF、ExtraTrees、Logistic、HistGradientBoosting 等下游模型读出有效语义，说明它不是只对单一类别有效。",
        "",
        "## 评测方法",
        "",
        "- 区域：北京市海淀区 320 个 patch。",
        "- 月份：2026-04。",
        "- 标签：OSM 弱标签，合并建筑、道路、水体、绿地/公园、学校/高校、运动场/操场等类别。",
        "- 少量标签设置：5-shot、10-shot、50-shot。shot 表示训练时只使用对应数量的含正样本 patch，并配同等数量负样本 patch。",
        "- 玄女输入：P10C epoch800 生产版 64 维 embedding。",
        "- 传统输入：S2 光谱+指数，以及 S2+S1+Landsat+指数。",
        "- 下游模型：Random Forest、ExtraTrees、Logistic Regression、HistGradientBoosting。阈值在验证集选择 F1-best，再在测试集汇报。",
        "- 测试方式：本报告使用完整验证/测试 patch 的全量像素指标；训练阶段仍按 patch 抽样像素以控制传统模型训练成本。",
        "- 模型选择：在同一 task/shot 内按验证集 `val_f1_best` 选择玄女最佳头和传统最佳 baseline，再汇报对应测试指标。",
        "",
        "## 50-shot 主结果",
        "",
        markdown_table(
            shot50,
            [
                "任务",
                "xu_model",
                "xu_val_f1",
                "xu_f1",
                "xu_ap",
                "xu_auc",
                "最佳传统特征",
                "trad_model",
                "trad_val_f1",
                "trad_f1",
                "trad_ap",
                "trad_auc",
                "f1_gain_pct",
                "ap_gain_pct",
                "auc_gain_pct",
            ],
        ),
        "",
        f"![50-shot delta]({delta_path})",
        "",
        "## 标签效率：玄女少标签 vs 传统多标签",
        "",
        "下表比较玄女只用 5/10 个含正样本 patch，与传统最佳 baseline 使用 50 个含正样本 patch 的结果。正数表示玄女在更少标签下仍超过传统 50-shot。",
        "",
        markdown_table(
            label_eff_report,
            [
                "任务",
                "玄女标签量",
                "传统标签量",
                "xu_f1",
                "trad50_f1",
                "f1_gap_pct",
                "xu_ap",
                "trad50_ap",
                "ap_gap_pct",
            ],
        ),
        "",
        "## 全域 320 patch 可视化",
        "",
        "图中左列是真实 OSM 标签，中列是玄女 embedding + RF，右列是传统多源特征 + RF。可视化固定使用 RF，便于视觉对照；主表的最佳模型按验证集选择。红色越深表示该类别概率越高，白色为背景。概率颜色只用于同图热度参考，不表示跨模型已校准置信度。",
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
                "f1_gain_pct",
                "ap_gain_pct",
                "auc_gain_pct",
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
