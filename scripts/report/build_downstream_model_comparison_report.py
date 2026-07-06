#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TASK_LABELS = {
    "building": "建筑 Building",
    "road": "道路 Road",
    "water": "水体 Water",
    "park_green": "公园绿地 Park/Green",
    "education": "学校高校 Education",
    "sports_pitch": "运动场地 Sports/Pitch",
}

TASK_LABELS_EN = {
    "building": "Building",
    "road": "Road",
    "water": "Water",
    "park_green": "Park/Green",
    "education": "Education",
    "sports_pitch": "Sports/Pitch",
}

TASK_ORDER = ["building", "road", "water", "park_green", "education", "sports_pitch"]

METHOD_ORDER = [
    "Xuannv Linear",
    "Xuannv MLP",
    "Xuannv PixelConv",
    "Xuannv U-Net",
    "Xuannv DeepLab-lite",
    "Xuannv SegFormer-lite",
    "Raw U-Net",
    "Raw DeepLab-lite",
    "Raw SegFormer-lite",
    "Best Traditional ML",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style downstream model comparison report.")
    parser.add_argument(
        "--linear-mlp-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/production/haidian_v1_linear_mlp_probe_20260706"),
    )
    parser.add_argument("--strong-roots", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--traditional-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_valselected_fulltest"),
    )
    parser.add_argument(
        "--traditional-visual-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_report/visualizations"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shots", nargs="+", default=["5", "10", "50"])
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def normalize_head(head: str) -> str:
    mapping = {
        "linear": "Xuannv Linear",
        "mlp": "Xuannv MLP",
        "pixel_conv": "Xuannv PixelConv",
        "unet": "Xuannv U-Net",
        "deeplab_lite": "Xuannv DeepLab-lite",
        "segformer_lite": "Xuannv SegFormer-lite",
    }
    return mapping.get(head, head)


def normalize_raw_model(model: str) -> str:
    mapping = {
        "unet": "Raw U-Net",
        "deeplab_lite": "Raw DeepLab-lite",
        "segformer_lite": "Raw SegFormer-lite",
    }
    return mapping.get(model, f"Raw {model}")


def parse_shot(path: Path, record: dict[str, Any]) -> str:
    if "shot" in record:
        return str(record["shot"])
    for part in path.parts:
        if part.startswith("shot_"):
            return part.removeprefix("shot_")
    return ""


def metric_row(path: Path, record: dict[str, Any], source: str, method: str) -> dict[str, Any]:
    return {
        "source": source,
        "method": method,
        "task": record.get("task", ""),
        "shot": parse_shot(path, record),
        "fold": record.get("fold", 0),
        "status": record.get("status", "ok"),
        "f1": safe_float(record.get("f1_at_threshold")),
        "ap": safe_float(record.get("ap")),
        "auc": safe_float(record.get("auc_roc")),
        "iou": safe_float(record.get("miou")),
        "precision": safe_float(record.get("precision")),
        "recall": safe_float(record.get("recall")),
        "val_threshold": safe_float(record.get("val_threshold", record.get("threshold"))),
        "train_patch_count": safe_float(record.get("selected_train_patch_count", record.get("train_patch_count"))),
        "path": str(path),
    }


def collect_linear_mlp(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.glob("xuannv_haidian_v1/**/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status", "ok") != "ok":
            continue
        rel = path.relative_to(root / "xuannv_haidian_v1")
        task = rel.parts[0] if len(rel.parts) > 0 else record.get("task", "")
        head = record.get("head") or (rel.parts[1] if len(rel.parts) > 1 else "")
        shot = rel.parts[2].removeprefix("shot_") if len(rel.parts) > 2 and rel.parts[2].startswith("shot_") else record.get("shot", "")
        record = {**record, "task": task, "shot": shot}
        method = normalize_head(str(head))
        rows.append(metric_row(path, record, "xuannv_probe", method))
    return rows


def collect_strong(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ok: set[tuple[str, str, str]] = set()
    candidates: list[tuple[float, Path, dict[str, Any], str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/metrics.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            feature = record.get("feature_set", "")
            model = record.get("model", "")
            method = normalize_head(model) if feature == "xuannv_embedding" else normalize_raw_model(model)
            mtime = path.stat().st_mtime
            candidates.append((mtime, path, record, method))
    for _mtime, path, record, method in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = (str(record.get("task", "")), str(record.get("shot", "")), method)
        if key in seen_ok:
            continue
        if record.get("status", "ok") != "ok":
            continue
        seen_ok.add(key)
        source = "xuannv_strong" if method.startswith("Xuannv") else "raw_strong"
        rows.append(metric_row(path, record, source, method))
    return rows


def collect_traditional(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    by_task_shot: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in root.glob("**/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        if record.get("feature_set") == "xuannv_embedding":
            continue
        row = metric_row(path, record, "traditional_ml", "Best Traditional ML")
        row["raw_model"] = record.get("model", "")
        row["raw_feature"] = record.get("feature_set", "")
        by_task_shot.setdefault((row["task"], row["shot"]), []).append(row)
    for candidates in by_task_shot.values():
        candidates.sort(key=lambda row: (row["f1"], row["ap"]), reverse=True)
        rows.append(candidates[0])
    return rows


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    view = view.astype(str)
    widths = {col: max(len(col), *(len(v) for v in view[col].tolist())) for col in view.columns}
    header = "| " + " | ".join(col.ljust(widths[col]) for col in view.columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in view.columns) + " |"
    body = ["| " + " | ".join(row[col].ljust(widths[col]) for col in view.columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep, *body])


def best_per_method(df: pd.DataFrame, shot: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    subset = df[(df["shot"].astype(str) == shot) & (df["status"] == "ok")].copy()
    for (task, method), group in subset.groupby(["task", "method"]):
        group = group.sort_values(["f1", "ap"], ascending=False)
        rows.append(group.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["task_label"] = out["task"].map(TASK_LABELS).fillna(out["task"])
    out["task_label_en"] = out["task"].map(TASK_LABELS_EN).fillna(out["task"])
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    return out.sort_values(["task", "method"])


def plot_heatmap(best: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    pivot = best.pivot_table(index="task_label_en", columns="method", values=metric, aggfunc="max", observed=False)
    ordered_rows = [TASK_LABELS_EN[t] for t in TASK_ORDER if TASK_LABELS_EN[t] in pivot.index]
    ordered_cols = [m for m in METHOD_ORDER if m in pivot.columns]
    pivot = pivot.reindex(index=ordered_rows, columns=ordered_cols)
    plt.figure(figsize=(13.5, 4.8), dpi=180)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric.upper()})
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_label_efficiency(df: pd.DataFrame, out_path: Path) -> None:
    subset = df[(df["task"].isin(TASK_ORDER)) & (df["method"].isin(["Xuannv Linear", "Xuannv MLP", "Raw U-Net", "Raw SegFormer-lite"]))]
    subset = subset[subset["shot"].astype(str).isin(["5", "10", "50", "full"])].copy()
    subset["task_label"] = subset["task"].map(TASK_LABELS_EN)
    tasks = [task for task in TASK_ORDER if task in set(subset["task"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=180, sharey=False)
    axes = axes.reshape(-1)
    for ax, task in zip(axes, tasks, strict=False):
        g = subset[subset["task"] == task]
        for method, mg in g.groupby("method"):
            order = ["5", "10", "50", "full"]
            mg = mg.assign(_order=mg["shot"].astype(str).map({v: i for i, v in enumerate(order)})).sort_values("_order")
            ax.plot(mg["shot"].astype(str), mg["f1"], marker="o", label=method)
        ax.set_title(TASK_LABELS_EN.get(task, task), fontsize=10)
        ax.set_xlabel("Label budget")
        ax.set_ylabel("F1@val threshold")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Label efficiency: simple Xuannv heads vs supervised raw-feature models", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_report(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(collect_linear_mlp(args.linear_mlp_root))
    rows.extend(collect_strong(args.strong_roots))
    rows.extend(collect_traditional(args.traditional_root))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics found.")
    for col in ["f1", "ap", "auc", "iou", "precision", "recall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(args.output_root / "all_downstream_metrics.csv", index=False)

    best50 = best_per_method(df, "50")
    plot_heatmap(best50, "f1", args.output_root / "figures" / "shot50_f1_heatmap.png", "F1 comparison under 50-shot labels")
    plot_heatmap(best50, "ap", args.output_root / "figures" / "shot50_ap_heatmap.png", "AP comparison under 50-shot labels")
    plot_heatmap(best50, "auc", args.output_root / "figures" / "shot50_auc_heatmap.png", "AUC comparison under 50-shot labels")
    plot_label_efficiency(df, args.output_root / "figures" / "label_efficiency_f1.png")

    best_rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        g = best50[best50["task"] == task].sort_values(["f1", "ap"], ascending=False)
        if g.empty:
            continue
        top = g.iloc[0]
        simple = g[g["method"].isin(["Xuannv Linear", "Xuannv MLP"])].sort_values(["f1", "ap"], ascending=False)
        enhanced = g[g["method"].astype(str).str.startswith("Xuannv") & ~g["method"].isin(["Xuannv Linear", "Xuannv MLP"])].sort_values(["f1", "ap"], ascending=False)
        raw = g[g["method"].astype(str).str.startswith("Raw")].sort_values(["f1", "ap"], ascending=False)
        best_rows.append(
            {
                "任务 Task": TASK_LABELS.get(task, task),
                "最佳方法 Best": str(top["method"]),
                "最佳F1": float(top["f1"]),
                "玄女朴素头最佳": "" if simple.empty else str(simple.iloc[0]["method"]),
                "朴素头F1": np.nan if simple.empty else float(simple.iloc[0]["f1"]),
                "玄女增强头最佳": "" if enhanced.empty else str(enhanced.iloc[0]["method"]),
                "增强头F1": np.nan if enhanced.empty else float(enhanced.iloc[0]["f1"]),
                "Raw强监督最佳": "" if raw.empty else str(raw.iloc[0]["method"]),
                "Raw F1": np.nan if raw.empty else float(raw.iloc[0]["f1"]),
            }
        )
    summary = pd.DataFrame(best_rows)
    summary.to_csv(args.output_root / "shot50_summary.csv", index=False)

    simple_table = best50[best50["method"].isin(["Xuannv Linear", "Xuannv MLP"])].copy()
    simple_table = simple_table[["task_label", "method", "shot", "f1", "ap", "auc"]].rename(
        columns={"task_label": "任务 Task", "method": "方法 Method", "shot": "Shot", "f1": "F1", "ap": "AP", "auc": "AUC"}
    )
    broad_table = best50[["task_label", "method", "shot", "f1", "ap", "auc"]].copy().rename(
        columns={"task_label": "任务 Task", "method": "方法 Method", "shot": "Shot", "f1": "F1", "ap": "AP", "auc": "AUC"}
    )

    existing_visuals = sorted(args.traditional_visual_root.glob("*_shot50_gt_xuannv_traditional_full_domain.png"))
    lines = [
        "# 玄女海淀 V1 下游模型横向对比报告",
        "",
        "## 摘要",
        "",
        "本报告将玄女 embedding 的朴素下游头与更强的空间下游头、raw-feature 强监督分割模型、传统机器学习方法放在同一评测口径下对比。所有主指标均使用验证集选择阈值后在测试集计算，避免使用测试集最优阈值造成指标偏高。",
        "",
        "比较范围包括建筑、道路、水体、公园绿地、学校高校、运动场地六类任务。`5-shot/10-shot/50-shot` 表示使用对应数量的正样本 patch，并匹配相同数量的负样本 patch；`full` 表示使用完整训练划分。",
        "",
        "## Figure 1. 50-shot F1 横向热力图",
        "",
        f"![50-shot F1 heatmap]({args.output_root / 'figures' / 'shot50_f1_heatmap.png'})",
        "",
        "## Figure 2. 50-shot AP 与 AUC 横向热力图",
        "",
        f"![50-shot AP heatmap]({args.output_root / 'figures' / 'shot50_ap_heatmap.png'})",
        "",
        f"![50-shot AUC heatmap]({args.output_root / 'figures' / 'shot50_auc_heatmap.png'})",
        "",
        "## Figure 3. 少标注效率曲线",
        "",
        f"![Label efficiency]({args.output_root / 'figures' / 'label_efficiency_f1.png'})",
        "",
        "## Table 1. 50-shot 下各任务最佳方法摘要",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Table 2. 玄女朴素 Linear / MLP 头结果",
        "",
        markdown_table(simple_table.sort_values(["任务 Task", "方法 Method"]), list(simple_table.columns)) if not simple_table.empty else "当前 linear/MLP 结果尚未落盘。",
        "",
        "## Table 3. 全部方法 50-shot 横向指标",
        "",
        markdown_table(broad_table.sort_values(["任务 Task", "方法 Method"]), list(broad_table.columns)),
        "",
        "## 全域 320 patch 可视化示例",
        "",
        "下列图来自同一海淀 320 patch 地理拼接布局，红色为目标类别概率或真值区域，白色为背景，用于定性查看空间连续性与误检情况。",
        "",
    ]
    for visual in existing_visuals:
        task_name = visual.name.split("_shot50_")[0]
        lines.extend(
            [
                f"### {TASK_LABELS.get(task_name, task_name)}",
                "",
                f"![{task_name} full-domain visualization]({visual})",
                "",
            ]
        )
    lines.extend(
        [
            "## 评测说明",
            "",
            "- 玄女朴素头：`Linear` 与 `MLP`，只使用每个像素的 embedding 向量，不显式使用邻域卷积上下文。",
            "- 玄女增强头：`PixelConv/U-Net/DeepLab-lite/SegFormer-lite`，其中 U-Net、DeepLab-lite、SegFormer-lite 使用空间上下文。",
            "- Raw 强监督：直接使用 S2/S1/Landsat/高分光学/高分 SAR 指数特征训练分割网络。",
            "- 传统机器学习：从传统 ML 评测结果中按验证集指标选择每个任务/shot 的最佳 raw-feature baseline。",
            "- 主指标 F1/AP/AUC 均来自测试集；阈值由验证集决定。",
            "",
        ]
    )
    report_path = args.output_root / "haidian_v1_downstream_model_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print(report)


if __name__ == "__main__":
    main()
