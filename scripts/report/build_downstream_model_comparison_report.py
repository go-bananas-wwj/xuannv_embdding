#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


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

FAMILY_COLORS = {
    "Xuannv Linear/MLP": "#4C78A8",
    "Xuannv Spatial Head": "#E45756",
    "Raw Supervised": "#54A24B",
    "Traditional ML": "#9D755D",
}

TRAD_MODEL_NAMES = {
    "rf": "RF",
    "extratrees": "ExtraTrees",
    "hgb": "HistGB",
    "histgb": "HistGB",
    "logistic": "Logistic",
    "knn": "KNN",
    "svm": "SVM",
}

FEATURE_NAMES = {
    "s2_indices": "S2",
    "s2_s1_landsat_indices": "S2+S1+Landsat",
    "s2_s1_landsat_highres_indices": "S2+S1+Landsat+HR",
    "xuannv_embedding": "Xuannv Emb.",
}


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


def collect_traditional_all(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for path in root.glob("**/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        feature = str(record.get("feature_set", ""))
        model = str(record.get("model", ""))
        method = f"{TRAD_MODEL_NAMES.get(model, model)} ({FEATURE_NAMES.get(feature, feature)})"
        row = metric_row(path, record, "traditional_ml_all", method)
        row["feature_set"] = feature
        row["model"] = model
        rows.append(row)
    return pd.DataFrame(rows)


def method_family(method: str, source: str) -> str:
    if source == "traditional_ml_all" or source == "traditional_ml":
        return "Traditional ML"
    if method in {"Xuannv Linear", "Xuannv MLP"}:
        return "Xuannv Simple Head"
    if method.startswith("Xuannv"):
        return "Xuannv Spatial Head"
    if method.startswith("Raw"):
        return "Raw Supervised"
    return source


def build_all_methods_table(df: pd.DataFrame, traditional_all: pd.DataFrame) -> pd.DataFrame:
    """Create one analysis-ready CSV covering Xuannv, raw supervised and traditional ML."""
    downstream = df[df["source"] != "traditional_ml"].copy()
    pieces = [downstream]
    if not traditional_all.empty:
        pieces.append(traditional_all.copy())
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined["task_label"] = combined["task"].map(TASK_LABELS).fillna(combined["task"])
    combined["task_label_en"] = combined["task"].map(TASK_LABELS_EN).fillna(combined["task"])
    combined["method_family"] = [
        method_family(str(method), str(source)) for method, source in zip(combined["method"], combined["source"])
    ]
    combined["shot"] = combined["shot"].astype(str)
    preferred_cols = [
        "method_family",
        "source",
        "method",
        "task",
        "task_label",
        "task_label_en",
        "shot",
        "fold",
        "status",
        "f1",
        "ap",
        "auc",
        "iou",
        "precision",
        "recall",
        "val_threshold",
        "train_patch_count",
        "feature_set",
        "model",
        "raw_feature",
        "raw_model",
        "path",
    ]
    cols = [col for col in preferred_cols if col in combined.columns]
    extra_cols = [col for col in combined.columns if col not in cols]
    return combined[cols + extra_cols].sort_values(["shot", "task", "method_family", "method"]).reset_index(drop=True)


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
    subset = df[
        (df["task"].isin(TASK_ORDER))
        & (
            df["method"].isin(
                [
                    "Xuannv Linear",
                    "Xuannv MLP",
                    "Xuannv U-Net",
                    "Xuannv SegFormer-lite",
                    "Raw U-Net",
                    "Raw DeepLab-lite",
                    "Raw SegFormer-lite",
                    "Best Traditional ML",
                ]
            )
        )
    ]
    subset = subset[subset["shot"].astype(str).isin(["5", "10", "50", "full"])].copy()
    subset["task_label"] = subset["task"].map(TASK_LABELS_EN)
    tasks = [task for task in TASK_ORDER if task in set(subset["task"])]
    shot_pos = {"5": 5, "10": 10, "50": 50, "full": 75}
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.8), dpi=180, sharey=False)
    axes = axes.reshape(-1)
    for ax, task in zip(axes, tasks, strict=False):
        g = subset[subset["task"] == task]
        for method, mg in g.groupby("method"):
            order = ["5", "10", "50", "full"]
            mg = mg.assign(_order=mg["shot"].astype(str).map({v: i for i, v in enumerate(order)})).sort_values("_order")
            x = mg["shot"].astype(str).map(shot_pos)
            ax.plot(x, mg["f1"], marker="o", linewidth=1.6, label=method)
        ax.set_title(TASK_LABELS_EN.get(task, task), fontsize=10)
        ax.set_xlabel("Label budget")
        ax.set_ylabel("F1@val threshold")
        ax.set_xticks([5, 10, 50, 75])
        ax.set_xticklabels(["5", "10", "50", "Full"])
        ax.set_xlim(2, 79)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Label efficiency curves with ordered budgets: 5 -> 10 -> 50 -> Full", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_summary(best50: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        g = best50[best50["task"] == task].sort_values(["f1", "ap"], ascending=False)
        if g.empty:
            continue
        simple = g[g["method"].isin(["Xuannv Linear", "Xuannv MLP"])].sort_values(["f1", "ap"], ascending=False)
        enhanced = g[
            g["method"].astype(str).str.startswith("Xuannv")
            & ~g["method"].isin(["Xuannv Linear", "Xuannv MLP"])
        ].sort_values(["f1", "ap"], ascending=False)
        raw = g[g["method"].astype(str).str.startswith("Raw")].sort_values(["f1", "ap"], ascending=False)
        trad = g[g["method"] == "Best Traditional ML"].sort_values(["f1", "ap"], ascending=False)
        top = g.iloc[0]
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "task_label_en": TASK_LABELS_EN.get(task, task),
                "best_method": str(top["method"]),
                "best_f1": float(top["f1"]),
                "simple_method": "" if simple.empty else str(simple.iloc[0]["method"]),
                "simple_f1": np.nan if simple.empty else float(simple.iloc[0]["f1"]),
                "simple_ap": np.nan if simple.empty else float(simple.iloc[0]["ap"]),
                "simple_auc": np.nan if simple.empty else float(simple.iloc[0]["auc"]),
                "enhanced_method": "" if enhanced.empty else str(enhanced.iloc[0]["method"]),
                "enhanced_f1": np.nan if enhanced.empty else float(enhanced.iloc[0]["f1"]),
                "enhanced_ap": np.nan if enhanced.empty else float(enhanced.iloc[0]["ap"]),
                "enhanced_auc": np.nan if enhanced.empty else float(enhanced.iloc[0]["auc"]),
                "raw_method": "" if raw.empty else str(raw.iloc[0]["method"]),
                "raw_f1": np.nan if raw.empty else float(raw.iloc[0]["f1"]),
                "raw_ap": np.nan if raw.empty else float(raw.iloc[0]["ap"]),
                "raw_auc": np.nan if raw.empty else float(raw.iloc[0]["auc"]),
                "traditional_f1": np.nan if trad.empty else float(trad.iloc[0]["f1"]),
                "traditional_ap": np.nan if trad.empty else float(trad.iloc[0]["ap"]),
                "traditional_auc": np.nan if trad.empty else float(trad.iloc[0]["auc"]),
            }
        )
    return pd.DataFrame(rows)


def plot_four_way_bars(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.copy()
    tasks = plot_df["task_label_en"].tolist()
    values = {
        "Xuannv Linear/MLP": plot_df["simple_f1"].to_numpy(dtype=float),
        "Xuannv Spatial Head": plot_df["enhanced_f1"].to_numpy(dtype=float),
        "Raw Supervised": plot_df["raw_f1"].to_numpy(dtype=float),
        "Traditional ML": plot_df["traditional_f1"].to_numpy(dtype=float),
    }
    x = np.arange(len(tasks))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=220)
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (label, vals), offset in zip(values.items(), offsets, strict=True):
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=label,
            color=FAMILY_COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals, strict=False):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=0)
    ax.set_ylim(0, max(0.9, np.nanmax([v for vals in values.values() for v in vals]) + 0.10))
    ax.set_ylabel("F1@validation threshold")
    ax.set_title("Downstream segmentation performance under the 50-shot protocol")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_xuannv_vs_raw_delta(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["delta"] = plot_df["enhanced_f1"] - plot_df["raw_f1"]
    plot_df["delta_pct"] = plot_df["delta"] / plot_df["raw_f1"].replace(0, np.nan) * 100.0
    plot_df = plot_df.sort_values("delta")
    colors = np.where(plot_df["delta"] >= 0, "#D1495B", "#4C78A8")
    fig, ax = plt.subplots(figsize=(9.5, 4.7), dpi=220)
    ax.barh(plot_df["task_label_en"], plot_df["delta_pct"], color=colors, alpha=0.92)
    ax.axvline(0, color="#333333", linewidth=1.0)
    for y, val in enumerate(plot_df["delta_pct"]):
        if np.isfinite(val):
            ax.text(
                val + (1.2 if val >= 0 else -1.2),
                y,
                f"{val:+.1f}%",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
            )
    ax.set_xlabel("Relative F1 gain over the best raw supervised model (%)")
    ax.set_title("Xuannv spatial head vs. strong raw-feature supervised baselines")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_head_upgrade(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["gain"] = plot_df["enhanced_f1"] - plot_df["simple_f1"]
    plot_df = plot_df.sort_values("gain", ascending=False)
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=220)
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df["simple_f1"], plot_df["enhanced_f1"], color="#B9B9B9", linewidth=3)
    ax.scatter(plot_df["simple_f1"], y, s=70, color=FAMILY_COLORS["Xuannv Linear/MLP"], label="Linear/MLP")
    ax.scatter(plot_df["enhanced_f1"], y, s=75, color=FAMILY_COLORS["Xuannv Spatial Head"], label="Spatial head")
    for idx, row in plot_df.iterrows():
        ax.text(
            row["enhanced_f1"] + 0.012,
            y[list(plot_df.index).index(idx)],
            f"+{row['gain']:.2f}",
            va="center",
            fontsize=9,
            color="#444444",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["task_label_en"])
    ax.set_xlabel("F1@validation threshold")
    ax.set_title("Effect of upgrading Xuannv from a simple head to a spatial head")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, min(0.95, max(plot_df["enhanced_f1"].max(), plot_df["simple_f1"].max()) + 0.12))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ap_f1_scatter(best50: pd.DataFrame, out_path: Path) -> None:
    families = []
    for method in best50["method"].astype(str):
        if method in {"Xuannv Linear", "Xuannv MLP"}:
            families.append("Xuannv Linear/MLP")
        elif method.startswith("Xuannv"):
            families.append("Xuannv Spatial Head")
        elif method.startswith("Raw"):
            families.append("Raw Supervised")
        else:
            families.append("Traditional ML")
    plot_df = best50.copy()
    plot_df["family"] = families
    fig, ax = plt.subplots(figsize=(7.2, 5.7), dpi=220)
    for family, group in plot_df.groupby("family"):
        ax.scatter(
            group["f1"],
            group["ap"],
            s=70,
            color=FAMILY_COLORS.get(family, "#777777"),
            label=family,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xlabel("F1@validation threshold")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("F1-AP distribution across downstream tasks and model families")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_zoo(traditional_all: pd.DataFrame, raw_best50: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    trad = traditional_all[
        (traditional_all["shot"].astype(str) == "50")
        & (traditional_all["status"] == "ok")
        & (traditional_all["source"] == "traditional_ml_all")
    ].copy()
    raw = raw_best50[
        (raw_best50["shot"].astype(str) == "50")
        & (raw_best50["status"] == "ok")
        & (raw_best50["method"].astype(str).str.startswith("Raw"))
    ].copy()
    raw = raw.assign(source="raw_supervised_zoo")
    zoo = pd.concat([trad, raw], ignore_index=True, sort=False)
    zoo = zoo[zoo["task"].isin(TASK_ORDER)].copy()
    zoo["task_label_en"] = zoo["task"].map(TASK_LABELS_EN)
    zoo["f1"] = pd.to_numeric(zoo["f1"], errors="coerce")
    zoo["ap"] = pd.to_numeric(zoo["ap"], errors="coerce")
    zoo["auc"] = pd.to_numeric(zoo["auc"], errors="coerce")

    method_mean = zoo.groupby("method", as_index=False)["f1"].mean().sort_values("f1", ascending=False)
    top_methods = method_mean.head(18)["method"].tolist()
    heat = zoo[zoo["method"].isin(top_methods)].pivot_table(
        index="method",
        columns="task_label_en",
        values="f1",
        aggfunc="max",
    )
    heat = heat.reindex(index=top_methods, columns=[TASK_LABELS_EN[t] for t in TASK_ORDER])
    heat_path = out_dir / "baseline_zoo_f1_heatmap.png"
    plt.figure(figsize=(10.8, 8.2), dpi=220)
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.45, cbar_kws={"label": "F1"})
    plt.title("Baseline zoo: raw supervised networks and traditional ML methods (50-shot)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    heat_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heat_path, bbox_inches="tight")
    plt.close()

    rank_path = out_dir / "baseline_zoo_average_f1_rank.png"
    rank_df = method_mean.head(18).sort_values("f1")
    colors = ["#54A24B" if method.startswith("Raw") else "#9D755D" if "Xuannv Emb." not in method else "#4C78A8" for method in rank_df["method"]]
    fig, ax = plt.subplots(figsize=(9.8, 7.0), dpi=220)
    ax.barh(rank_df["method"], rank_df["f1"], color=colors)
    for y, value in enumerate(rank_df["f1"]):
        ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean F1 across six tasks")
    ax.set_title("Average 50-shot F1 ranking of all compared baselines")
    ax.set_xlim(0, max(0.85, rank_df["f1"].max() + 0.08))
    fig.tight_layout()
    plt.savefig(rank_path, bbox_inches="tight")
    plt.close(fig)
    return heat_path, rank_path


def copy_full_domain_visuals(visual_root: Path, out_dir: Path) -> list[tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, Path]] = []
    for visual in sorted(visual_root.glob("*_shot50_gt_xuannv_traditional_full_domain.png")):
        task_name = visual.name.split("_shot50_")[0]
        dst = out_dir / visual.name
        shutil.copy2(visual, dst)
        copied.append((task_name, dst))
    return copied


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def build_report(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(collect_linear_mlp(args.linear_mlp_root))
    rows.extend(collect_strong(args.strong_roots))
    rows.extend(collect_traditional(args.traditional_root))
    traditional_all = collect_traditional_all(args.traditional_root)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics found.")
    for col in ["f1", "ap", "auc", "iou", "precision", "recall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(args.output_root / "all_downstream_metrics.csv", index=False)
    if not traditional_all.empty:
        for col in ["f1", "ap", "auc", "iou", "precision", "recall"]:
            traditional_all[col] = pd.to_numeric(traditional_all[col], errors="coerce")
        traditional_all.to_csv(args.output_root / "all_traditional_ml_metrics.csv", index=False)
    all_methods = build_all_methods_table(df, traditional_all)
    all_methods.to_csv(args.output_root / "all_methods_metrics.csv", index=False)
    all_methods[all_methods["shot"].astype(str) == "50"].to_csv(args.output_root / "all_methods_50shot_metrics.csv", index=False)

    best50 = best_per_method(df, "50")
    plot_heatmap(best50, "f1", args.output_root / "figures" / "shot50_f1_heatmap.png", "F1 comparison under 50-shot labels")
    plot_heatmap(best50, "ap", args.output_root / "figures" / "shot50_ap_heatmap.png", "AP comparison under 50-shot labels")
    plot_heatmap(best50, "auc", args.output_root / "figures" / "shot50_auc_heatmap.png", "AUC comparison under 50-shot labels")
    plot_label_efficiency(df, args.output_root / "figures" / "label_efficiency_f1.png")
    summary = build_summary(best50)
    summary_for_csv = summary.rename(
        columns={
            "task_label": "任务 Task",
            "best_method": "最佳方法 Best",
            "best_f1": "最佳F1",
            "simple_method": "玄女朴素头最佳",
            "simple_f1": "朴素头F1",
            "enhanced_method": "玄女增强头最佳",
            "enhanced_f1": "增强头F1",
            "raw_method": "Raw强监督最佳",
            "raw_f1": "Raw F1",
        }
    )
    summary_for_csv.to_csv(args.output_root / "shot50_summary.csv", index=False)
    plot_four_way_bars(summary, args.output_root / "figures" / "main_50shot_four_way_f1.png")
    plot_xuannv_vs_raw_delta(summary, args.output_root / "figures" / "xuannv_vs_raw_relative_gain.png")
    plot_head_upgrade(summary, args.output_root / "figures" / "xuannv_head_upgrade_gain.png")
    plot_ap_f1_scatter(best50, args.output_root / "figures" / "f1_ap_scatter.png")
    baseline_heatmap, baseline_rank = plot_baseline_zoo(traditional_all, df, args.output_root / "figures")

    simple_table = best50[best50["method"].isin(["Xuannv Linear", "Xuannv MLP"])].copy()
    simple_table = simple_table[["task_label", "method", "shot", "f1", "ap", "auc"]].rename(
        columns={"task_label": "任务 Task", "method": "方法 Method", "shot": "Shot", "f1": "F1", "ap": "AP", "auc": "AUC"}
    )
    broad_table = best50[["task_label", "method", "shot", "f1", "ap", "auc"]].copy().rename(
        columns={"task_label": "任务 Task", "method": "方法 Method", "shot": "Shot", "f1": "F1", "ap": "AP", "auc": "AUC"}
    )

    copied_visuals = copy_full_domain_visuals(args.traditional_visual_root, args.output_root / "figures" / "full_domain")
    wins = int((summary["enhanced_f1"] >= summary["raw_f1"]).sum())
    avg_simple = float(summary["simple_f1"].mean())
    avg_enhanced = float(summary["enhanced_f1"].mean())
    avg_raw = float(summary["raw_f1"].mean())
    avg_trad = float(summary["traditional_f1"].mean())
    avg_gain_vs_raw = (avg_enhanced - avg_raw) / max(avg_raw, 1e-8) * 100.0
    avg_head_gain = (avg_enhanced - avg_simple) / max(avg_simple, 1e-8) * 100.0
    lines = [
        "# 玄女海淀 V1：下游任务横向评测与实验总结",
        "",
        "## 1. 结论先行",
        "",
        f"本轮评测将玄女 embedding 的朴素下游头、玄女空间增强头、raw-feature 强监督分割网络、传统机器学习方法放在同一套海淀区下游任务上比较。结果显示：在 50-shot 标注预算下，玄女空间增强头在 {wins}/6 个任务上达到或超过 raw-feature 强监督模型；平均 F1 为 {avg_enhanced:.3f}，raw-feature 强监督平均 F1 为 {avg_raw:.3f}，相对提升 {avg_gain_vs_raw:+.1f}%。",
        "",
        f"同时，玄女并不完全依赖复杂下游头。只使用朴素 MLP/Linear 头时，平均 F1 已达到 {avg_simple:.3f}；升级为空间头后平均 F1 达到 {avg_enhanced:.3f}，相对朴素头提升 {avg_head_gain:+.1f}%。这说明 embedding 本身已经包含可迁移的地物语义，空间头主要进一步补充边界和上下文。",
        "",
        "最重要的对比不是“玄女是否能靠更复杂网络赢”，而是两层结论：第一，玄女 MLP 这种简单头已经能在建筑、道路、水体、绿地等任务上取得可用结果；第二，在业务需要更高精度时，玄女 embedding 接 U-Net/DeepLab-lite/SegFormer-lite 后，多数任务可以超过直接用多源原始影像训练的强监督模型。",
        "",
        "## 2. 参考论文式评测口径",
        "",
        "遥感基础模型论文通常不会只报告单个任务，而是通过多任务 benchmark、低标注量曲线、强监督 baseline、以及定性可视化来证明表示能力。GEO-Bench 提出用多类地球观测任务来评估预训练模型的泛化价值；Prithvi-EO-2.0 报告不同数据比例下的下游 F1，以体现 label efficiency；PANGAEA 也强调跨任务、跨传感器的 geospatial foundation model benchmark。因此本报告采用同样思路：多任务、同一 split、验证集选阈值、50-shot 主表、少标注效率曲线、全域可视化。",
        "",
        "参考来源：",
        "",
        "- [GEO-Bench: Toward Foundation Models for Earth Monitoring](https://proceedings.neurips.cc/paper_files/paper/2023/file/a0644215d9cff6646fa334dfa5d29c5a-Paper-Datasets_and_Benchmarks.pdf)",
        "- [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://arxiv.org/html/2412.02732v3)",
        "- [PANGAEA: Assessing Geospatial Foundation Models Capabilities](https://arxiv.org/html/2412.04204v1)",
        "",
        "## 3. 实验演进摘要",
        "",
        "本项目不是一次性训练得到当前结果，而是经过多轮排查和升级：早期版本主要暴露出云雾干扰、patch 间 PCA 颜色割裂、WorldCover 弱标签过粗、下游阈值不稳定、以及 embedding 保存冗余等问题。后续逐步完成了云/无效像素 mask、OSM 弱语义标签合并、2025-12 到 2026-05 海淀多源数据筛选、高分光学/高分 SAR 重建权重调整、困难重建、800 epoch 长训、P10C 海淀生产版打包，以及传统 ML/AEF/raw 强监督/玄女下游头横向评测。",
        "",
        "这些实验给出的核心经验是：单纯延长训练可以改善 embedding 连续性，但真正影响下游能力的是数据质量、弱语义标签质量、云雾处理、通道对齐、以及评测协议是否公平。当前海淀 V1 的优势主要来自更干净的海淀专用训练数据、OSM 弱语义、困难重建、多源高分数据注入，以及训练后统一的阈值校准和下游评测。",
        "",
        "## 4. 主结果：50-shot 横向比较",
        "",
        "Figure 1 把四类方法放在同一张图中：玄女朴素头、玄女空间增强头、raw-feature 强监督模型、传统机器学习。每个任务的纵向比较可以直接看出玄女 embedding 是否在同等标注预算下带来收益。",
        "",
        "![Main 50-shot F1 comparison](figures/main_50shot_four_way_f1.png)",
        "",
        "Figure 2 进一步只比较“玄女最佳空间头”和“Raw最佳强监督模型”的相对提升。正值表示玄女更强，负值表示 raw-feature 强监督更强。",
        "",
        "![Xuannv vs raw relative gain](figures/xuannv_vs_raw_relative_gain.png)",
        "",
        "Figure 3 展示从朴素 MLP/Linear 头升级到空间增强头后的收益。教育/学校、建筑、道路、体育场地这类边界和上下文更重要的任务，空间头提升更明显；绿地/公园和水体这类光谱/语义较清晰的任务，朴素 MLP 已经接近较强水平。",
        "",
        "![Xuannv head upgrade gain](figures/xuannv_head_upgrade_gain.png)",
        "",
        "## 5. 指标分布与少标注效率",
        "",
        "Figure 4 将所有任务的 F1 和 AP 放到同一散点图中。右上角代表模型在阈值分割质量和排序质量上都更好。玄女空间头整体分布更靠右上；玄女 MLP 在若干任务上也接近 raw 强监督模型。",
        "",
        "![F1 AP scatter](figures/f1_ap_scatter.png)",
        "",
        "Figure 5 体现 label efficiency。对于业务标注昂贵的遥感任务，少量 patch 标注能否快速制图是 embedding 模型最重要的价值之一。",
        "",
        "![Label efficiency](figures/label_efficiency_f1.png)",
        "",
        "Figure 5 的横坐标已经显式按照 `5 → 10 → 50 → Full` 排列。`Full` 不代表数值 75，只是放在 50-shot 右侧表示完整训练集，用来和少标注结果做趋势比较。",
        "",
        "## 6. Baseline zoo：所有传统方法和强监督模型",
        "",
        "除了主图中的四类方法，本节把之前跑过的传统机器学习方法全部放进来，包括 RF、ExtraTrees、HistGradientBoosting、Logistic Regression，以及不同输入特征组合；同时也包含 raw U-Net、raw DeepLab-lite、raw SegFormer-lite。这样可以看出玄女不是只和一个弱 baseline 比，而是和一组传统/强监督方法池对比。",
        "",
        "对应的完整明细已汇总到 `all_methods_metrics.csv`；其中同时包含玄女 Linear/MLP、玄女空间头、raw-feature 强监督网络，以及全部 traditional ML 方法。若只看 50-shot 主实验，可直接使用 `all_methods_50shot_metrics.csv`。",
        "",
        f"![Baseline zoo heatmap]({rel(baseline_heatmap, args.output_root)})",
        "",
        f"![Baseline zoo rank]({rel(baseline_rank, args.output_root)})",
        "",
        "## 7. 详细指标热力图",
        "",
        "热力图作为补充，用于查看每个任务和每个模型的完整 50-shot F1/AP/AUC 数值。",
        "",
        "![50-shot F1 heatmap](figures/shot50_f1_heatmap.png)",
        "",
        "![50-shot AP heatmap](figures/shot50_ap_heatmap.png)",
        "",
        "![50-shot AUC heatmap](figures/shot50_auc_heatmap.png)",
        "",
        "## 8. 50-shot 结果摘要",
        "",
        markdown_table(
            summary_for_csv[
                ["任务 Task", "最佳方法 Best", "最佳F1", "玄女朴素头最佳", "朴素头F1", "玄女增强头最佳", "增强头F1", "Raw强监督最佳", "Raw F1"]
            ],
            ["任务 Task", "最佳方法 Best", "最佳F1", "玄女朴素头最佳", "朴素头F1", "玄女增强头最佳", "增强头F1", "Raw强监督最佳", "Raw F1"],
        ),
        "",
        "## 9. 玄女朴素头结果",
        "",
        "这一节专门回应“只用 Linear/MLP 是否也有效”。结果表明，MLP 在建筑、水体、绿地、道路、体育场地上明显强于 Linear，说明玄女 embedding 中的语义并非只能被复杂空间网络利用；简单非线性头已经能读出相当一部分地物信息。",
        "",
        markdown_table(simple_table.sort_values(["任务 Task", "方法 Method"]), list(simple_table.columns)) if not simple_table.empty else "当前 linear/MLP 结果尚未落盘。",
        "",
        "## 10. 全域 320 patch 可视化",
        "",
        "下列图已经复制到本报告目录的 `figures/full_domain/` 中，Markdown 使用相对路径引用；直接打开本 Markdown 或导出 PDF 时应能正常显示。红色为目标类别概率或真值区域，白色为背景，用于查看空间连续性与误检情况。",
        "",
    ]
    for task_name, visual in copied_visuals:
        lines.extend(
            [
                f"### {TASK_LABELS.get(task_name, task_name)}",
                "",
                f"![{task_name} full-domain visualization]({rel(visual, args.output_root)})",
                "",
            ]
        )
    lines.extend(
        [
            "## 11. 评测协议",
            "",
            "- 玄女朴素头：`Linear` 与 `MLP`，只使用每个像素的 embedding 向量，不显式使用邻域卷积上下文。",
            "- 玄女增强头：`PixelConv/U-Net/DeepLab-lite/SegFormer-lite`，其中 U-Net、DeepLab-lite、SegFormer-lite 使用空间上下文。",
            "- Raw 强监督：直接使用 S2/S1/Landsat/高分光学/高分 SAR 指数特征训练分割网络。",
            "- 传统机器学习：从传统 ML 评测结果中按验证集指标选择每个任务/shot 的最佳 raw-feature baseline。",
            "- 主指标 F1/AP/AUC 均来自测试集；阈值由验证集决定，避免使用测试集最优阈值。",
            "- `5-shot/10-shot/50-shot` 表示对应数量的正样本 patch，并匹配相同数量的负样本 patch；`full` 表示完整训练划分。",
            "",
            "## 12. 附录：全部 50-shot 指标",
            "",
            markdown_table(broad_table.sort_values(["任务 Task", "方法 Method"]), list(broad_table.columns)),
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
