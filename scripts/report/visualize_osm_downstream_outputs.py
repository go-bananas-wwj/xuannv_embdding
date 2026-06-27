#!/usr/bin/env python3
"""Visualize OSM downstream predictions, GT masks, and P1B embeddings.

The diagnostic sheets intentionally include thresholded masks and TP/FP/FN
maps because ROC-AUC can look high on sparse masks while the segmentation is
still dominated by background or false positives.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from sklearn.decomposition import PCA

DEFAULT_RUN_ROOT = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/"
    "p1b_osm_quick_20260627"
)
DEFAULT_EMBEDDING_ROOT = Path(
    "/data/xuannv_embedding/embeddings/v2_202512_202605/"
    "20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_"
    "v2_p1_sparse_sampler_hardneg_20260627_090500"
)
DEFAULT_PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")

TASKS = {
    "haidian_building_osm": {"region": "haidian", "label_task": "building_osm"},
    "haidian_road_osm": {"region": "haidian", "label_task": "road_osm"},
    "harbin_building_osm": {"region": "harbin", "label_task": "building_osm"},
    "harbin_road_osm": {"region": "harbin", "label_task": "road_osm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--month", default="202605")
    parser.add_argument("--samples-per-task", type=int, default=8)
    return parser.parse_args()


def stretch(image: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0)
    if image.ndim == 2:
        lo, hi = np.percentile(image, [lower, upper])
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    out = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        band = image[..., channel]
        lo, hi = np.percentile(band, [lower, upper])
        if hi > lo:
            out[..., channel] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def choose_month_file(files: list[Path], month: str) -> Path | None:
    files = [path for path in files if not path.name.endswith("_mask.tif")]
    if not files:
        return None
    month_re = re.compile(rf"_{month}(?:\d{{2}})?_")
    candidates = [path for path in files if month_re.search(path.name)]
    if candidates:
        return sorted(candidates)[-1]
    return sorted(files)[-1]


def load_highres(processed_root: Path, region: str, patch_id: str, month: str) -> np.ndarray | None:
    root = processed_root / region / "patches" / "highres_optical"
    path = choose_month_file(sorted(root.glob(f"*{patch_id}.tif")), month)
    if path is None:
        return None
    with rasterio.open(path) as src:
        arr = src.read()
    return stretch(np.transpose(arr[:3], (1, 2, 0)))


def load_embedding(embedding_root: Path, region: str, patch_id: str, month: str) -> torch.Tensor | None:
    candidates = [
        embedding_root / region / patch_id / f"{month}_embedding_map.pt",
        embedding_root / region / f"{region}_{patch_id}" / f"{month}_embedding_map.pt",
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
    return None


def embedding_pca(embedding: torch.Tensor | None) -> np.ndarray | None:
    if embedding is None:
        return None
    arr = embedding.float().numpy()
    channels, height, width = arr.shape
    flat = arr.reshape(channels, -1).T
    flat = flat - flat.mean(axis=0, keepdims=True)
    if float(np.nanstd(flat)) < 1e-8:
        return np.zeros((height, width, 3), dtype=np.float32)
    rgb = PCA(n_components=3).fit_transform(flat).reshape(height, width, 3)
    return stretch(rgb)


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def overlay_mask(image: np.ndarray | None, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray | None:
    if image is None:
        return None
    out = image.copy()
    h, w = out.shape[:2]
    y_idx = np.linspace(0, mask.shape[0] - 1, h).astype(int)
    x_idx = np.linspace(0, mask.shape[1] - 1, w).astype(int)
    up = mask[np.ix_(y_idx, x_idx)] > 0
    color_arr = np.array(color, dtype=np.float32)
    out[up] = out[up] * 0.45 + color_arr * 0.55
    return out


def load_prediction(pred_path: Path) -> np.ndarray:
    with rasterio.open(pred_path) as src:
        return src.read(1)


def load_threshold(pred_path: Path) -> float:
    metrics_path = pred_path.parent.parent / "metrics.json"
    if not metrics_path.exists():
        return 0.5
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(data.get("val_threshold", data.get("threshold", 0.5)))


def load_summary_metrics(task_root: Path) -> dict[str, Any]:
    summary_path = task_root / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], dict):
                return data[0]
            numeric_keys = {key for row in data if isinstance(row, dict) for key, value in row.items() if isinstance(value, (int, float))}
            return {key: float(np.mean([row[key] for row in data if isinstance(row, dict) and key in row])) for key in numeric_keys}
        if isinstance(data, dict):
            return data
    metrics_files = sorted(task_root.glob("fold_*/metrics.json"))
    if not metrics_files:
        return {}
    return json.loads(metrics_files[0].read_text(encoding="utf-8"))


def binary_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray) -> dict[str, float | int]:
    pred_bool = pred_bool.astype(bool)
    gt_bool = gt_bool.astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pred_positive_pixels": int(pred_bool.sum()),
        "pred_positive_ratio": float(pred_bool.mean()),
        "gt_positive_pixels": int(gt_bool.sum()),
        "gt_positive_ratio": float(gt_bool.mean()),
    }


def make_error_map(pred_bool: np.ndarray, gt_bool: np.ndarray) -> np.ndarray:
    pred_bool = pred_bool.astype(bool)
    gt_bool = gt_bool.astype(bool)
    out = np.ones((*gt_bool.shape, 3), dtype=np.float32) * 0.92
    out[np.logical_and(pred_bool, gt_bool)] = (0.08, 0.62, 0.20)  # TP
    out[np.logical_and(pred_bool, ~gt_bool)] = (1.00, 0.55, 0.00)  # FP
    out[np.logical_and(~pred_bool, gt_bool)] = (0.92, 0.05, 0.08)  # FN
    return out


def red_binary_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    mask_bool = mask.astype(bool)
    out = np.ones((*mask_bool.shape, 3), dtype=np.float32)
    out[mask_bool] = (0.92, 0.05, 0.08)
    return out


def red_probability_map(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0), 0.0, 1.0)
    out = np.ones((*prob.shape, 3), dtype=np.float32)
    red = np.array((0.92, 0.05, 0.08), dtype=np.float32)
    out = out * (1.0 - prob[..., None]) + red * prob[..., None]
    return out


def positive_pixels(label_root: Path, patch_id: str) -> int:
    mask_path = label_root / "masks" / f"{patch_id}.tif"
    if not mask_path.exists():
        return 0
    return int((read_mask(mask_path) > 0).sum())


def select_predictions(run_root: Path, task: str, label_root: Path, samples_per_task: int) -> list[Path]:
    pred_paths = sorted((run_root / task).glob("fold_*/predictions/*_prob.tif"))
    scored: list[dict[str, Any]] = []
    for pred_path in pred_paths:
        patch_id = pred_path.stem.removesuffix("_prob")
        gt_path = label_root / "masks" / f"{patch_id}.tif"
        if not gt_path.exists():
            continue
        pred = load_prediction(pred_path)
        gt = read_mask(gt_path) > 0
        threshold = load_threshold(pred_path)
        metrics_thr = binary_metrics(pred >= threshold, gt)
        metrics_05 = binary_metrics(pred >= 0.5, gt)
        scored.append(
            {
                "path": pred_path,
                "gt_positive": positive_pixels(label_root, patch_id),
                "pred_max": float(np.nanmax(pred)),
                "f1_threshold": float(metrics_thr["f1"]),
                "f1_05": float(metrics_05["f1"]),
                "fp_threshold": int(metrics_thr["fp"]),
                "fn_threshold": int(metrics_thr["fn"]),
            }
        )
    positives = [item for item in scored if item["gt_positive"] > 0]
    top_positive = sorted(positives, key=lambda item: item["gt_positive"], reverse=True)
    worst_f1 = sorted(positives, key=lambda item: (item["f1_threshold"], -item["fp_threshold"] - item["fn_threshold"]))
    high_fp = sorted(positives, key=lambda item: item["fp_threshold"], reverse=True)
    high_score = sorted(scored, key=lambda item: item["pred_max"], reverse=True)

    selected: list[Path] = []
    for group in (top_positive, worst_f1, high_fp, high_score):
        for item in group:
            path = item["path"]
            if path not in selected:
                selected.append(path)
            if len(selected) >= samples_per_task:
                return selected
    return selected


def show_panel(ax: plt.Axes, image: np.ndarray | None, title: str, cmap: str | None = None) -> None:
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
    elif cmap is not None:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    elif image.ndim == 2:
        ax.imshow(image, cmap="magma")
    else:
        ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_visual(
    task: str,
    pred_path: Path,
    region: str,
    label_task: str,
    embedding_root: Path,
    processed_root: Path,
    output_dir: Path,
    month: str,
) -> dict[str, Any]:
    patch_id = pred_path.stem.removesuffix("_prob")
    label_root = processed_root / region / "labels" / label_task
    gt_path = label_root / "masks" / f"{patch_id}.tif"
    highres = load_highres(processed_root, region, patch_id, month)
    emb_pca = embedding_pca(load_embedding(embedding_root, region, patch_id, month))
    pred_raw = load_prediction(pred_path)
    pred = stretch(pred_raw, lower=0.0, upper=100.0)
    pred_red = red_probability_map(pred)
    gt = (read_mask(gt_path) > 0).astype(np.float32) if gt_path.exists() else None
    threshold = load_threshold(pred_path)
    pred_binary_05 = (pred_raw >= 0.5).astype(np.float32)
    pred_binary_thr = (pred_raw >= threshold).astype(np.float32)
    gt_for_metrics = gt.astype(bool) if gt is not None else np.zeros_like(pred_binary_thr, dtype=bool)
    metrics_05 = binary_metrics(pred_binary_05 > 0, gt_for_metrics)
    metrics_thr = binary_metrics(pred_binary_thr > 0, gt_for_metrics)
    error_map = make_error_map(pred_binary_thr > 0, gt_for_metrics)
    gt_overlay = overlay_mask(highres, gt if gt is not None else np.zeros((128, 128)), (1.0, 0.05, 0.05))

    panels = [
        (highres, f"High-res {month}", None),
        (emb_pca, f"P1B Emb PCA {month}", None),
        (pred_red, "Prediction Prob", None),
        (red_binary_mask(pred_binary_05), "Pred >= 0.50", None),
        (red_binary_mask(pred_binary_thr), f"Pred >= {threshold:.3f}", None),
        (red_binary_mask(gt), "OSM GT", None),
        (error_map, "TP/FP/FN @thr", None),
        (gt_overlay, "GT Overlay", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.35))
    for ax, (image, title, cmap) in zip(axes, panels, strict=True):
        show_panel(ax, image, title, cmap)
    fig.suptitle(
        f"{task} | {patch_id} | F1@0.5={metrics_05['f1']:.3f} | "
        f"F1@thr={metrics_thr['f1']:.3f} | GT={metrics_thr['gt_positive_ratio']:.3f} | "
        f"Pred@thr={metrics_thr['pred_positive_ratio']:.3f}",
        fontsize=10,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task}_{patch_id}_osm_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "task": task,
        "region": region,
        "label_task": label_task,
        "patch_id": patch_id,
        "fold": pred_path.parent.parent.name,
        "figure": str(out_path),
        "prediction": str(pred_path),
        "gt_mask": str(gt_path) if gt_path.exists() else None,
        "threshold": threshold,
        "metrics_at_0_5": metrics_05,
        "metrics_at_val_threshold": metrics_thr,
        "missing": {
            "highres": highres is None,
            "embedding": emb_pca is None,
            "gt_mask": gt is None,
        },
    }


def write_index(records: list[dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# P1B OSM Downstream Diagnostic Sheets",
        "",
        "Legend: TP is green, FP is orange, FN is red. These sheets expose background-dominance and threshold-calibration problems that ROC-AUC can hide.",
        "",
        "## Task Metrics",
        "",
    ]
    task_roots = {record["task"]: Path(record["prediction"]).parents[2] for record in records}
    for task in sorted(task_roots):
        summary = load_summary_metrics(task_roots[task])
        if not summary:
            continue
        lines.append(
            f"- `{task}`: AUC={summary.get('auc_roc', 0):.4f}, "
            f"F1@0.5={summary.get('f1_0.5', 0):.4f}, "
            f"F1@val_thr={summary.get('f1_at_threshold', 0):.4f}, "
            f"mIoU={summary.get('miou', 0):.4f}, "
            f"val_thr={summary.get('threshold', summary.get('val_threshold', 0)):.4f}"
        )
    lines.append("")
    for record in records:
        fig = Path(record["figure"])
        m05 = record["metrics_at_0_5"]
        mthr = record["metrics_at_val_threshold"]
        lines.extend(
            [
                f"## {record['task']} / {record['patch_id']}",
                "",
                f"- region: `{record['region']}`",
                f"- label_task: `{record['label_task']}`",
                f"- fold: `{record['fold']}`",
                f"- val_threshold: `{record['threshold']:.6f}`",
                f"- GT positive ratio: `{mthr['gt_positive_ratio']:.6f}`",
                f"- Pred positive ratio @0.5: `{m05['pred_positive_ratio']:.6f}`, F1: `{m05['f1']:.4f}`",
                f"- Pred positive ratio @val_threshold: `{mthr['pred_positive_ratio']:.6f}`, F1: `{mthr['f1']:.4f}`, FP: `{mthr['fp']}`, FN: `{mthr['fn']}`",
                f"- prediction: `{record['prediction']}`",
                f"- gt_mask: `{record['gt_mask']}`",
                f"- missing: `{record['missing']}`",
                "",
                f"![{fig.name}]({fig.relative_to(output_root).as_posix()})",
                "",
            ]
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.run_root / "osm_diagnostic_sheets"
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        info = TASKS[task]
        label_root = args.processed_root / info["region"] / "labels" / info["label_task"]
        for pred_path in select_predictions(args.run_root, task, label_root, args.samples_per_task):
            records.append(
                make_visual(
                    task=task,
                    pred_path=pred_path,
                    region=info["region"],
                    label_task=info["label_task"],
                    embedding_root=args.embedding_root,
                    processed_root=args.processed_root,
                    output_dir=output_root / task,
                    month=args.month,
                )
            )
    write_index(records, output_root)
    print(f"saved {len(records)} diagnostic sheets to {output_root}")


if __name__ == "__main__":
    main()
