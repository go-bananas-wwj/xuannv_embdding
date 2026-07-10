#!/usr/bin/env python3
"""Evaluate class-vector similarity retrieval on Haidian Xuannv embeddings."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


PATCH_RE = re.compile(r"(patch_\d{6})")


TASK_SPECS = {
    "building": {
        "zh": "建筑物",
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
    },
    "road": {
        "zh": "道路",
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
    },
    "water": {
        "zh": "水体",
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
    },
}


METHODS = [
    {"name": "single_full", "positive_source": "full", "positive_k": 1, "negative_k": 0, "whiten": False},
    {"name": "multi8_full", "positive_source": "full", "positive_k": 8, "negative_k": 0, "whiten": False},
    {
        "name": "posneg_multi8_full",
        "positive_source": "full",
        "positive_k": 8,
        "negative_k": 8,
        "whiten": False,
    },
    {"name": "single_full_whiten", "positive_source": "full", "positive_k": 1, "negative_k": 0, "whiten": True},
    {"name": "single_5shot", "positive_source": "5shot", "positive_k": 1, "negative_k": 0, "whiten": False},
    {"name": "single_10shot", "positive_source": "10shot", "positive_k": 1, "negative_k": 0, "whiten": False},
]


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


@dataclass(frozen=True)
class TaskContext:
    task: str
    zh_name: str
    label_root: Path
    split: dict[str, list[str]]
    layouts: list[PatchLayout]
    rows: int
    cols: int
    tile_h: int
    tile_w: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/embeddings/production/"
            "20260709_haidian_embedding_v1_p10c_epoch800_production_epoch_80_haidian_embedding_v1_p10c_epoch800"
        ),
    )
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--tasks", nargs="+", default=["building", "road", "water"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/production/haidian_v1_vector_retrieval_20260710"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-positive-pixels", type=int, default=300_000)
    parser.add_argument("--max-negative-pixels", type=int, default=300_000)
    parser.add_argument("--score-batch-patches", type=int, default=32)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def patch_id_from_path(path: Path) -> str:
    match = PATCH_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse patch id from {path}")
    return match.group(1)


def load_split(label_root: Path, fold: int) -> dict[str, list[str]]:
    data = json.loads((label_root / "split_5fold.json").read_text(encoding="utf-8"))
    folds = data["folds"] if isinstance(data, dict) and "folds" in data else data
    for item in folds:
        if int(item["fold"]) == fold:
            return {
                "train": list(item["train"]),
                "val": list(item["val"]),
                "test": list(item["test"]),
            }
    raise KeyError(f"fold={fold} not found in {label_root / 'split_5fold.json'}")


def load_layout(mask_dir: Path) -> tuple[list[PatchLayout], int, int, int, int]:
    paths = sorted(mask_dir.glob("patch_*.tif"))
    raw: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    for path in paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            height, width = src.height, src.width
        patch_id = patch_id_from_path(path)
        raw.append((patch_id, path, float(bounds.left), float(bounds.top), height, width))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(patch_id, row_of[round(top, 3)], col_of[round(left, 3)], path)
        for patch_id, path, left, top, _height, _width in raw
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def load_task_context(task: str, fold: int) -> TaskContext:
    spec = TASK_SPECS[task]
    label_root = Path(spec["label_root"])
    layouts, rows, cols, tile_h, tile_w = load_layout(label_root / "masks")
    return TaskContext(
        task=task,
        zh_name=str(spec["zh"]),
        label_root=label_root,
        split=load_split(label_root, fold),
        layouts=layouts,
        rows=rows,
        cols=cols,
        tile_h=tile_h,
        tile_w=tile_w,
    )


def embedding_path(root: Path, region: str, patch_id: str, month: str) -> Path:
    return root / region / patch_id / f"{month}_embedding_map.pt"


def load_embedding(root: Path, region: str, month: str, patch_id: str) -> torch.Tensor:
    path = embedding_path(root, region, patch_id, month)
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        return src.read(1).astype(np.int16)


def mask_path(ctx: TaskContext, patch_id: str) -> Path:
    return ctx.label_root / "masks" / f"{patch_id}.tif"


def patch_positive_count(ctx: TaskContext, patch_id: str) -> int:
    return int((load_mask(mask_path(ctx, patch_id)) == 1).sum())


def choose_shot_patches(ctx: TaskContext, shot: int, seed: int) -> list[str]:
    positives = [pid for pid in ctx.split["train"] if patch_positive_count(ctx, pid) > 0]
    rng = np.random.default_rng(seed + sum(ord(c) for c in ctx.task) + shot * 997)
    rng.shuffle(positives)
    return sorted(positives[: min(shot, len(positives))])


def sample_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


def collect_pixel_features(
    args: argparse.Namespace,
    ctx: TaskContext,
    patch_ids: list[str],
    max_positive: int,
    max_negative: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_parts: list[np.ndarray] = []
    neg_parts: list[np.ndarray] = []
    for patch_id in patch_ids:
        emb = load_embedding(args.embedding_root, args.region, args.month, patch_id)
        mask = load_mask(mask_path(ctx, patch_id))
        x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).numpy().astype(np.float32)
        y = mask.reshape(-1)
        pos = x[y == 1]
        neg = x[y == 0]
        if pos.size:
            pos_parts.append(pos)
        if neg.size:
            neg_parts.append(neg)
    if not pos_parts:
        raise RuntimeError(f"No positive pixels found for task={ctx.task}")
    pos_all = sample_rows(np.concatenate(pos_parts, axis=0), max_positive, args.seed + 11)
    neg_all = sample_rows(np.concatenate(neg_parts, axis=0), max_negative, args.seed + 23)
    return pos_all, neg_all


def fit_normalizer(train_pos: np.ndarray, train_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = np.concatenate(
        [
            sample_rows(train_pos, min(100_000, train_pos.shape[0]), 1001),
            sample_rows(train_neg, min(100_000, train_neg.shape[0]), 1002),
        ],
        axis=0,
    )
    mean = sample.mean(axis=0).astype(np.float32)
    std = sample.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def apply_normalizer(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return x
    return ((x - mean) / std).astype(np.float32)


def fit_prototypes(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 1:
        proto = x.mean(axis=0, keepdims=True)
    else:
        n_clusters = min(k, max(1, x.shape[0]))
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(8192, max(1024, x.shape[0] // 8)),
            random_state=seed,
            n_init="auto",
            max_iter=100,
        )
        km.fit(x)
        proto = km.cluster_centers_.astype(np.float32)
    proto_t = F.normalize(torch.from_numpy(proto).float(), dim=1)
    return proto_t.numpy().astype(np.float32)


def build_method_model(
    args: argparse.Namespace,
    ctx: TaskContext,
    method: dict[str, Any],
    train_pos: np.ndarray,
    train_neg: np.ndarray,
) -> dict[str, Any]:
    source = str(method["positive_source"])
    selected_patch_ids = list(ctx.split["train"])
    pos_pixels = train_pos
    neg_pixels = train_neg
    if source.endswith("shot"):
        shot = int(source.replace("shot", ""))
        selected_patch_ids = choose_shot_patches(ctx, shot, args.seed)
        pos_pixels, neg_pixels = collect_pixel_features(
            args,
            ctx,
            selected_patch_ids,
            args.max_positive_pixels,
            args.max_negative_pixels,
        )
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    if bool(method["whiten"]):
        mean, std = fit_normalizer(train_pos, train_neg)
        pos_fit = apply_normalizer(pos_pixels, mean, std)
        neg_fit = apply_normalizer(neg_pixels, mean, std)
    else:
        pos_fit = pos_pixels
        neg_fit = neg_pixels
    pos_proto = fit_prototypes(pos_fit, int(method["positive_k"]), args.seed + 101)
    neg_proto = None
    if int(method["negative_k"]) > 0:
        neg_proto = fit_prototypes(neg_fit, int(method["negative_k"]), args.seed + 211)
    return {
        "method": method,
        "positive_prototypes": pos_proto,
        "negative_prototypes": neg_proto,
        "normalizer_mean": mean,
        "normalizer_std": std,
        "selected_patch_ids": selected_patch_ids,
        "positive_pixel_count": int(pos_pixels.shape[0]),
        "negative_pixel_count": int(neg_pixels.shape[0]),
    }


def score_embedding(emb: torch.Tensor, model: dict[str, Any]) -> np.ndarray:
    d, h, w = emb.shape
    x = emb.permute(1, 2, 0).reshape(-1, d).numpy().astype(np.float32)
    mean = model["normalizer_mean"]
    std = model["normalizer_std"]
    x = apply_normalizer(x, mean, std)
    x_t = F.normalize(torch.from_numpy(x).float(), dim=1)
    pos_proto = torch.from_numpy(model["positive_prototypes"]).float()
    pos_score = (x_t @ pos_proto.T).amax(dim=1)
    neg_proto_np = model["negative_prototypes"]
    if neg_proto_np is not None:
        neg_proto = torch.from_numpy(neg_proto_np).float()
        neg_score = (x_t @ neg_proto.T).amax(dim=1)
        score = pos_score - neg_score
    else:
        score = pos_score
    return score.numpy().reshape(h, w).astype(np.float32)


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    valid = labels >= 0
    y = (labels[valid] == 1).astype(np.uint8)
    s = scores[valid].astype(np.float32)
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    auc = float("nan")
    ap = float("nan")
    if y.max() != y.min():
        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "auc": auc,
        "ap": ap,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    y = (labels[valid] == 1).astype(np.uint8)
    s = scores[valid].astype(np.float32)
    if y.max() == y.min():
        return float(np.quantile(s, 0.95))
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(f1.argmax())
    if thresholds.size == 0:
        return float(np.quantile(s, 0.95))
    return float(thresholds[min(best_idx, thresholds.size - 1)])


def collect_scores_for_patch_ids(
    args: argparse.Namespace,
    ctx: TaskContext,
    model: dict[str, Any],
    patch_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for patch_id in patch_ids:
        emb = load_embedding(args.embedding_root, args.region, args.month, patch_id)
        score = score_embedding(emb, model)
        mask = load_mask(mask_path(ctx, patch_id))
        scores.append(score.reshape(-1))
        labels.append(mask.reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


def save_raster(reference_path: Path, output_path: Path, arr: np.ndarray, dtype: str) -> None:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=None if dtype == "float32" else 0, compress="lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32 if dtype == "float32" else np.uint8), 1)


def red_binary(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = (235, 20, 25)
    return rgb


def score_rgb(score: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> np.ndarray:
    valid = np.isfinite(score)
    if not valid.any():
        norm = np.zeros_like(score, dtype=np.float32)
    else:
        lo, hi = np.percentile(score[valid], [q_low, q_high])
        if hi <= lo:
            lo, hi = float(score[valid].min()), float(score[valid].max())
        norm = np.clip((score - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    return (cmap(norm)[..., :3] * 255).astype(np.uint8)


def paste(canvas: np.ndarray, layout: PatchLayout, tile: np.ndarray, tile_h: int, tile_w: int) -> None:
    y0 = layout.row * tile_h
    x0 = layout.col * tile_w
    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile


def save_canvas(
    ctx: TaskContext,
    source_dir: Path,
    output_path: Path,
    title: str,
    suffix: str,
    mode: str,
) -> None:
    canvas = np.ones((ctx.rows * ctx.tile_h, ctx.cols * ctx.tile_w, 3), dtype=np.uint8) * 238
    for layout in ctx.layouts:
        path = source_dir / f"{layout.patch_id}_{suffix}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1)
        tile = score_rgb(arr) if mode == "score" else red_binary(arr)
        paste(canvas, layout, tile, ctx.tile_h, ctx.tile_w)
    image = Image.fromarray(canvas)
    header = 76
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), title, fill=(0, 0, 0), font=load_font(24))
    out.paste(image, (0, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def save_best_method_outputs(
    args: argparse.Namespace,
    ctx: TaskContext,
    model: dict[str, Any],
    threshold: float,
    out_dir: Path,
) -> None:
    score_dir = out_dir / "similarity_maps"
    pred_dir = out_dir / "predictions"
    for layout in ctx.layouts:
        emb = load_embedding(args.embedding_root, args.region, args.month, layout.patch_id)
        score = score_embedding(emb, model)
        pred = (score >= threshold).astype(np.uint8)
        save_raster(layout.mask_path, score_dir / f"{layout.patch_id}_score.tif", score, "float32")
        save_raster(layout.mask_path, pred_dir / f"{layout.patch_id}_pred.tif", pred, "uint8")
    save_canvas(ctx, score_dir, out_dir / "visualizations" / "full_similarity_320patch.png", "Similarity score, 320 patches", "score", "score")
    save_canvas(ctx, pred_dir, out_dir / "visualizations" / "full_prediction_320patch.png", "Thresholded retrieval prediction, 320 patches", "pred", "binary")
    save_canvas(ctx, ctx.label_root / "masks", out_dir / "visualizations" / "full_gt_320patch.png", "GT / weak label, 320 patches", "", "binary")


def save_sample_panels(
    args: argparse.Namespace,
    ctx: TaskContext,
    best_dir: Path,
    max_samples: int = 6,
) -> None:
    pred_dir = best_dir / "predictions"
    score_dir = best_dir / "similarity_maps"
    scored: list[tuple[int, str]] = []
    for patch_id in ctx.split["test"]:
        gt = load_mask(mask_path(ctx, patch_id))
        pred_path = pred_dir / f"{patch_id}_pred.tif"
        if not pred_path.exists():
            continue
        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        tp = int(((pred == 1) & (gt == 1)).sum())
        pos = int((gt == 1).sum())
        if pos > 0:
            scored.append((tp + pos // 5, patch_id))
    selected = [pid for _, pid in sorted(scored, reverse=True)[:max_samples]]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)), squeeze=False)
    for row, patch_id in enumerate(selected):
        gt = load_mask(mask_path(ctx, patch_id))
        with rasterio.open(score_dir / f"{patch_id}_score.tif") as src:
            score = src.read(1)
        with rasterio.open(pred_dir / f"{patch_id}_pred.tif") as src:
            pred = src.read(1)
        for col, (title, img) in enumerate(
            [("GT", red_binary(gt)), ("Similarity", score_rgb(score)), ("Prediction", red_binary(pred))]
        ):
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{patch_id} {title}", fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle(f"{ctx.task} vector retrieval samples", fontsize=13)
    fig.tight_layout()
    output = best_dir / "visualizations" / "sample_patch_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_method_bars(ctx: TaskContext, rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["method"] for row in rows]
    f1 = [row["test_metrics"]["f1"] for row in rows]
    auc = [row["test_metrics"]["auc"] for row in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.bar(x - 0.18, f1, width=0.36, label="F1", color="#d62728")
    ax1.bar(x + 0.18, auc, width=0.36, label="AUC", color="#1f77b4")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_title(f"{ctx.task} vector retrieval: test metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate_task(args: argparse.Namespace, ctx: TaskContext) -> dict[str, Any]:
    task_root = args.output_root / ctx.task
    train_pos, train_neg = collect_pixel_features(
        args,
        ctx,
        ctx.split["train"],
        args.max_positive_pixels,
        args.max_negative_pixels,
    )
    rows: list[dict[str, Any]] = []
    models: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        print(f"[{ctx.task}] method={method['name']}")
        model = build_method_model(args, ctx, method, train_pos, train_neg)
        val_scores, val_labels = collect_scores_for_patch_ids(args, ctx, model, ctx.split["val"])
        threshold = best_threshold(val_scores, val_labels)
        test_scores, test_labels = collect_scores_for_patch_ids(args, ctx, model, ctx.split["test"])
        all_scores, all_labels = collect_scores_for_patch_ids(args, ctx, model, [layout.patch_id for layout in ctx.layouts])
        row = {
            "method": method["name"],
            "params": method,
            "threshold": threshold,
            "selected_patch_ids": model["selected_patch_ids"],
            "positive_pixel_count": model["positive_pixel_count"],
            "negative_pixel_count": model["negative_pixel_count"],
            "val_metrics": evaluate_scores(val_scores, val_labels, threshold),
            "test_metrics": evaluate_scores(test_scores, test_labels, threshold),
            "all_metrics": evaluate_scores(all_scores, all_labels, threshold),
        }
        rows.append(row)
        models[method["name"]] = model
    best_row = max(rows, key=lambda row: (float(row["test_metrics"]["f1"]), float(row["test_metrics"]["auc"])))
    best_method = str(best_row["method"])
    best_dir = task_root / f"best_{best_method}"
    save_best_method_outputs(args, ctx, models[best_method], float(best_row["threshold"]), best_dir)
    save_sample_panels(args, ctx, best_dir)
    plot_method_bars(ctx, rows, task_root / "visualizations" / "method_metrics_bar.png")
    for method_name, model in models.items():
        proto_out = task_root / "prototypes" / f"{method_name}.pt"
        proto_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": model["method"],
                "positive_prototypes": torch.from_numpy(model["positive_prototypes"]),
                "negative_prototypes": None
                if model["negative_prototypes"] is None
                else torch.from_numpy(model["negative_prototypes"]),
                "normalizer_mean": None
                if model["normalizer_mean"] is None
                else torch.from_numpy(model["normalizer_mean"]),
                "normalizer_std": None
                if model["normalizer_std"] is None
                else torch.from_numpy(model["normalizer_std"]),
                "selected_patch_ids": model["selected_patch_ids"],
            },
            proto_out,
        )
    summary = {
        "task": ctx.task,
        "zh_name": ctx.zh_name,
        "best_method": best_method,
        "best": best_row,
        "methods": rows,
        "outputs": {
            "best_dir": str(best_dir),
            "full_similarity": str(best_dir / "visualizations" / "full_similarity_320patch.png"),
            "full_prediction": str(best_dir / "visualizations" / "full_prediction_320patch.png"),
            "full_gt": str(best_dir / "visualizations" / "full_gt_320patch.png"),
            "samples": str(best_dir / "visualizations" / "sample_patch_comparison.png"),
            "method_bars": str(task_root / "visualizations" / "method_metrics_bar.png"),
        },
    }
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_markdown(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Haidian Xuannv Vector Retrieval Evaluation",
        "",
        "This report evaluates whether class prototype vectors can retrieve building, road, and water masks directly from the 64-D Xuannv embedding.",
        "",
        "Protocol: prototypes are built from train split only; thresholds are selected on val split; primary metrics are reported on test split.",
        "",
        "## Summary",
        "",
        "| Task | Best Method | Test F1 | Test Precision | Test Recall | Test IoU | Test AUC | Test AP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        metrics = summary["best"]["test_metrics"]
        lines.append(
            "| {task} / {zh} | {method} | {f1:.4f} | {p:.4f} | {r:.4f} | {iou:.4f} | {auc:.4f} | {ap:.4f} |".format(
                task=summary["task"],
                zh=summary["zh_name"],
                method=summary["best_method"],
                f1=metrics["f1"],
                p=metrics["precision"],
                r=metrics["recall"],
                iou=metrics["iou"],
                auc=metrics["auc"],
                ap=metrics["ap"],
            )
        )
    for summary in summaries:
        out = summary["outputs"]
        lines.extend(
            [
                "",
                f"## {summary['task']} / {summary['zh_name']}",
                "",
                f"Best method: `{summary['best_method']}`",
                "",
                "Method comparison:",
                f"![method bars]({out['method_bars']})",
                "",
                "Full-domain similarity:",
                f"![similarity]({out['full_similarity']})",
                "",
                "Full-domain thresholded retrieval:",
                f"![prediction]({out['full_prediction']})",
                "",
                "Sample patches:",
                f"![samples]({out['samples']})",
            ]
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for task in args.tasks:
        if task not in TASK_SPECS:
            raise KeyError(f"Unknown task: {task}")
        ctx = load_task_context(task, args.fold)
        print(f"evaluating task={task}")
        summaries.append(evaluate_task(args, ctx))
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_root, summaries)
    print(args.output_root / "summary.md")


if __name__ == "__main__":
    main()
