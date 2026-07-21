#!/usr/bin/env python3
"""Evaluate the embedding-api PU + Query recipe on held-out Haidian patches.

This is deliberately an evaluation implementation, not a copy of the API
gallery.  Support polygons are taken only from the train split; thresholding
uses the positive-versus-reliable-background support data specified by the
recipe; all reported metrics are computed only on the independent test split.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import average_precision_score, roc_auc_score


TASKS = {
    "building": ("建筑物", Path("/data/xuannv_embedding/processed/haidian/labels/building_osm")),
    "road": ("道路", Path("/data/xuannv_embedding/processed/haidian/labels/road_osm")),
    "water": ("水体", Path("/data/xuannv_embedding/processed/haidian/labels/osm_water")),
}
BACKGROUND_WEIGHT = 0.65
BACKGROUND_QUANTILE = 0.30
BACKGROUND_EXCLUSION_PIXELS = 3
MAX_BACKGROUND_PER_SUPPORT = 2048
QUERY_BLEND = 0.12
QUERY_QUANTILE = 0.997
QUERY_MIN_PIXELS, QUERY_MAX_PIXELS = 4, 128
QUERY_MIN_MARGIN = 0.05
QUERY_MAX_GROWTH, QUERY_MIN_AREA_CAP = 1.35, 64

PLOT_NAMES = {"building": "Building", "road": "Road", "water": "Water"}


@dataclass(frozen=True)
class PolygonSupport:
    patch_id: str
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=Path(
        "/data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604"))
    parser.add_argument("--month", default="202604")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--polygon-counts", nargs="+", type=int, default=[1, 3, 5, 9])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--output-root", type=Path, default=Path(
        "/data/xuannv_embedding/experiments/production/haidian_p10c_pu_query_strict_20260721"))
    return parser.parse_args()


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def emb_path(root: Path, patch_id: str, month: str) -> Path:
    return root / "haidian" / patch_id / f"{month}_embedding_map.pt"


def load_embedding(args: argparse.Namespace, patch_id: str) -> np.ndarray:
    return torch.load(emb_path(args.embedding_root, patch_id, args.month), map_location="cpu", weights_only=True).numpy().astype(np.float32)


def load_mask(root: Path, patch_id: str) -> np.ndarray:
    with rasterio.open(root / "masks" / f"{patch_id}.tif") as src:
        return src.read(1).astype(np.uint8) == 1


def split_for(root: Path, fold: int) -> dict[str, list[str]]:
    data = json.loads((root / "split_5fold.json").read_text(encoding="utf-8"))
    for item in data.get("folds", data):
        if int(item["fold"]) == fold:
            return {key: list(item[key]) for key in ("train", "val", "test")}
    raise KeyError(f"fold={fold} not found")


def collect_components(root: Path, patch_ids: list[str], minimum_area: int = 5) -> list[PolygonSupport]:
    supports: list[PolygonSupport] = []
    for patch_id in patch_ids:
        components, count = label(load_mask(root, patch_id), structure=np.ones((3, 3), dtype=np.uint8))
        for component_id in range(1, count + 1):
            component = components == component_id
            if int(component.sum()) >= minimum_area:
                supports.append(PolygonSupport(patch_id, component))
    return supports


def pick_supports(candidates: list[PolygonSupport], count: int, seed: int) -> list[PolygonSupport]:
    # Deterministic shuffle avoids hand-picked easy locations while retaining polygon-level samples.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order[:count]]


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(steps):
        padded = np.pad(result, 1)
        result = np.logical_or.reduce([padded[y:y + result.shape[0], x:x + result.shape[1]] for y in range(3) for x in range(3)])
    return result


def normalize_map(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return l2((np.moveaxis(feature, 0, -1) - mean) / np.maximum(std, 1e-5))


def fbeta(positive_scores: np.ndarray, negative_scores: np.ndarray, threshold: float, beta: float = 0.5) -> float:
    tp = float((positive_scores >= threshold).sum())
    fp = float((negative_scores >= threshold).sum())
    fn = float((positive_scores < threshold).sum())
    b2 = beta * beta
    return (1 + b2) * tp / max((1 + b2) * tp + b2 * fn + fp, 1e-8)


def train_pu_query(args: argparse.Namespace, supports: list[PolygonSupport]) -> dict[str, np.ndarray | float | int]:
    unique_ids = sorted({item.patch_id for item in supports})
    raw = {patch_id: load_embedding(args, patch_id) for patch_id in unique_ids}
    all_pixels = np.concatenate([raw[patch_id].reshape(raw[patch_id].shape[0], -1).T for patch_id in unique_ids])
    mean, std = all_pixels.mean(0), np.maximum(all_pixels.std(0), 1e-5)
    features = {patch_id: normalize_map(value, mean, std) for patch_id, value in raw.items()}
    polygon_vectors = [l2(features[item.patch_id][item.mask].mean(0, keepdims=True))[0] for item in supports]
    foreground = l2(np.mean(polygon_vectors, axis=0, keepdims=True))[0]
    union_masks: dict[str, np.ndarray] = {key: np.zeros(features[key].shape[:2], dtype=bool) for key in unique_ids}
    for item in supports:
        union_masks[item.patch_id] |= item.mask
    backgrounds: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed + len(supports) * 17)
    for patch_id in unique_ids:
        feature = features[patch_id]
        reliable = ~dilate(union_masks[patch_id], BACKGROUND_EXCLUSION_PIXELS)
        candidates = feature[reliable]
        similarities = candidates @ foreground
        cutoff = np.quantile(similarities, BACKGROUND_QUANTILE)
        candidates = candidates[similarities <= cutoff]
        if candidates.size:
            if len(candidates) > MAX_BACKGROUND_PER_SUPPORT:
                candidates = candidates[rng.choice(len(candidates), MAX_BACKGROUND_PER_SUPPORT, replace=False)]
            backgrounds.append(candidates)
    background = l2(np.concatenate(backgrounds).mean(0, keepdims=True))[0]
    positive_scores = np.concatenate([features[item.patch_id][item.mask] @ foreground - BACKGROUND_WEIGHT * (features[item.patch_id][item.mask] @ background) for item in supports])
    negative_scores = np.concatenate([values @ foreground - BACKGROUND_WEIGHT * (values @ background) for values in backgrounds])
    lo, hi = min(positive_scores.min(), negative_scores.min()), max(positive_scores.max(), negative_scores.max())
    thresholds = np.linspace(lo, hi, 180, dtype=np.float32)
    threshold = float(max(thresholds, key=lambda value: fbeta(positive_scores, negative_scores, float(value))))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "foreground": foreground.astype(np.float32), "background": background.astype(np.float32), "threshold": threshold, "polygon_count": len(supports)}


def score_pu_query(feature: np.ndarray, model: dict[str, np.ndarray | float | int]) -> tuple[np.ndarray, bool]:
    pixels = normalize_map(feature, model["mean"], model["std"])
    foreground, background, threshold = model["foreground"], model["background"], float(model["threshold"])
    base = gaussian_filter(pixels @ foreground - BACKGROUND_WEIGHT * (pixels @ background), sigma=0.55)
    confidence = max(float(np.quantile(base, QUERY_QUANTILE)), threshold + QUERY_MIN_MARGIN)
    selected = base >= confidence
    selected_count = int(selected.sum())
    if not QUERY_MIN_PIXELS <= selected_count <= QUERY_MAX_PIXELS:
        return base.astype(np.float32), False
    query = l2(pixels[selected].mean(0, keepdims=True))[0]
    query_score = gaussian_filter(pixels @ query - BACKGROUND_WEIGHT * (pixels @ background), sigma=0.55)
    refined = (1.0 - QUERY_BLEND) * base + QUERY_BLEND * query_score
    base_area, refined_area = int((base >= threshold).sum()), int((refined >= threshold).sum())
    if refined_area > max(QUERY_MIN_AREA_CAP, math.ceil(base_area * QUERY_MAX_GROWTH)):
        return base.astype(np.float32), False
    return refined.astype(np.float32), True


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    y = labels.astype(np.uint8).reshape(-1)
    s = scores.astype(np.float32).reshape(-1)
    prediction = s >= threshold
    tp, fp, fn = int((prediction & (y == 1)).sum()), int((prediction & (y == 0)).sum()), int((~prediction & (y == 1)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8), "iou": tp / max(tp + fp + fn, 1), "auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def binary_rgb(mask: np.ndarray, color: tuple[int, int, int] = (226, 42, 42)) -> np.ndarray:
    output = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    output[mask] = color
    return output


def make_visual(root: Path, args: argparse.Namespace, task: str, label_root: Path, supports: list[PolygonSupport], model: dict[str, np.ndarray | float | int], test_ids: list[str]) -> Path:
    # Select a representative only after the aggregate test calculation; it never affects the score.
    candidates: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for patch_id in test_ids:
        score, _ = score_pu_query(load_embedding(args, patch_id), model)
        gt = load_mask(label_root, patch_id)
        value = metrics(score, gt, float(model["threshold"]))["f1"] if gt.any() else -1.0
        candidates.append((value, patch_id, score, gt, load_embedding(args, patch_id)))
    _, patch_id, score, gt, feature = max(candidates, key=lambda item: item[0])
    support = supports[0]
    support_feature = load_embedding(args, support.patch_id)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        ("Support polygon (train only)", binary_rgb(support.mask, (180, 40, 180))),
        ("Support embedding PCA", pca_rgb(support_feature)),
        ("Test embedding PCA", pca_rgb(feature)),
        ("Test GT (not used in PU fitting)", binary_rgb(gt, (45, 170, 80))),
        ("PU + Query score", score),
        ("Prediction (red)", binary_rgb(score >= float(model["threshold"]))),
    ]
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image, cmap="turbo" if image.ndim == 2 else None)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle(f"{PLOT_NAMES[task]} | {len(supports)} polygons | representative independent test patch {patch_id}", fontsize=13)
    fig.tight_layout()
    output = root / f"{task}_{len(supports)}polygons_example.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def pca_rgb(feature: np.ndarray) -> np.ndarray:
    pixels = np.moveaxis(feature, 0, -1).reshape(-1, feature.shape[0]).astype(np.float32)
    pixels -= pixels.mean(0)
    _, _, vectors = np.linalg.svd(pixels[::8], full_matrices=False)
    output = (pixels @ vectors[:3].T).reshape(feature.shape[1], feature.shape[2], 3)
    low, high = np.percentile(output, 2, axis=(0, 1)), np.percentile(output, 98, axis=(0, 1))
    return (255 * np.clip((output - low) / np.maximum(high - low, 1e-6), 0, 1)).astype(np.uint8)


def plot_summary(rows: list[dict[str, object]], output: Path) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    counts = sorted({int(row["polygon_count"]) for row in rows})
    fig, (ax_f1, ax_auc) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for task in tasks:
        values = [next(float(row["metrics"]["f1"]) for row in rows if row["task"] == task and row["polygon_count"] == count) for count in counts]
        auc = [next(float(row["metrics"]["auc"]) for row in rows if row["task"] == task and row["polygon_count"] == count) for count in counts]
        ax_f1.plot(counts, values, marker="o", label=PLOT_NAMES[task])
        ax_auc.plot(counts, auc, marker="o", label=PLOT_NAMES[task])
    for axis, title in ((ax_f1, "Independent-test F1"), (ax_auc, "Independent-test AUC")):
        axis.set_title(title); axis.set_xlabel("Number of labelled polygons"); axis.set_xticks(counts); axis.set_ylim(0, 1); axis.grid(alpha=.25); axis.legend()
    fig.suptitle("P10C PU + Query: sparse polygon mapping")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for task in args.tasks:
        name, label_root = TASKS[task]
        split = split_for(label_root, args.fold)
        components = collect_components(label_root, split["train"])
        if len(components) < max(args.polygon_counts):
            raise RuntimeError(f"{task}: only {len(components)} usable train polygons")
        for polygon_count in args.polygon_counts:
            supports = pick_supports(components, polygon_count, args.seed + sum(map(ord, task)) + polygon_count * 1009)
            model = train_pu_query(args, supports)
            test_scores, test_labels, adapted = [], [], 0
            for patch_id in split["test"]:
                score, did_adapt = score_pu_query(load_embedding(args, patch_id), model)
                test_scores.append(score); test_labels.append(load_mask(label_root, patch_id)); adapted += int(did_adapt)
            result = metrics(np.concatenate(test_scores), np.concatenate(test_labels), float(model["threshold"]))
            row = {"task": task, "task_zh": name, "polygon_count": polygon_count, "support_patch_ids": [item.patch_id for item in supports], "unique_support_patches": len({item.patch_id for item in supports}), "threshold": model["threshold"], "query_adapted_test_patches": adapted, "test_patch_count": len(split["test"]), "metrics": result}
            rows.append(row)
            if polygon_count == 5:
                row["visualization"] = str(make_visual(args.output_root, args, task, label_root, supports, model, split["test"]))
            print(f"[{task}] polygons={polygon_count} test_f1={result['f1']:.4f} auc={result['auc']:.4f}", flush=True)
    plot_summary(rows, args.output_root / "pu_query_sparse_summary.png")
    payload = {"protocol": {"source": "embedding-api commit 530b8f4", "fold": args.fold, "month": args.month, "embedding": "P10C epoch_800 64D", "train_split_only": True, "test_metrics_only": True, "threshold_source": "support positives + reliable PU background, F0.5 sweep (180 thresholds)", "query": "sigma=.55, q=.997, blend=.12, guarded area growth"}, "rows": rows}
    (args.output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
