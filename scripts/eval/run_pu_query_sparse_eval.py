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

import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.eval.run_strong_downstream_benchmark import (
    fixed_highres_feature_map,
    load_manifest,
)


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

@dataclass(frozen=True)
class PolygonSupport:
    patch_id: str
    mask: np.ndarray


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    root: Path
    month: str
    channels: int
    data_root: Path | None = None


MANIFEST_CACHE: dict[tuple[Path, Path], dict[str, object]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, default=Path(
        "/data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604"))
    parser.add_argument(
        "--aef-embedding-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/processed/haidian/"
            "manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"
        ),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--output-root", type=Path, default=Path(
        "/data/xuannv_embedding/experiments/production/haidian_p10c_pu_query_strict_20260721"))
    return parser.parse_args()


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def emb_path(root: Path, patch_id: str, month: str) -> Path:
    return root / "haidian" / patch_id / f"{month}_embedding_map.pt"


def feature_specs(args: argparse.Namespace) -> dict[str, FeatureSpec]:
    return {
        "xuannv": FeatureSpec("xuannv", "embedding_map", args.embedding_root, "202604", 64),
        "aef": FeatureSpec("aef", "embedding_map", args.aef_embedding_root, "202512", 64),
        "traditional": FeatureSpec(
            "traditional",
            "fixed_highres_feature_map",
            args.manifest,
            "202604",
            42,
            args.data_root,
        ),
    }


def load_feature(spec: FeatureSpec, patch_id: str) -> np.ndarray:
    if spec.kind == "embedding_map":
        path = emb_path(spec.root, patch_id, spec.month)
        feature = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    elif spec.kind == "fixed_highres_feature_map":
        if spec.data_root is None:
            raise ValueError("Traditional FeatureSpec requires data_root")
        cache_key = (spec.root, spec.data_root)
        records = MANIFEST_CACHE.get(cache_key)
        if records is None:
            records = load_manifest(spec.root, spec.data_root)
            MANIFEST_CACHE[cache_key] = records
        feature = fixed_highres_feature_map(records[patch_id], spec.month)
    else:
        raise KeyError(f"Unsupported feature kind: {spec.kind}")
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 3 or feature.shape[0] != spec.channels:
        raise ValueError(
            f"Expected {spec.channels}xHxW {spec.name} feature for {patch_id}, got {feature.shape}"
        )
    return feature


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


def train_pu_query(spec: FeatureSpec, supports: list[PolygonSupport], seed: int) -> dict[str, np.ndarray | float | int]:
    unique_ids = sorted({item.patch_id for item in supports})
    raw = {patch_id: load_feature(spec, patch_id) for patch_id in unique_ids}
    all_pixels = np.concatenate([raw[patch_id].reshape(raw[patch_id].shape[0], -1).T for patch_id in unique_ids])
    mean, std = all_pixels.mean(0), np.maximum(all_pixels.std(0), 1e-5)
    features = {patch_id: normalize_map(value, mean, std) for patch_id, value in raw.items()}
    polygon_vectors = [l2(features[item.patch_id][item.mask].mean(0, keepdims=True))[0] for item in supports]
    foreground = l2(np.mean(polygon_vectors, axis=0, keepdims=True))[0]
    union_masks: dict[str, np.ndarray] = {key: np.zeros(features[key].shape[:2], dtype=bool) for key in unique_ids}
    for item in supports:
        union_masks[item.patch_id] |= item.mask
    backgrounds: list[np.ndarray] = []
    rng = np.random.default_rng(seed + len(supports) * 17)
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


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    specs = feature_specs(args)
    for task in args.tasks:
        name, label_root = TASKS[task]
        split = split_for(label_root, args.fold)
        components = collect_components(label_root, split["train"])
        polygon_count = 3
        if len(components) < polygon_count:
            raise RuntimeError(f"{task}: only {len(components)} usable train polygons")
        supports = pick_supports(components, polygon_count, args.seed + sum(map(ord, task)))
        for spec in specs.values():
            model = train_pu_query(spec, supports, args.seed)
            test_scores, test_labels, adapted = [], [], 0
            for patch_id in split["test"]:
                score, did_adapt = score_pu_query(load_feature(spec, patch_id), model)
                test_scores.append(score); test_labels.append(load_mask(label_root, patch_id)); adapted += int(did_adapt)
            result = metrics(np.concatenate(test_scores), np.concatenate(test_labels), float(model["threshold"]))
            row = {"task": task, "task_zh": name, "feature": spec.name, "polygon_count": polygon_count, "support_patch_ids": [item.patch_id for item in supports], "test_patch_ids": list(split["test"]), "unique_support_patches": len({item.patch_id for item in supports}), "threshold": model["threshold"], "query_adapted_test_patches": adapted, "test_patch_count": len(split["test"]), "metrics": result}
            rows.append(row)
            print(f"[{task}] feature={spec.name} polygons={polygon_count} test_f1={result['f1']:.4f} auc={result['auc']:.4f}", flush=True)
    payload = {"protocol": {"polygon_count": 3, "fold": args.fold, "shared_supports": True, "test_patch_count": len(split["test"])}, "rows": rows}
    (args.output_root / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    run_comparison(parse_args())


if __name__ == "__main__":
    main()
