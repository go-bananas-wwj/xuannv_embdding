#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_TASKS: dict[str, list[str]] = {
    "building": ["building_osm"],
    "road": ["road_osm"],
    "water": ["osm_water"],
    "park_green": ["osm_park", "osm_green", "osm_grass", "osm_garden"],
    "education": ["osm_education", "osm_school", "osm_university"],
    "sports_pitch": ["osm_sports", "osm_pitch", "osm_playground"],
    # 18 类地物评测新增（20260713）
    "river": ["osm_river"],
    "lake": ["osm_lake"],
    "pond": ["osm_pond"],
    "forest": ["osm_forest"],
    "grass": ["osm_grass"],
    "farmland": ["osm_agriculture"],
    "bare": ["osm_bare"],
    "parking": ["osm_parking"],
    "train_station": ["osm_train_station"],
    "stadium": ["osm_stadium"],
}

FEATURE_SETS = {
    "xuannv_embedding",
    "s2_indices",
    "s2_s1_landsat_indices",
    "s2_s1_landsat_highres_indices",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_roots: list[Path]
    split_root: Path


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    sources: dict[str, list[Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Xuannv embeddings against traditional remote-sensing ML "
            "features under the same sparse-label split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
        help="Root containing haidian/patch_xxx/<month>_embedding_map.pt.",
    )
    parser.add_argument("--label-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian/labels"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--feature-sets", nargs="+", default=["xuannv_embedding", "s2_indices", "s2_s1_landsat_indices"])
    parser.add_argument("--models", nargs="+", default=["rf", "extratrees", "logistic", "knn"])
    parser.add_argument("--shots", nargs="+", default=["5", "10", "50", "full"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--spatial-split",
        type=Path,
        default=Path("configs/eval/haidian_spatial_5fold_buffer1_seed42.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels-per-patch", type=int, default=2048)
    parser.add_argument(
        "--max-eval-pixels-per-patch",
        type=int,
        default=None,
        help=(
            "Optional validation/test sampling budget per patch. By default validation "
            "and test use all pixels. Use this for expensive traditional models."
        ),
    )
    parser.add_argument(
        "--eval-sampling-mode",
        choices=["uniform", "stratified"],
        default="uniform",
        help="Sampling mode for validation/test when --max-eval-pixels-per-patch is set.",
    )
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--predict-all", action="store_true")
    parser.add_argument("--smoke-patches", type=int, default=None)
    return parser.parse_args()


def month_of(path: Path) -> str | None:
    match = re.search(r"_(20\d{4})\d{2}_", path.name)
    if match:
        return match.group(1)
    match = re.search(r"_(20\d{4})_", path.name)
    if match:
        return match.group(1)
    return None


def load_manifest(path: Path, data_root: Path) -> dict[str, PatchRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, PatchRecord] = {}
    for entry in raw:
        patch_id = entry["patch_id"]
        sources: dict[str, list[Path]] = {}
        for key, value in entry.items():
            if key == "patch_id" or not isinstance(value, list):
                continue
            paths = [data_root / rel for rel in value]
            sources[key] = paths
        records[patch_id] = PatchRecord(patch_id=patch_id, sources=sources)
    return records


def resolve_mask(mask_dir: Path, patch_id: str) -> Path:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"*{patch_id}.tif")) + sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No mask found for {patch_id} in {mask_dir}")
    return candidates[-1]


def load_binary_mask(task: TaskSpec, patch_id: str) -> np.ndarray:
    mask: np.ndarray | None = None
    for root in task.label_roots:
        path = resolve_mask(root / "masks", patch_id)
        with rasterio.open(path) as src:
            arr = src.read(1)
        arr = (arr == 1).astype(np.uint8)
        mask = arr if mask is None else np.maximum(mask, arr)
    if mask is None:
        raise RuntimeError(f"Empty task roots for {task.name}")
    return mask.astype(np.int16)


def load_split(
    task: TaskSpec,
    fold: int,
    split_path: Path | None = None,
) -> dict[str, list[str]]:
    split_path = split_path or (task.split_root / "split_5fold.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold_info = split["folds"][fold]
    return {
        "train": list(fold_info["train"]),
        "val": list(fold_info["val"]),
        "test": list(fold_info["test"]),
    }


def positive_pixel_count(task: TaskSpec, patch_id: str) -> int:
    return int((load_binary_mask(task, patch_id) == 1).sum())


def select_train_patch_ids(
    task: TaskSpec,
    train_ids: list[str],
    shot: str,
    seed: int,
    fold: int,
) -> list[str]:
    if shot == "full":
        return list(train_ids)
    budget = int(shot)
    positives: list[str] = []
    negatives: list[str] = []
    for patch_id in train_ids:
        positive_pixels = positive_pixel_count(task, patch_id)
        if positive_pixels >= 64:
            positives.append(patch_id)
        elif positive_pixels == 0:
            negatives.append(patch_id)
    rng = random.Random(seed + fold * 1009)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < budget or len(negatives) < budget:
        raise RuntimeError(
            f"Exact {budget}+{budget} shot budget infeasible for {task.name}: "
            f"positive={len(positives)} negative={len(negatives)}"
        )
    selected_pos = positives[:budget]
    selected_neg = negatives[:budget]
    selected = selected_pos + selected_neg
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError(f"No train patches selected for {task.name} shot={shot}")
    return selected


def read_raster_mean(
    paths: list[Path],
    month: str,
    out_shape: tuple[int, int] = (128, 128),
) -> np.ndarray | None:
    selected = [p for p in paths if month_of(p) == month and p.exists() and not p.name.endswith("_mask.tif")]
    if not selected:
        return None
    arrays: list[np.ndarray] = []
    for path in selected:
        with rasterio.open(path) as src:
            arr = src.read(
                out_shape=(
                    src.count,
                    out_shape[0],
                    out_shape[1],
                ),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
        arr[~np.isfinite(arr)] = np.nan
        arrays.append(arr)
    stacked = np.stack(arrays, axis=0)
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    summed = np.where(valid, stacked, 0.0).sum(axis=0)
    mean = np.divide(summed, np.maximum(count, 1), out=np.zeros_like(summed), where=count > 0)
    mean[~np.isfinite(mean)] = 0.0
    return mean


def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def add_optical_indices(arr: np.ndarray, kind: str) -> list[np.ndarray]:
    channels = [arr]
    if kind == "s2" and arr.shape[0] >= 12:
        blue = arr[1]
        green = arr[2]
        red = arr[3]
        nir = arr[7]
        swir1 = arr[10]
        swir2 = arr[11]
    elif kind == "landsat" and arr.shape[0] >= 7:
        blue = arr[1]
        green = arr[2]
        red = arr[3]
        nir = arr[4]
        swir1 = arr[5]
        swir2 = arr[6]
    else:
        return channels
    ndvi = safe_index(nir, red)[None, :, :]
    ndwi = safe_index(green, nir)[None, :, :]
    ndbi = safe_index(swir1, nir)[None, :, :]
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-6)
    mndwi = safe_index(green, swir1)[None, :, :]
    channels.extend([ndvi, ndwi, ndbi, bsi[None, :, :], mndwi])
    return channels


def local_stats(arr: np.ndarray) -> list[np.ndarray]:
    # Lightweight texture proxy: per-pixel channel mean/std and gradient magnitude.
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    gy, gx = np.gradient(mean[0])
    grad = np.sqrt(gx * gx + gy * gy)[None, :, :]
    return [mean.astype(np.float32), std.astype(np.float32), grad.astype(np.float32)]


def load_feature_map(
    feature_set: str,
    record: PatchRecord,
    embedding_root: Path,
    region: str,
    month: str,
) -> np.ndarray:
    if feature_set == "xuannv_embedding":
        path = embedding_root / region / record.patch_id / f"{month}_embedding_map.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding map: {path}")
        emb = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
        if emb.ndim != 3:
            raise ValueError(f"Expected C,H,W embedding, got {emb.shape}: {path}")
        return emb.astype(np.float32)

    parts: list[np.ndarray] = []
    s2 = read_raster_mean(record.sources.get("s2", []), month)
    if s2 is not None:
        parts.extend(add_optical_indices(s2, "s2"))
    if feature_set in {"s2_s1_landsat_indices", "s2_s1_landsat_highres_indices"}:
        s1 = read_raster_mean(record.sources.get("s1", []), month)
        if s1 is not None:
            parts.append(s1)
            if s1.shape[0] >= 2:
                parts.append(safe_index(s1[0], s1[1])[None, :, :])
        landsat = read_raster_mean(record.sources.get("landsat", []), month)
        if landsat is not None:
            parts.extend(add_optical_indices(landsat, "landsat"))
    if feature_set == "s2_s1_landsat_highres_indices":
        highres_opt = read_raster_mean(record.sources.get("highres_optical_haidian", []), month)
        if highres_opt is not None:
            parts.append(highres_opt)
            parts.extend(local_stats(highres_opt))
        highres_sar = read_raster_mean(record.sources.get("highres_sar_haidian", []), month)
        if highres_sar is not None:
            parts.append(highres_sar)
            parts.extend(local_stats(highres_sar))
    if not parts:
        raise RuntimeError(f"No features available for {record.patch_id} feature_set={feature_set} month={month}")
    out = np.concatenate(parts, axis=0).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def sample_pixels_from_patch(
    features: np.ndarray,
    mask: np.ndarray,
    max_pixels: int,
    positive_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = features.transpose(1, 2, 0).reshape(-1, features.shape[0])
    y = mask.reshape(-1).astype(np.uint8)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        total = min(max_pixels, len(y))
        idx = rng.choice(len(y), size=total, replace=False)
        return x[idx], y[idx]
    n_pos = min(len(pos), max(1, int(max_pixels * positive_fraction)))
    n_neg = min(len(neg), max_pixels - n_pos)
    pos_idx = rng.choice(pos, size=n_pos, replace=False)
    neg_idx = rng.choice(neg, size=max(1, n_neg), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return x[idx], y[idx]


def sample_uniform_pixels_from_patch(
    features: np.ndarray,
    mask: np.ndarray,
    max_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = features.transpose(1, 2, 0).reshape(-1, features.shape[0])
    y = mask.reshape(-1).astype(np.uint8)
    total = min(max_pixels, len(y))
    idx = rng.choice(len(y), size=total, replace=False)
    return x[idx], y[idx]


def build_dataset(
    records: dict[str, PatchRecord],
    task: TaskSpec,
    patch_ids: list[str],
    feature_set: str,
    embedding_root: Path,
    region: str,
    month: str,
    max_pixels_per_patch: int | None,
    positive_fraction: float,
    seed: int,
    sampling_mode: str = "stratified",
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for patch_id in patch_ids:
        fmap = load_feature_map(feature_set, records[patch_id], embedding_root, region, month)
        mask = load_binary_mask(task, patch_id)
        if max_pixels_per_patch is None:
            x = fmap.transpose(1, 2, 0).reshape(-1, fmap.shape[0])
            y = mask.reshape(-1).astype(np.uint8)
        else:
            if sampling_mode == "uniform":
                x, y = sample_uniform_pixels_from_patch(fmap, mask, max_pixels_per_patch, rng)
            else:
                x, y = sample_pixels_from_patch(fmap, mask, max_pixels_per_patch, positive_fraction, rng)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def make_model(name: str, seed: int) -> Any:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
    if name in {"hgb", "histgb"}:
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=220,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1e-3,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="saga",
                random_state=seed,
            ),
        )
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1))
    if name == "centroid":
        return make_pipeline(StandardScaler(), NearestCentroid())
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xgboost is not installed") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
        )
    raise KeyError(f"Unknown model: {name}")


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].astype(np.float32)
        return prob.reshape(-1).astype(np.float32)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)
    pred = model.predict(x)
    return pred.astype(np.float32)


def compute_metrics(prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    target = (target == 1).astype(np.uint8)
    pred = (prob >= threshold).astype(np.uint8)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    out: dict[str, float | int] = {
        "f1_at_threshold": float(f1),
        "f1_0.5": float(binary_f1(prob, target, 0.5)),
        "miou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_ratio": float(target.mean()) if target.size else 0.0,
    }
    if target.sum() == 0 or target.sum() == len(target):
        out.update({"f1_best": 0.0, "best_threshold": 0.5, "ap": 0.0, "auprc": 0.0, "auc_roc": 0.0})
        return out
    p_arr, r_arr, thresholds = precision_recall_curve(target, prob)
    f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
    best_idx = int(f1s.argmax())
    best_threshold = 0.5
    if thresholds.size:
        best_threshold = float(thresholds[min(best_idx, thresholds.size - 1)])
    out.update(
        {
            "f1_best": float(f1s[best_idx]),
            "best_threshold": best_threshold,
            "ap": float(average_precision_score(target, prob)),
            "auprc": float(auc(r_arr, p_arr)),
            "auc_roc": float(roc_auc_score(target, prob)),
        }
    )
    return out


def binary_f1(prob: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def write_prediction_maps(
    model: Any,
    records: dict[str, PatchRecord],
    task: TaskSpec,
    patch_ids: list[str],
    feature_set: str,
    embedding_root: Path,
    region: str,
    month: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for patch_id in patch_ids:
        fmap = load_feature_map(feature_set, records[patch_id], embedding_root, region, month)
        x = fmap.transpose(1, 2, 0).reshape(-1, fmap.shape[0])
        prob = predict_prob(model, x).reshape(fmap.shape[1], fmap.shape[2]).astype(np.float32)
        ref_path = resolve_mask(task.label_roots[0] / "masks", patch_id)
        with rasterio.open(ref_path) as src:
            profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=None, compress="lzw")
        with rasterio.open(out_dir / f"{patch_id}_prob.tif", "w", **profile) as dst:
            dst.write(prob, 1)


def task_spec(name: str, label_root: Path) -> TaskSpec:
    if name not in DEFAULT_TASKS:
        raise KeyError(f"Unknown task {name}. Available: {sorted(DEFAULT_TASKS)}")
    roots = [label_root / item for item in DEFAULT_TASKS[name]]
    for root in roots:
        if not (root / "masks").exists():
            raise FileNotFoundError(f"Missing masks for task {name}: {root / 'masks'}")
    split_root = roots[0]
    if not (split_root / "split_5fold.json").exists():
        fallback = label_root / "osm_water"
        if not (fallback / "split_5fold.json").exists():
            raise FileNotFoundError(f"No split for task {name}: {split_root}")
        split_root = fallback
    return TaskSpec(name=name, label_roots=roots, split_root=split_root)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for feature_set in args.feature_sets:
        if feature_set not in FEATURE_SETS:
            raise KeyError(f"Unknown feature set {feature_set}. Available: {sorted(FEATURE_SETS)}")
    records = load_manifest(args.manifest, args.data_root)
    all_metrics: list[dict[str, Any]] = []

    for task_name in args.tasks:
        task = task_spec(task_name, args.label_root)
        split = load_split(task, args.fold, args.spatial_split)
        if args.smoke_patches is not None:
            for key in split:
                split[key] = split[key][: args.smoke_patches]
        for shot in args.shots:
            train_ids = select_train_patch_ids(task, split["train"], shot, args.seed, args.fold)
            for feature_set in args.feature_sets:
                train_x, train_y = build_dataset(
                    records,
                    task,
                    train_ids,
                    feature_set,
                    args.embedding_root,
                    args.region,
                    args.month,
                    args.max_pixels_per_patch,
                    args.positive_fraction,
                    args.seed,
                    "stratified",
                )
                val_x, val_y = build_dataset(
                    records,
                    task,
                    split["val"],
                    feature_set,
                    args.embedding_root,
                    args.region,
                    args.month,
                    args.max_eval_pixels_per_patch,
                    args.positive_fraction,
                    args.seed + 17,
                    args.eval_sampling_mode,
                )
                test_x, test_y = build_dataset(
                    records,
                    task,
                    split["test"],
                    feature_set,
                    args.embedding_root,
                    args.region,
                    args.month,
                    args.max_eval_pixels_per_patch,
                    args.positive_fraction,
                    args.seed + 31,
                    args.eval_sampling_mode,
                )
                for model_name in args.models:
                    out_dir = args.output_root / task_name / feature_set / model_name / f"shot_{shot}" / f"fold_{args.fold}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        model = make_model(model_name, args.seed)
                        model.fit(train_x, train_y)
                    except Exception as exc:
                        record = {
                            "task": task_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "shot": shot,
                            "fold": args.fold,
                            "status": "failed",
                            "paper_eligible": False,
                            "protocol_status": "diagnostic_until_all_registered_evaluation_gates_pass",
                            "error": repr(exc),
                        }
                        (out_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                        all_metrics.append(record)
                        continue
                    val_prob = predict_prob(model, val_x)
                    val_metrics = compute_metrics(val_prob, val_y)
                    threshold = float(val_metrics["best_threshold"])
                    test_prob = predict_prob(model, test_x)
                    test_metrics = compute_metrics(test_prob, test_y, threshold=threshold)
                    record = {
                        **test_metrics,
                        "task": task_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "shot": shot,
                        "fold": args.fold,
                        "status": "ok",
                        "paper_eligible": False,
                        "protocol_status": "diagnostic_until_all_registered_evaluation_gates_pass",
                        "val_f1_best": val_metrics["f1_best"],
                        "val_ap": val_metrics["ap"],
                        "val_auc_roc": val_metrics["auc_roc"],
                        "val_threshold": threshold,
                        "train_patch_count": len(train_ids),
                        "train_pixels": int(train_y.shape[0]),
                        "val_pixels": int(val_y.shape[0]),
                        "test_pixels": int(test_y.shape[0]),
                    }
                    (out_dir / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    (out_dir / "train_selection.json").write_text(
                        json.dumps({"patch_ids": train_ids}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if args.predict_all:
                        write_prediction_maps(
                            model,
                            records,
                            task,
                            sorted(records),
                            feature_set,
                            args.embedding_root,
                            args.region,
                            args.month,
                            out_dir / "predictions_all",
                        )
                    all_metrics.append(record)
                    print(
                        f"{task_name} {feature_set} {model_name} shot={shot} "
                        f"f1={record['f1_best']:.4f} ap={record['ap']:.4f} auc={record['auc_roc']:.4f}",
                        flush=True,
                    )

    summary_path = args.output_root / "all_metrics.json"
    summary_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output_root / "all_metrics.csv"
    keys = sorted({key for row in all_metrics for key in row})
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in all_metrics:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = ""
                values.append(str(value).replace(",", ";"))
            f.write(",".join(values) + "\n")
    meta = {
        "paper_eligible": False,
        "protocol_status": "diagnostic_until_all_registered_evaluation_gates_pass",
        "manifest": str(args.manifest),
        "embedding_root": str(args.embedding_root),
        "month": args.month,
        "tasks": args.tasks,
        "feature_sets": args.feature_sets,
        "models": args.models,
        "shots": args.shots,
        "fold": args.fold,
        "spatial_split": str(args.spatial_split),
        "spatial_split_sha256": hashlib.sha256(args.spatial_split.read_bytes()).hexdigest(),
        "max_pixels_per_patch": args.max_pixels_per_patch,
        "max_eval_pixels_per_patch": args.max_eval_pixels_per_patch,
        "eval_sampling_mode": args.eval_sampling_mode,
    }
    (args.output_root / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
