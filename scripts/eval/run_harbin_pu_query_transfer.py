#!/usr/bin/env python3
"""Prepare the lock-driven Harbin polygon-prompt PU+Query transfer protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter, label
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_WEIGHT = 0.65
BACKGROUND_QUANTILE = 0.30
BACKGROUND_EXCLUSION_PIXELS = 3
MAX_BACKGROUND_PER_SUPPORT = 2048
QUERY_BLEND = 0.12
QUERY_QUANTILE = 0.997
QUERY_MIN_PIXELS, QUERY_MAX_PIXELS = 4, 128
QUERY_MIN_MARGIN = 0.05
QUERY_MAX_GROWTH, QUERY_MIN_AREA_CAP = 1.35, 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_label_id(patch_id: str, mapping: dict[str, str]) -> str:
    if patch_id not in mapping:
        raise KeyError(f"unlocked label mapping for {patch_id}")
    return mapping[patch_id]


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def aggregate_foreground_similarity(
    pixels: np.ndarray, prototypes: np.ndarray, mode: str
) -> np.ndarray:
    similarities = pixels @ prototypes.T
    if mode == "max":
        return similarities.max(axis=-1)
    if mode == "single":
        return similarities[:, 0]
    raise ValueError(f"unsupported prototype mode: {mode}")


def normalize_map(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return l2((np.moveaxis(feature, 0, -1) - mean) / np.maximum(std, 1e-5))


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(steps):
        padded = np.pad(result, 1)
        result = np.logical_or.reduce(
            [
                padded[y : y + result.shape[0], x : x + result.shape[1]]
                for y in range(3)
                for x in range(3)
            ]
        )
    return result


def select_validation_threshold(scores: Any, labels: Any) -> float:
    """Select a pooled F1 threshold from validation maps only."""
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    flat_labels = np.asarray(labels, dtype=bool).reshape(-1)
    if flat_scores.shape != flat_labels.shape or not flat_scores.size:
        raise ValueError("validation scores and labels must have one non-empty shared shape")
    order = np.argsort(flat_scores)[::-1]
    sorted_scores = flat_scores[order]
    sorted_labels = flat_labels[order].astype(np.int64)
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    last_in_score_group = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    true_positive = true_positive[last_in_score_group]
    false_positive = false_positive[last_in_score_group]
    candidates = sorted_scores[last_in_score_group]
    false_negative = int(sorted_labels.sum()) - true_positive
    f1 = 2.0 * true_positive / np.maximum(2.0 * true_positive + false_positive + false_negative, 1)
    best = f1 == f1.max()
    # Match the prior deterministic rule: on equal F1 select the lowest threshold.
    return float(candidates[best].min())


def score_pu_query(
    feature: np.ndarray,
    model: dict[str, Any],
    *,
    prototype_mode: str = "single",
    query_mode: str = "adaptive",
) -> tuple[np.ndarray, bool]:
    """Score one map without labels; Query gates only on support-derived quantities."""
    pixels = normalize_map(feature, model["mean"], model["std"])
    prototypes = np.asarray(model["foreground_prototypes"], dtype=np.float32)
    if prototype_mode == "single":
        foreground = pixels @ np.asarray(model["foreground"], dtype=np.float32)
    elif prototype_mode == "max":
        foreground = aggregate_foreground_similarity(pixels, prototypes, "max")
    else:
        raise ValueError(f"unsupported prototype mode: {prototype_mode}")
    background = np.asarray(model["background"], dtype=np.float32)
    base = gaussian_filter(foreground - BACKGROUND_WEIGHT * (pixels @ background), sigma=0.55)
    if query_mode == "disabled":
        return base.astype(np.float32), False
    if query_mode != "adaptive":
        raise ValueError(f"unsupported query mode: {query_mode}")
    support_threshold = float(model["support_threshold"])
    confidence = max(float(np.quantile(base, QUERY_QUANTILE)), support_threshold + QUERY_MIN_MARGIN)
    selected = base >= confidence
    selected_count = int(selected.sum())
    if not QUERY_MIN_PIXELS <= selected_count <= QUERY_MAX_PIXELS:
        return base.astype(np.float32), False
    query = l2(pixels[selected].mean(0, keepdims=True))[0]
    query_score = gaussian_filter(
        pixels @ query - BACKGROUND_WEIGHT * (pixels @ background), sigma=0.55
    )
    refined = (1.0 - QUERY_BLEND) * base + QUERY_BLEND * query_score
    base_area = int((base >= support_threshold).sum())
    refined_area = int((refined >= support_threshold).sum())
    if refined_area > max(QUERY_MIN_AREA_CAP, math.ceil(base_area * QUERY_MAX_GROWTH)):
        return base.astype(np.float32), False
    return refined.astype(np.float32), True


@dataclass(frozen=True)
class PreparedProtocol:
    patch_ids: list[str]
    label_ids: dict[str, str]
    schedules: dict[str, dict[str, Any]]
    matrix: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    output_root: Path | None = None

    def schedule_for(self, task: str, fold: int, seed: int, polygon_count: int) -> dict[str, Any]:
        key = f"{task}|{fold}|{seed}|{polygon_count}"
        if key not in self.schedules:
            raise KeyError(f"unlocked polygon schedule for {key}")
        return self.schedules[key]


def read_mask(label_root: Path, source_patch_id: str) -> np.ndarray:
    path = label_root / "masks" / f"{source_patch_id}.tif"
    with rasterio.open(path) as source:
        return source.read(1).astype(np.uint8) == 1


def polygon_candidates(
    label_root: Path,
    train_patch_ids: list[str],
    label_ids: dict[str, str],
    minimum_area: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for patch_id in sorted(train_patch_ids):
        source_patch_id = resolve_label_id(patch_id, label_ids)
        components, count = label(
            read_mask(label_root, source_patch_id), structure=np.ones((3, 3), dtype=np.uint8)
        )
        for component_index in range(1, count + 1):
            area = int((components == component_index).sum())
            if area >= minimum_area:
                candidates.append(
                    {
                        "patch_id": patch_id,
                        "source_patch_id": source_patch_id,
                        "component_index": component_index,
                        "area_pixels": area,
                    }
                )
    return candidates


def build_schedule(
    candidates: list[dict[str, Any]],
    *,
    task: str,
    fold: int,
    seed: int,
    polygon_counts: list[int],
) -> dict[str, dict[str, Any]]:
    if not candidates:
        raise ValueError(f"{task}/fold{fold} has no usable train polygons")
    order = np.random.default_rng(seed + fold * 1009 + sum(map(ord, task))).permutation(
        len(candidates)
    )
    ordered = [candidates[int(index)] for index in order]
    largest = max(polygon_counts)
    if len(ordered) < largest:
        raise ValueError(f"{task}/fold{fold} has {len(ordered)} polygons, below required {largest}")
    schedules: dict[str, dict[str, Any]] = {}
    for polygon_count in polygon_counts:
        selected = ordered[:polygon_count]
        schedules[f"{task}|{fold}|{seed}|{polygon_count}"] = {
            "task": task,
            "fold": fold,
            "seed": seed,
            "polygon_count": polygon_count,
            "candidate_count": len(ordered),
            "support_polygons": selected,
            "support_polygon_sha256": canonical_sha256(selected),
        }
    return schedules


def _locked_file(record: dict[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"PU+Query config lacks {key} lock")
    path = resolve_path(value["path"])
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"PU+Query config {key} lock does not match")
    return path


def prepare_harbin_pu_query(config_path: Path, output_root: Path) -> PreparedProtocol:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("coverage_patch_count") != 380:
        raise ValueError("Harbin PU+Query protocol requires exactly 380 coverage patches")
    if config.get("protocol_id") != "harbin_pu_query_transfer_20260729":
        raise ValueError("unexpected Harbin PU+Query protocol ID")
    matrix_path = _locked_file(config, "base_matrix")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("protocol_id") != "harbin_aef_locked_380_strict_conv3x3":
        raise ValueError("PU+Query base matrix is not the locked Harbin paired protocol")
    input_lock_path = _locked_file(config, "base_matrix_input_lock")
    input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
    if input_lock.get("matrix_sha256") != sha256_file(matrix_path):
        raise ValueError("base matrix input lock does not bind the selected matrix")
    if input_lock.get("coverage_patch_count") != 380 or input_lock.get("job_count") != 270:
        raise ValueError("base matrix input lock has an unexpected coverage or cell count")
    coverage_ref = matrix["coverage_inventory"]
    coverage_path = resolve_path(coverage_ref["path"])
    if sha256_file(coverage_path) != coverage_ref["sha256"]:
        raise ValueError("Harbin coverage inventory differs from base matrix lock")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    records = coverage.get("records")
    if not isinstance(records, list) or len(records) != 380:
        raise ValueError("Harbin coverage inventory must contain exactly 380 records")
    label_ids = {str(record["patch_id"]): str(record["source_patch_id"]) for record in records}
    patch_ids = sorted(label_ids)
    if len(patch_ids) != 380 or len(set(label_ids.values())) != 380:
        raise ValueError("Harbin coverage inventory has a non-bijective label mapping")
    split_ref = matrix["spatial_split"]
    split_path = resolve_path(split_ref["path"])
    if sha256_file(split_path) != split_ref["sha256"]:
        raise ValueError("Harbin spatial split differs from base matrix lock")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split.get("folds", [])) != 5:
        raise ValueError("Harbin PU+Query protocol requires five spatial folds")
    label_root = Path(matrix["label_root"])
    polygon_counts = list(config["polygon_counts"])
    if polygon_counts != [1, 3, 5, 9]:
        raise ValueError("polygon prompt budgets must be exactly [1, 3, 5, 9]")
    schedules: dict[str, dict[str, Any]] = {}
    schedule_dir = output_root / "frozen_polygon_schedules"
    for task in matrix["tasks"]:
        task_root = (
            label_root
            / {"building": "building_osm", "road": "road_osm", "water": "osm_water"}[task]
        )
        for fold_index in matrix["folds"]:
            fold = split["folds"][fold_index]
            universe = (
                set(fold["train"]) | set(fold["val"]) | set(fold["test"]) | set(fold["buffer"])
            )
            if universe != set(patch_ids):
                raise ValueError(f"fold {fold_index} does not partition the 380 locked patches")
            candidates = polygon_candidates(
                task_root,
                list(fold["train"]),
                label_ids,
                int(config["minimum_component_area"]),
            )
            for seed in matrix["seeds"]:
                schedules.update(
                    build_schedule(
                        candidates,
                        task=task,
                        fold=fold_index,
                        seed=seed,
                        polygon_counts=polygon_counts,
                    )
                )
                payload = {
                    "protocol_id": config["protocol_id"],
                    "base_matrix_sha256": sha256_file(matrix_path),
                    "coverage_inventory_sha256": sha256_file(coverage_path),
                    "spatial_split_sha256": sha256_file(split_path),
                    "task": task,
                    "fold": fold_index,
                    "seed": seed,
                    "sets": {
                        str(count): schedules[f"{task}|{fold_index}|{seed}|{count}"]
                        for count in polygon_counts
                    },
                }
                destination = schedule_dir / f"{task}_fold{fold_index}_seed{seed}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (
                    destination.exists()
                    and json.loads(destination.read_text(encoding="utf-8")) != payload
                ):
                    raise ValueError(f"existing frozen polygon schedule differs: {destination}")
                destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return PreparedProtocol(
        patch_ids=patch_ids,
        label_ids=label_ids,
        schedules=schedules,
        matrix=matrix,
        config=config,
        output_root=output_root,
    )


FEATURE_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def task_label_root(prepared: PreparedProtocol, task: str) -> Path:
    suffixes = {"building": "building_osm", "road": "road_osm", "water": "osm_water"}
    if task not in suffixes:
        raise KeyError(f"unsupported PU+Query task: {task}")
    return Path(prepared.matrix["label_root"]) / suffixes[task]


def family_spec(prepared: PreparedProtocol, family: str) -> dict[str, Any]:
    for spec in prepared.matrix["families"]:
        if spec["id"] == family:
            return spec
    raise KeyError(f"unlocked embedding family: {family}")


def load_embedding(family: dict[str, Any], patch_id: str) -> np.ndarray:
    root = Path(family["embedding_root"])
    month = str(family["month"])
    key = (str(root), month, patch_id)
    cached = FEATURE_CACHE.get(key)
    if cached is None:
        path = root / "harbin" / patch_id / f"{month}_embedding_map.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed embedding map: {path}")
        cached = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
        if cached.shape != (64, 128, 128) or not np.isfinite(cached).all():
            raise ValueError(f"invalid sealed embedding map: {path}")
        FEATURE_CACHE[key] = cached.astype(np.float32)
    return cached


def support_mask(prepared: PreparedProtocol, task: str, support: dict[str, Any]) -> np.ndarray:
    patch_id = str(support["patch_id"])
    source_patch_id = resolve_label_id(patch_id, prepared.label_ids)
    if source_patch_id != support.get("source_patch_id"):
        raise ValueError(f"support label mapping differs from frozen schedule: {patch_id}")
    components, count = label(
        read_mask(task_label_root(prepared, task), source_patch_id),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_index = int(support["component_index"])
    if component_index > count:
        raise ValueError(f"support component is absent from frozen label: {patch_id}")
    mask = components == component_index
    if int(mask.sum()) != int(support["area_pixels"]):
        raise ValueError(f"support component area differs from frozen schedule: {patch_id}")
    return mask


def support_threshold(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.unique(np.concatenate([positive, negative]))
    best_value, best_f1 = float(values[0]), -1.0
    for value in values:
        predicted_positive = positive >= value
        predicted_negative = negative >= value
        tp, fp, fn = (
            int(predicted_positive.sum()),
            int(predicted_negative.sum()),
            int((~predicted_positive).sum()),
        )
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_value, best_f1 = float(value), f1
    return best_value


def fit_pu_model(
    prepared: PreparedProtocol,
    family: str,
    task: str,
    supports: list[dict[str, Any]],
    prototype_mode: str,
    seed: int,
) -> dict[str, Any]:
    """Fit normalization, foreground and reliable background from prompt polygons only."""
    spec = family_spec(prepared, family)
    raw = {
        patch_id: load_embedding(spec, patch_id)
        for patch_id in sorted({item["patch_id"] for item in supports})
    }
    pixels = np.concatenate([value.reshape(64, -1).T for value in raw.values()])
    mean, std = pixels.mean(0), np.maximum(pixels.std(0), 1e-5)
    normalized = {patch_id: normalize_map(value, mean, std) for patch_id, value in raw.items()}
    masks = {id(item): support_mask(prepared, task, item) for item in supports}
    vectors = [
        l2(normalized[item["patch_id"]][masks[id(item)]].mean(0, keepdims=True))[0]
        for item in supports
    ]
    prototypes = np.stack(vectors).astype(np.float32)
    foreground = l2(np.mean(prototypes, axis=0, keepdims=True))[0]

    def foreground_score(values: np.ndarray) -> np.ndarray:
        if prototype_mode == "single":
            return values @ foreground
        return aggregate_foreground_similarity(values, prototypes, "max")

    union_masks = {patch_id: np.zeros((128, 128), dtype=bool) for patch_id in raw}
    for item in supports:
        union_masks[item["patch_id"]] |= masks[id(item)]
    rng = np.random.default_rng(seed + len(supports) * 17)
    backgrounds: list[np.ndarray] = []
    for patch_id, features in normalized.items():
        reliable = ~dilate(union_masks[patch_id], BACKGROUND_EXCLUSION_PIXELS)
        candidates = features[reliable]
        cutoff = np.quantile(foreground_score(candidates), BACKGROUND_QUANTILE)
        candidates = candidates[foreground_score(candidates) <= cutoff]
        if len(candidates):
            if len(candidates) > MAX_BACKGROUND_PER_SUPPORT:
                candidates = candidates[
                    rng.choice(len(candidates), MAX_BACKGROUND_PER_SUPPORT, replace=False)
                ]
            backgrounds.append(candidates)
    if not backgrounds:
        raise ValueError("prompt polygons did not yield reliable background samples")
    background = l2(np.concatenate(backgrounds).mean(0, keepdims=True))[0]
    positive = np.concatenate(
        [
            foreground_score(normalized[item["patch_id"]][masks[id(item)]])
            - BACKGROUND_WEIGHT * (normalized[item["patch_id"]][masks[id(item)]] @ background)
            for item in supports
        ]
    )
    negative = np.concatenate(
        [
            foreground_score(values) - BACKGROUND_WEIGHT * (values @ background)
            for values in backgrounds
        ]
    )
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "foreground": foreground.astype(np.float32),
        "foreground_prototypes": prototypes,
        "background": background.astype(np.float32),
        "support_threshold": support_threshold(positive, negative),
    }


def score_patch_ids(
    prepared: PreparedProtocol,
    family: str,
    patch_ids: list[str],
    model: dict[str, Any],
    prototype_mode: str,
    query_mode: str,
) -> tuple[np.ndarray, int]:
    """Return score maps without opening labels; this preserves test-label isolation."""
    spec = family_spec(prepared, family)
    scores: list[np.ndarray] = []
    adapted = 0
    for patch_id in patch_ids:
        score, did_adapt = score_pu_query(
            load_embedding(spec, patch_id),
            model,
            prototype_mode=prototype_mode,
            query_mode=query_mode,
        )
        scores.append(score)
        adapted += int(did_adapt)
    return np.stack(scores), adapted


def labels_for(prepared: PreparedProtocol, task: str, patch_ids: list[str]) -> np.ndarray:
    return np.stack(
        [
            read_mask(
                task_label_root(prepared, task), resolve_label_id(patch_id, prepared.label_ids)
            )
            for patch_id in patch_ids
        ]
    )


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    score = scores.reshape(-1).astype(np.float32)
    target = labels.reshape(-1).astype(np.uint8)
    predicted = score >= threshold
    tp = int(np.logical_and(predicted, target == 1).sum())
    fp = int(np.logical_and(predicted, target == 0).sum())
    fn = int(np.logical_and(~predicted, target == 1).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "ap": float(average_precision_score(target, score)),
        "auc_roc": float(roc_auc_score(target, score)),
        "precision": precision,
        "recall": recall,
    }


def evaluate_cell(
    prepared: PreparedProtocol,
    family: str,
    task: str,
    fold: int,
    seed: int,
    polygon_count: int,
    prototype_mode: str,
    query_mode: str,
) -> dict[str, Any]:
    """Evaluate one paired prompt cell with validation-only calibration and test-last labels."""
    schedule = prepared.schedule_for(task, fold, seed, polygon_count)
    split = json.loads(
        resolve_path(prepared.matrix["spatial_split"]["path"]).read_text(encoding="utf-8")
    )["folds"][fold]
    model = fit_pu_model(
        prepared,
        family,
        task,
        list(schedule["support_polygons"]),
        prototype_mode,
        seed,
    )
    validation_scores, validation_adapted = score_patch_ids(
        prepared, family, list(split["val"]), model, prototype_mode, query_mode
    )
    validation_labels = labels_for(prepared, task, list(split["val"]))
    threshold = select_validation_threshold(validation_scores, validation_labels)
    test_scores, test_adapted = score_patch_ids(
        prepared, family, list(split["test"]), model, prototype_mode, query_mode
    )
    test_labels = labels_for(prepared, task, list(split["test"]))
    return {
        "protocol_id": prepared.config["protocol_id"],
        "family": family,
        "task": task,
        "fold": fold,
        "seed": seed,
        "polygon_count": polygon_count,
        "prototype_mode": prototype_mode,
        "query_mode": query_mode,
        "support_schedule": schedule,
        "support_threshold": float(model["support_threshold"]),
        "validation_threshold": threshold,
        "validation_query_adapted_patches": validation_adapted,
        "test_query_adapted_patches": test_adapted,
        "validation_metrics": binary_metrics(validation_scores, validation_labels, threshold),
        "test_metrics": binary_metrics(test_scores, test_labels, threshold),
        "test_patch_ids": list(split["test"]),
    }


def cell_destination(output_root: Path, job: tuple[str, str, int, int, int, int, str, str]) -> Path:
    family, task, fold, seed, polygon_count, _index, prototype_mode, query_mode = job
    return (
        output_root
        / "results"
        / family
        / task
        / f"fold{fold}"
        / f"seed{seed}"
        / f"polygon{polygon_count}"
        / prototype_mode
        / query_mode
        / "result.json"
    )


def matrix_jobs(prepared: PreparedProtocol) -> list[tuple[str, str, int, int, int, int, str, str]]:
    return [
        (family["id"], task, fold, seed, polygon_count, index, prototype_mode, query_mode)
        for index, (
            family,
            task,
            fold,
            seed,
            polygon_count,
            prototype_mode,
            query_mode,
        ) in enumerate(
            (
                (family, task, fold, seed, polygon_count, prototype_mode, query_mode)
                for family in prepared.matrix["families"]
                for task in prepared.matrix["tasks"]
                for fold in prepared.matrix["folds"]
                for seed in prepared.matrix["seeds"]
                for polygon_count in prepared.config["polygon_counts"]
                for prototype_mode in prepared.config["prototype_modes"]
                for query_mode in prepared.config["query_modes"]
            )
        )
    ]


def write_protocol_lock(prepared: PreparedProtocol) -> Path:
    if prepared.output_root is None:
        raise ValueError("prepared protocol has no output root")
    lock_path = prepared.output_root / "protocol_input_lock.json"
    schedules = {
        str(path.relative_to(prepared.output_root)): sha256_file(path)
        for path in sorted((prepared.output_root / "frozen_polygon_schedules").glob("*.json"))
    }
    payload = {
        "protocol_id": prepared.config["protocol_id"],
        "config_sha256": canonical_sha256(prepared.config),
        "base_matrix_sha256": prepared.config["base_matrix"]["sha256"],
        "base_matrix_input_lock_sha256": prepared.config["base_matrix_input_lock"]["sha256"],
        "patch_count": len(prepared.patch_ids),
        "label_mapping_sha256": canonical_sha256(prepared.label_ids),
        "polygon_schedule_sha256": schedules,
        "expected_cells": len(matrix_jobs(prepared)),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != payload:
        raise ValueError("PU+Query input lock differs; refusing to mix protocol inputs")
    temporary = lock_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, lock_path)
    return lock_path


def run_job(
    prepared: PreparedProtocol,
    job: tuple[str, str, int, int, int, int, str, str],
    protocol_lock: Path,
) -> None:
    family, task, fold, seed, polygon_count, _index, prototype_mode, query_mode = job
    destination = cell_destination(prepared.output_root, job)  # type: ignore[arg-type]
    if destination.is_file():
        return
    result = evaluate_cell(
        prepared,
        family,
        task,
        fold,
        seed,
        polygon_count,
        prototype_mode,
        query_mode,
    )
    result["protocol_input_lock"] = str(protocol_lock.resolve())
    result["protocol_input_lock_sha256"] = sha256_file(protocol_lock)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(
        f"[done] {family} {task} fold={fold} seed={seed} polygons={polygon_count} "
        f"prototype={prototype_mode} query={query_mode}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_harbin_pu_query(args.config, args.output_root)
    print(
        "prepared "
        f"{len(prepared.patch_ids)} patches and {len(prepared.schedules)} polygon schedules",
        flush=True,
    )
    if not args.prepare_only:
        if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
            raise ValueError("worker index must be inside the positive worker count")
        protocol_lock = write_protocol_lock(prepared)
        for job in matrix_jobs(prepared):
            if job[5] % args.worker_count == args.worker_index:
                run_job(prepared, job, protocol_lock)


if __name__ == "__main__":
    main()
