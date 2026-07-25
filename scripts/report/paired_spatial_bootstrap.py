#!/usr/bin/env python3
"""Paired spatial-block bootstrap for sealed downstream-probe artifacts.

This utility compares two encoder families only after pairing results on task,
labelled-patch budget, spatial fold, and probe seed.  Within every paired
evaluation unit it resamples complete test patches, never individual pixels.
The resulting intervals therefore preserve the spatial block as the smallest
resampling unit.  It intentionally reports only metrics reconstructable from
the recorded patch-level confusion matrices; AP and ROC-AUC require score-map
resampling and remain separate point-estimate metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.report import aggregate_registered_paper_results as aggregate

CONFUSION_KEYS = ("tp", "fp", "fn", "tn")
METRICS = ("f1", "miou", "precision", "recall")

PatchConfusion = Mapping[str, Mapping[str, int | float]]
GroupKey = tuple[int, int]


def _empty_confusion() -> dict[str, int]:
    return {key: 0 for key in CONFUSION_KEYS}


def _add_confusion(total: dict[str, int], item: Mapping[str, int | float]) -> None:
    for key in CONFUSION_KEYS:
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Patch confusion must contain a non-negative integer {key}")
        total[key] += value


def _aggregate(items: list[Mapping[str, int | float]]) -> dict[str, int]:
    total = _empty_confusion()
    for item in items:
        _add_confusion(total, item)
    return total


def metrics_from_confusion(confusion: Mapping[str, int | float]) -> dict[str, float]:
    """Return binary metrics from aggregate patch-level confusion counts."""
    tp = float(confusion["tp"])
    fp = float(confusion["fp"])
    fn = float(confusion["fn"])
    f1_denom = 2.0 * tp + fp + fn
    iou_denom = tp + fp + fn
    precision_denom = tp + fp
    recall_denom = tp + fn
    return {
        "f1": 0.0 if f1_denom == 0.0 else 2.0 * tp / f1_denom,
        "miou": 0.0 if iou_denom == 0.0 else tp / iou_denom,
        "precision": 0.0 if precision_denom == 0.0 else tp / precision_denom,
        "recall": 0.0 if recall_denom == 0.0 else tp / recall_denom,
    }


def _summarize_differences(
    point: dict[str, float], samples: dict[str, np.ndarray]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = samples[metric]
        summary[metric] = {
            "point_difference": point[metric],
            "bootstrap_mean_difference": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "probability_candidate_better": float(np.mean(values > 0.0)),
        }
    return summary


def _validate_patch_pair(baseline: PatchConfusion, candidate: PatchConfusion) -> list[str]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        raise ValueError("Paired bootstrap requires identical patch IDs")
    if not baseline_ids:
        raise ValueError("Paired bootstrap requires at least one test patch")
    return sorted(baseline_ids)


def bootstrap_group_difference(
    baseline: PatchConfusion,
    candidate: PatchConfusion,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap one fold/seed pair by resampling whole test patches."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    patch_ids = _validate_patch_pair(baseline, candidate)
    baseline_values = [baseline[patch_id] for patch_id in patch_ids]
    candidate_values = [candidate[patch_id] for patch_id in patch_ids]
    point_a = metrics_from_confusion(_aggregate(baseline_values))
    point_b = metrics_from_confusion(_aggregate(candidate_values))
    point = {metric: point_b[metric] - point_a[metric] for metric in METRICS}

    generator = np.random.default_rng(seed)
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in METRICS}
    for index in range(n_resamples):
        draw = generator.integers(0, len(patch_ids), size=len(patch_ids))
        metrics_a = metrics_from_confusion(_aggregate([baseline_values[item] for item in draw]))
        metrics_b = metrics_from_confusion(_aggregate([candidate_values[item] for item in draw]))
        for metric in METRICS:
            samples[metric][index] = metrics_b[metric] - metrics_a[metric]
    return {
        "n_patches": len(patch_ids),
        "candidate_minus_baseline": _summarize_differences(point, samples),
    }


def hierarchical_paired_bootstrap(
    baseline: Mapping[GroupKey, PatchConfusion],
    candidate: Mapping[GroupKey, PatchConfusion],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap matched fold/seed groups and their complete test patches."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if set(baseline) != set(candidate):
        raise ValueError("Paired bootstrap requires identical evaluation groups")
    if not baseline:
        raise ValueError("Paired bootstrap requires at least one evaluation group")
    group_keys = sorted(baseline)
    patch_ids = {
        group: _validate_patch_pair(baseline[group], candidate[group]) for group in group_keys
    }
    folds = sorted({fold for fold, _ in group_keys})
    seeds_by_fold = {
        fold: tuple(sorted(seed for grouped_fold, seed in group_keys if grouped_fold == fold))
        for fold in folds
    }
    first_seed_set = seeds_by_fold[folds[0]]
    if not first_seed_set or any(seeds != first_seed_set for seeds in seeds_by_fold.values()):
        raise ValueError("Each spatial fold must contain the same non-empty probe-seed set")
    patch_ids_by_fold: dict[int, list[str]] = {}
    for fold in folds:
        reference = patch_ids[(fold, first_seed_set[0])]
        if any(patch_ids[(fold, seed)] != reference for seed in seeds_by_fold[fold]):
            raise ValueError("All probe seeds in one spatial fold must share identical patch IDs")
        patch_ids_by_fold[fold] = reference
    point_a = _empty_confusion()
    point_b = _empty_confusion()
    for group in group_keys:
        _add_confusion(point_a, _aggregate([baseline[group][patch] for patch in patch_ids[group]]))
        _add_confusion(point_b, _aggregate([candidate[group][patch] for patch in patch_ids[group]]))
    point_metrics_a = metrics_from_confusion(point_a)
    point_metrics_b = metrics_from_confusion(point_b)
    point = {metric: point_metrics_b[metric] - point_metrics_a[metric] for metric in METRICS}

    generator = np.random.default_rng(seed)
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in METRICS}
    for index in range(n_resamples):
        sampled_folds = generator.integers(0, len(folds), size=len(folds))
        total_a = _empty_confusion()
        total_b = _empty_confusion()
        for fold_index in sampled_folds:
            fold = folds[int(fold_index)]
            ids = patch_ids_by_fold[fold]
            draw = generator.integers(0, len(ids), size=len(ids))
            for seed_key in seeds_by_fold[fold]:
                group = (fold, seed_key)
                _add_confusion(total_a, _aggregate([baseline[group][ids[item]] for item in draw]))
                _add_confusion(total_b, _aggregate([candidate[group][ids[item]] for item in draw]))
        metrics_a = metrics_from_confusion(total_a)
        metrics_b = metrics_from_confusion(total_b)
        for metric in METRICS:
            samples[metric][index] = metrics_b[metric] - metrics_a[metric]
    return {
        "n_spatial_folds": len(folds),
        "probe_seeds_per_fold": list(first_seed_set),
        "n_patches_per_fold": {f"fold{fold}": len(patch_ids_by_fold[fold]) for fold in folds},
        "candidate_minus_baseline": _summarize_differences(point, samples),
    }


def _records_by_cell(
    records: list[dict[str, Any]], *, tasks: tuple[str, ...], shots: tuple[str, ...]
) -> dict[tuple[str, str, int, int], PatchConfusion]:
    indexed: dict[tuple[str, str, int, int], PatchConfusion] = {}
    for record in records:
        payload = aggregate._metric_payload(record)
        key = (
            str(payload["task"]),
            str(payload["shot"]),
            int(payload["fold"]),
            int(payload["shot_seed"]),
        )
        if key[0] not in tasks or key[1] not in shots:
            continue
        confusion = payload.get("per_patch_confusion")
        if not isinstance(confusion, dict):
            raise ValueError(f"Result lacks per_patch_confusion: {record['result_id']}")
        if key in indexed:
            raise ValueError(f"Duplicate result cell: {key}")
        indexed[key] = confusion
    return indexed


def compare_families(
    *,
    registry_path: Path,
    baseline_family: str,
    candidate_family: str,
    tasks: tuple[str, ...],
    shots: tuple[str, ...],
    allow_preliminary: bool,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Load sealed records and compute one paired hierarchical interval per task/shot."""
    baseline_records = aggregate.load_verified_records(
        registry_path, family=baseline_family, allow_preliminary=allow_preliminary
    )
    candidate_records = aggregate.load_verified_records(
        registry_path, family=candidate_family, allow_preliminary=allow_preliminary
    )
    baseline = _records_by_cell(baseline_records, tasks=tasks, shots=shots)
    candidate = _records_by_cell(candidate_records, tasks=tasks, shots=shots)
    expected = {
        (task, shot, fold, probe_seed)
        for task in tasks
        for shot in shots
        for fold in (0, 1, 2, 3, 4)
        for probe_seed in (42, 43, 44)
    }
    if set(baseline) != expected or set(candidate) != expected:
        raise ValueError("Each family must contain the complete registered 5-fold x 3-seed matrix")
    comparisons: dict[str, Any] = {}
    for task in tasks:
        for shot in shots:
            groups_a = {
                (fold, probe_seed): baseline[(task, shot, fold, probe_seed)]
                for fold in range(5)
                for probe_seed in (42, 43, 44)
            }
            groups_b = {
                (fold, probe_seed): candidate[(task, shot, fold, probe_seed)]
                for fold in range(5)
                for probe_seed in (42, 43, 44)
            }
            comparisons[f"{task}|{shot}"] = hierarchical_paired_bootstrap(
                groups_a,
                groups_b,
                n_resamples=n_resamples,
                seed=seed + len(comparisons),
            )
    return {
        "schema_version": 1,
        "method": "paired hierarchical spatial-block bootstrap",
        "comparison_direction": "candidate_minus_baseline",
        "baseline_family": baseline_family,
        "candidate_family": candidate_family,
        "preliminary": allow_preliminary,
        "paper_eligible": not allow_preliminary,
        "admission_status": (
            "registered_preliminary_pending_external_gates"
            if allow_preliminary
            else "registered_paper_eligible"
        ),
        "n_resamples": n_resamples,
        "random_seed": seed,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--baseline-family", required=True)
    parser.add_argument("--candidate-family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=("building", "road", "water"))
    parser.add_argument("--shots", nargs="+", default=("5", "10"))
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--allow-preliminary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_families(
        registry_path=args.registry,
        baseline_family=args.baseline_family,
        candidate_family=args.candidate_family,
        tasks=tuple(args.tasks),
        shots=tuple(args.shots),
        allow_preliminary=args.allow_preliminary,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
