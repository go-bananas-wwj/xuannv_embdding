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
PROTOCOL_N_RESAMPLES = 10000
PROTOCOL_RANDOM_SEED = 20260725
PAPER_PATCH_METADATA = Path(__file__).resolve().parents[2] / "configs/regions/haidian_patches.json"
PAPER_PATCH_METADATA_SHA256 = "17137cc379a874949a3eb460d48d8ad7c7698b63dbaaf5b4d3c1b40f84f69d97"

PatchConfusion = Mapping[str, Mapping[str, int | float]]
GroupKey = tuple[int, int]


def validate_paper_bootstrap_output_path(
    output_path: Path, *, baseline_family: str, candidate_family: str
) -> None:
    """Reserve one committed output path for the formal V5 paired bootstrap."""
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        repo_root
        / "configs"
        / "eval"
        / "release_admissions"
        / f"rse_v5_osm_assisted_{baseline_family}_vs_{candidate_family}_paired_bootstrap.json"
    )
    if output_path.resolve() != expected.resolve():
        raise ValueError(f"Paired bootstrap must use canonical paired-bootstrap path: {expected}")


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


def _sample_standard_deviation(values: list[float]) -> float:
    """Return a descriptive sample standard deviation, including singleton support."""
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _validate_patch_pair(baseline: PatchConfusion, candidate: PatchConfusion) -> list[str]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        raise ValueError("Paired bootstrap requires identical patch IDs")
    if not baseline_ids:
        raise ValueError("Paired bootstrap requires at least one test patch")
    return sorted(baseline_ids)


def build_two_by_two_geographic_clusters(
    bounds_by_patch: Mapping[str, tuple[float, float, float, float]],
    patch_ids: list[str],
) -> list[tuple[str, ...]]:
    """Group a regular patch grid into deterministic 2 x 2 geographic blocks."""
    if not patch_ids:
        raise ValueError("Cannot build geographic clusters without patch IDs")
    missing = sorted(set(patch_ids) - set(bounds_by_patch))
    if missing:
        raise ValueError(f"Patch metadata is missing bounds for: {', '.join(missing[:3])}")
    all_bounds = list(bounds_by_patch.values())
    if not np.all(np.isfinite(np.asarray(all_bounds, dtype=np.float64))):
        raise ValueError("Geographic patch bounds must be finite")
    widths = np.array([bounds[2] - bounds[0] for bounds in all_bounds], dtype=np.float64)
    heights = np.array([bounds[3] - bounds[1] for bounds in all_bounds], dtype=np.float64)
    if np.any(widths <= 0.0) or np.any(heights <= 0.0):
        raise ValueError("Geographic patch bounds must have positive width and height")
    width = float(np.median(widths))
    height = float(np.median(heights))
    if not np.allclose(widths, width) or not np.allclose(heights, height):
        raise ValueError("2 x 2 geographic clusters require a regular patch grid")
    # Use the full region's lattice origin.  A held-out fold may omit patches
    # along an outer edge, and re-anchoring blocks on that subset would make
    # the definition of a "2 x 2" block vary across folds.
    origin_x = min(bounds[0] for bounds in all_bounds)
    origin_y = min(bounds[1] for bounds in all_bounds)
    grouped: dict[tuple[int, int], list[str]] = {}
    occupied_cells: set[tuple[int, int]] = set()
    for patch_id in sorted(patch_ids):
        min_x, min_y, _, _ = bounds_by_patch[patch_id]
        scaled_col = (min_x - origin_x) / width
        scaled_row = (min_y - origin_y) / height
        col = int(round(scaled_col))
        row = int(round(scaled_row))
        if not np.isclose(scaled_col, col) or not np.isclose(scaled_row, row):
            raise ValueError("Geographic patch bounds must lie on one regular patch lattice")
        if (row, col) in occupied_cells:
            raise ValueError("Geographic patch bounds must not overlap on one lattice cell")
        occupied_cells.add((row, col))
        grouped.setdefault((row // 2, col // 2), []).append(patch_id)
    clusters = [tuple(sorted(grouped[key])) for key in sorted(grouped)]
    incomplete = [cluster for cluster in clusters if len(cluster) != 4]
    if incomplete:
        raise ValueError(
            "Each test fold must be a union of complete 2 x 2 geographic patch blocks; "
            "regenerate the registered spatial split before inference"
        )
    return clusters


def load_patch_bounds(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Load the projected patch bounds needed for geographic cluster resampling."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Patch metadata must be a list of patch records")
    bounds_by_patch: dict[str, tuple[float, float, float, float]] = {}
    for record in payload:
        if not isinstance(record, dict):
            raise ValueError("Patch metadata records must be objects")
        patch_id = record.get("patch_id")
        bounds = record.get("bounds")
        if not isinstance(patch_id, str) or not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError("Patch metadata records require patch_id and four projected bounds")
        if patch_id in bounds_by_patch:
            raise ValueError(f"Duplicate patch metadata ID: {patch_id}")
        bounds_by_patch[patch_id] = tuple(float(value) for value in bounds)
    return bounds_by_patch


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


def _hierarchical_paired_resample(
    baseline: Mapping[GroupKey, PatchConfusion],
    candidate: Mapping[GroupKey, PatchConfusion],
    *,
    n_resamples: int,
    seed: int,
    clusters_by_fold: Mapping[int, list[tuple[str, ...]]],
) -> dict[str, Any]:
    """Resample an already validated hierarchy; internal testable implementation."""
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
    resolved_clusters: dict[int, list[tuple[str, ...]]] = {}
    for fold in folds:
        reference = patch_ids[(fold, first_seed_set[0])]
        if any(patch_ids[(fold, seed)] != reference for seed in seeds_by_fold[fold]):
            raise ValueError("All probe seeds in one spatial fold must share identical patch IDs")
        patch_ids_by_fold[fold] = reference
        clusters = [tuple(cluster) for cluster in clusters_by_fold.get(fold, [])]
        if any(len(cluster) != 4 for cluster in clusters):
            raise ValueError("Geographic bootstrap clusters must each contain exactly four patches")
        flattened = [patch_id for cluster in clusters for patch_id in cluster]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(reference):
            raise ValueError("Geographic clusters must partition each fold's patch support")
        resolved_clusters[fold] = clusters
    point_a = _empty_confusion()
    point_b = _empty_confusion()
    for group in group_keys:
        _add_confusion(point_a, _aggregate([baseline[group][patch] for patch in patch_ids[group]]))
        _add_confusion(point_b, _aggregate([candidate[group][patch] for patch in patch_ids[group]]))
    point_metrics_a = metrics_from_confusion(point_a)
    point_metrics_b = metrics_from_confusion(point_b)
    point = {metric: point_metrics_b[metric] - point_metrics_a[metric] for metric in METRICS}

    # These are descriptive variation summaries, distinct from the inferential
    # interval below.  They expose whether uncertainty is driven by geography
    # (the five spatial folds) or by the three few-shot seed schedules.
    fold_differences = {metric: [] for metric in METRICS}
    for fold in folds:
        fold_a = _empty_confusion()
        fold_b = _empty_confusion()
        for probe_seed in seeds_by_fold[fold]:
            group = (fold, probe_seed)
            _add_confusion(
                fold_a, _aggregate([baseline[group][patch] for patch in patch_ids[group]])
            )
            _add_confusion(
                fold_b, _aggregate([candidate[group][patch] for patch in patch_ids[group]])
            )
        metrics_a = metrics_from_confusion(fold_a)
        metrics_b = metrics_from_confusion(fold_b)
        for metric in METRICS:
            fold_differences[metric].append(metrics_b[metric] - metrics_a[metric])

    seed_differences = {metric: [] for metric in METRICS}
    for probe_seed in first_seed_set:
        seed_a = _empty_confusion()
        seed_b = _empty_confusion()
        for fold in folds:
            group = (fold, probe_seed)
            _add_confusion(
                seed_a, _aggregate([baseline[group][patch] for patch in patch_ids[group]])
            )
            _add_confusion(
                seed_b, _aggregate([candidate[group][patch] for patch in patch_ids[group]])
            )
        metrics_a = metrics_from_confusion(seed_a)
        metrics_b = metrics_from_confusion(seed_b)
        for metric in METRICS:
            seed_differences[metric].append(metrics_b[metric] - metrics_a[metric])

    generator = np.random.default_rng(seed)
    samples = {metric: np.empty(n_resamples, dtype=np.float64) for metric in METRICS}
    for index in range(n_resamples):
        sampled_folds = generator.integers(0, len(folds), size=len(folds))
        total_a = _empty_confusion()
        total_b = _empty_confusion()
        for fold_index in sampled_folds:
            fold = folds[int(fold_index)]
            clusters = resolved_clusters[fold]
            cluster_draw = generator.integers(0, len(clusters), size=len(clusters))
            seed_draw = generator.integers(
                0, len(seeds_by_fold[fold]), size=len(seeds_by_fold[fold])
            )
            for seed_index in seed_draw:
                seed_key = seeds_by_fold[fold][int(seed_index)]
                group = (fold, seed_key)
                for cluster_index in cluster_draw:
                    cluster = clusters[int(cluster_index)]
                    _add_confusion(
                        total_a, _aggregate([baseline[group][patch_id] for patch_id in cluster])
                    )
                    _add_confusion(
                        total_b, _aggregate([candidate[group][patch_id] for patch_id in cluster])
                    )
        metrics_a = metrics_from_confusion(total_a)
        metrics_b = metrics_from_confusion(total_b)
        for metric in METRICS:
            samples[metric][index] = metrics_b[metric] - metrics_a[metric]
    return {
        "n_spatial_folds": len(folds),
        "support_schedule_seeds_per_fold": list(first_seed_set),
        "n_patches_per_fold": {f"fold{fold}": len(patch_ids_by_fold[fold]) for fold in folds},
        "n_geographic_clusters_per_fold": {
            f"fold{fold}": len(resolved_clusters[fold]) for fold in folds
        },
        "geographic_cluster_membership_by_fold": {
            f"fold{fold}": [list(cluster) for cluster in resolved_clusters[fold]] for fold in folds
        },
        "resampling_hierarchy": [
            "fold",
            "complete_2x2_geographic_cluster",
            "support_schedule_seed",
        ],
        "descriptive_candidate_minus_baseline_standard_deviation": {
            metric: {
                "across_spatial_folds": _sample_standard_deviation(fold_differences[metric]),
                "across_support_schedule_seeds": _sample_standard_deviation(
                    seed_differences[metric]
                ),
            }
            for metric in METRICS
        },
        "candidate_minus_baseline": _summarize_differences(point, samples),
    }


def hierarchical_paired_bootstrap(
    baseline: Mapping[GroupKey, PatchConfusion],
    candidate: Mapping[GroupKey, PatchConfusion],
    *,
    n_resamples: int,
    seed: int,
    bounds_by_patch: Mapping[str, tuple[float, float, float, float]],
) -> dict[str, Any]:
    """Run only the registered 5-fold, 3-seed, complete-block bootstrap protocol."""
    if n_resamples != PROTOCOL_N_RESAMPLES:
        raise ValueError(f"Registered bootstrap requires exactly {PROTOCOL_N_RESAMPLES} resamples")
    expected_groups = {(fold, probe_seed) for fold in range(5) for probe_seed in (42, 43, 44)}
    if set(baseline) != expected_groups or set(candidate) != expected_groups:
        raise ValueError("Registered bootstrap requires exactly five folds and seeds 42, 43, 44")
    clusters_by_fold = {
        fold: build_two_by_two_geographic_clusters(
            bounds_by_patch, _validate_patch_pair(baseline[(fold, 42)], candidate[(fold, 42)])
        )
        for fold in range(5)
    }
    return _hierarchical_paired_resample(
        baseline,
        candidate,
        n_resamples=n_resamples,
        seed=seed,
        clusters_by_fold=clusters_by_fold,
    )


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


def _record_cells(
    records: list[dict[str, Any]], *, tasks: tuple[str, ...], shots: tuple[str, ...]
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    """Index selected registered records without discarding pairing provenance."""
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
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
        if key in indexed:
            raise ValueError(f"Duplicate result cell: {key}")
        indexed[key] = record
    return indexed


def _paired_value(record: dict[str, Any], key: str) -> Any:
    payload = aggregate._metric_payload(record)
    if key == "label_sha256":
        value = record.get(key)
    elif key == "shot_manifest_label_sha256":
        shot_manifest = payload.get("shot_manifest")
        value = shot_manifest.get("label_sha256") if isinstance(shot_manifest, dict) else None
    elif key == "shot_manifest_split_sha256":
        shot_manifest = payload.get("shot_manifest")
        value = shot_manifest.get("split_sha256") if isinstance(shot_manifest, dict) else None
    else:
        value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Result lacks required pairing provenance: {key}")
    return value


def _protocol_id(record: dict[str, Any]) -> str:
    """Read protocol identity, treating legacy pre-descriptor fixtures as v4 diagnostics."""
    payload = aggregate._metric_payload(record)
    protocol_id = payload.get("protocol_id", "v4_diagnostic")
    if not isinstance(protocol_id, str):
        raise ValueError("Result lacks required pairing provenance: protocol_id")
    return protocol_id


def verify_paired_record_provenance(
    baseline: Mapping[tuple[str, str, int, int], dict[str, Any]],
    candidate: Mapping[tuple[str, str, int, int], dict[str, Any]],
) -> str:
    """Fail closed unless paired runs used identical labels, split, and test patches."""
    if set(baseline) != set(candidate):
        raise ValueError("Paired bootstrap requires identical registered result cells")
    provenance_keys = (
        "label_sha256",
        "spatial_split_sha256",
        "manifest_sha256",
        "shot_manifest_sha256",
        "shot_manifest_label_sha256",
        "shot_manifest_split_sha256",
    )
    for cell in sorted(baseline):
        baseline_protocol = _protocol_id(baseline[cell])
        candidate_protocol = _protocol_id(candidate[cell])
        if baseline_protocol != candidate_protocol:
            raise ValueError(
                f"Paired bootstrap requires identical protocol_id for result cell {cell}"
            )
        cell_provenance_keys = provenance_keys
        if baseline_protocol == "v5_osm_assisted":
            cell_provenance_keys = (*provenance_keys, "test_patch_ids_sha256")
        for key in cell_provenance_keys:
            if _paired_value(baseline[cell], key) != _paired_value(candidate[cell], key):
                raise ValueError(
                    f"Paired bootstrap requires identical {key} for result cell {cell}"
                )
        baseline_payload = aggregate._metric_payload(baseline[cell])
        candidate_payload = aggregate._metric_payload(candidate[cell])
        baseline_confusion = baseline_payload.get("per_patch_confusion")
        candidate_confusion = candidate_payload.get("per_patch_confusion")
        if not isinstance(baseline_confusion, dict) or not isinstance(candidate_confusion, dict):
            raise ValueError("Paired bootstrap requires per-patch test confusion provenance")
        if set(baseline_confusion) != set(candidate_confusion):
            raise ValueError(f"Paired bootstrap requires identical test patch IDs for {cell}")
    protocol_ids = {_protocol_id(record) for record in [*baseline.values(), *candidate.values()]}
    if len(protocol_ids) != 1:
        raise ValueError("Paired bootstrap requires exactly one protocol_id")
    return next(iter(protocol_ids))


def result_identities(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return a deterministic, portable identity list for all bootstrap inputs."""
    return sorted(
        (aggregate.sealed_result_identity(record) for record in records),
        key=lambda item: item["result_id"],
    )


def select_records_for_matrix(
    records: list[dict[str, Any]], *, tasks: tuple[str, ...], shots: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Return only records that participate in the requested comparison matrix."""
    return [
        record
        for record in records
        if str(aggregate._metric_payload(record)["task"]) in tasks
        and str(aggregate._metric_payload(record)["shot"]) in shots
    ]


def build_input_identity_snapshot(
    baseline_identities: list[dict[str, str]], candidate_identities: list[dict[str, str]]
) -> dict[str, Any]:
    """Create a self-contained immutable identity snapshot for one comparison."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "baseline_input_results": baseline_identities,
        "candidate_input_results": candidate_identities,
        "baseline_result_count": len(baseline_identities),
        "candidate_result_count": len(candidate_identities),
        "baseline_input_results_sha256": aggregate.registered._canonical_sha256(
            {"results": baseline_identities}
        ),
        "candidate_input_results_sha256": aggregate.registered._canonical_sha256(
            {"results": candidate_identities}
        ),
    }
    payload["sha256"] = aggregate.registered._canonical_sha256(payload)
    return payload


def derived_bootstrap_admission() -> dict[str, Any]:
    """Derived intervals require a separate evidence-admission decision."""
    return {
        "preliminary": True,
        "paper_eligible": False,
        "admission_status": "derived_statistic_pending_external_admission",
    }


def validate_matrix_dimensions(*, tasks: tuple[str, ...], shots: tuple[str, ...]) -> None:
    """Reject duplicate matrix dimensions before reading any registered artifacts."""
    if len(set(tasks)) != len(tasks) or len(set(shots)) != len(shots):
        raise ValueError("Bootstrap task and shot dimensions must be unique")


def validate_paper_patch_metadata(path: Path) -> None:
    """Pin formal bootstrap clusters to the committed Haidian patch lattice."""
    if path.resolve() != PAPER_PATCH_METADATA.resolve():
        raise ValueError("Paper-facing bootstrap requires the registered patch metadata")
    aggregate.registered.verify_git_head_file(path)
    if aggregate.registered.sha256_file(path) != PAPER_PATCH_METADATA_SHA256:
        raise ValueError("Registered patch metadata hash differs from the V5 protocol")


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
    patch_metadata_path: Path,
    protocol: str | None = None,
    baseline_release_admission_path: Path | None = None,
    candidate_release_admission_path: Path | None = None,
) -> dict[str, Any]:
    """Load sealed records and compute one paired hierarchical interval per task/shot."""
    validate_matrix_dimensions(tasks=tasks, shots=shots)
    if n_resamples != PROTOCOL_N_RESAMPLES:
        raise ValueError(f"Registered bootstrap requires exactly {PROTOCOL_N_RESAMPLES} resamples")
    if not allow_preliminary and (
        tasks != aggregate.PRIMARY_TASKS
        or shots != aggregate.PRIMARY_SHOTS
        or seed != PROTOCOL_RANDOM_SEED
    ):
        raise ValueError(
            "Paper-facing bootstrap requires the preregistered tasks, shots, and random seed"
        )
    if not allow_preliminary:
        validate_paper_patch_metadata(patch_metadata_path)
    if allow_preliminary:
        if (
            baseline_release_admission_path is not None
            or candidate_release_admission_path is not None
        ):
            raise ValueError("Preliminary bootstrap must not consume release admissions")
        baseline_records = aggregate.load_verified_records(
            registry_path, family=baseline_family, allow_preliminary=True
        )
        candidate_records = aggregate.load_verified_records(
            registry_path, family=candidate_family, allow_preliminary=True
        )
        release_admissions = None
    else:
        baseline_records = aggregate.load_admitted_records(
            registry_path,
            family=baseline_family,
            release_admission_path=baseline_release_admission_path,
        )
        candidate_records = aggregate.load_admitted_records(
            registry_path,
            family=candidate_family,
            release_admission_path=candidate_release_admission_path,
        )
        release_admissions = {
            "baseline": {
                "path": str(baseline_release_admission_path.resolve()),
                "sha256": aggregate.registered.sha256_file(baseline_release_admission_path),
            },
            "candidate": {
                "path": str(candidate_release_admission_path.resolve()),
                "sha256": aggregate.registered.sha256_file(candidate_release_admission_path),
            },
        }
    baseline_selected = select_records_for_matrix(baseline_records, tasks=tasks, shots=shots)
    candidate_selected = select_records_for_matrix(candidate_records, tasks=tasks, shots=shots)
    baseline_record_cells = _record_cells(baseline_selected, tasks=tasks, shots=shots)
    candidate_record_cells = _record_cells(candidate_selected, tasks=tasks, shots=shots)
    protocol_id = verify_paired_record_provenance(baseline_record_cells, candidate_record_cells)
    if protocol_id == "v5_osm_assisted" and protocol != "v5_osm_assisted":
        raise ValueError("V5 bootstrap requires --protocol v5_osm_assisted")
    if protocol is not None and protocol_id != protocol:
        raise ValueError("Bootstrap --protocol does not match the result records")
    if protocol == "v5_osm_assisted":
        aggregate.registered.validate_v5_matrix_comparator(baseline_family, candidate_family)
        aggregate.validate_registered_v5_matrix_records(baseline_selected, baseline_family)
        aggregate.validate_registered_v5_matrix_records(candidate_selected, candidate_family)
    baseline = _records_by_cell(baseline_selected, tasks=tasks, shots=shots)
    candidate = _records_by_cell(candidate_selected, tasks=tasks, shots=shots)
    baseline_identities = result_identities(baseline_selected)
    candidate_identities = result_identities(candidate_selected)
    identity_snapshot = build_input_identity_snapshot(baseline_identities, candidate_identities)
    expected = {
        (task, shot, fold, probe_seed)
        for task in tasks
        for shot in shots
        for fold in (0, 1, 2, 3, 4)
        for probe_seed in (42, 43, 44)
    }
    if set(baseline) != expected or set(candidate) != expected:
        raise ValueError("Each family must contain the complete registered 5-fold x 3-seed matrix")
    bounds_by_patch = load_patch_bounds(patch_metadata_path)
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
                bounds_by_patch=bounds_by_patch,
            )
    return {
        "schema_version": 1,
        "method": "paired hierarchical spatial-block bootstrap",
        "comparison_direction": "candidate_minus_baseline",
        "baseline_family": baseline_family,
        "candidate_family": candidate_family,
        "tasks": list(tasks),
        "shots": list(shots),
        "protocol_id": protocol_id,
        **aggregate.registered.result_evidence(protocol_id),
        "source_registry_path": str(registry_path.resolve()),
        "patch_metadata_path": str(patch_metadata_path.resolve()),
        "patch_metadata_sha256": aggregate.registered.sha256_file(patch_metadata_path),
        "input_identity_snapshot": identity_snapshot,
        "release_admissions": release_admissions,
        **derived_bootstrap_admission(),
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
    parser.add_argument("--n-resamples", type=int, default=PROTOCOL_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=PROTOCOL_RANDOM_SEED)
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=Path("configs/regions/haidian_patches.json"),
        help="Projected patch metadata used to form 2 x 2 geographic bootstrap clusters.",
    )
    parser.add_argument("--allow-preliminary", action="store_true")
    parser.add_argument("--baseline-release-admission", type=Path, default=None)
    parser.add_argument("--candidate-release-admission", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        choices=tuple(aggregate.registered.PROTOCOL_DESCRIPTORS),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "v5_osm_assisted":
        validate_paper_bootstrap_output_path(
            args.output,
            baseline_family=args.baseline_family,
            candidate_family=args.candidate_family,
        )
    result = compare_families(
        registry_path=args.registry,
        baseline_family=args.baseline_family,
        candidate_family=args.candidate_family,
        tasks=tuple(args.tasks),
        shots=tuple(args.shots),
        allow_preliminary=args.allow_preliminary,
        n_resamples=args.n_resamples,
        seed=args.seed,
        patch_metadata_path=args.patch_metadata,
        protocol=args.protocol,
        baseline_release_admission_path=args.baseline_release_admission,
        candidate_release_admission_path=args.candidate_release_admission,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = result.pop("input_identity_snapshot")
    snapshot_path = args.output.with_name(f"{args.output.stem}_input_identity_snapshot.json")
    existing = [path for path in (args.output, snapshot_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Bootstrap report exists; refusing to overwrite: {existing[0]}")
    with snapshot_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    result["input_identity_snapshot"] = {
        "path": str(snapshot_path.resolve()),
        "sha256": aggregate.registered.sha256_file(snapshot_path),
        "identity_sha256": snapshot["sha256"],
        "baseline_result_count": snapshot["baseline_result_count"],
        "candidate_result_count": snapshot["candidate_result_count"],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
