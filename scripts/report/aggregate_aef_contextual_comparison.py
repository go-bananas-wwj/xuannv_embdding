#!/usr/bin/env python3
"""Fail-closed checks for the annual-AEF versus monthly-Xuannv contextual comparison.

The two products deliberately use different protocol identifiers.  This module
permits only that registered pair while requiring identical downstream split,
labels, support schedule, and held-out patch identities in every paired cell.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import numpy as np

from scripts.eval.run_registered_paper_downstream import (
    load_registered_v5_matrix,
    resolve_registered_protocol,
    sha256_file,
    verify_git_head_file,
    verify_artifact_registry_binding,
    target_support_sha256,
)

CellKey = tuple[str, str, int, int]

AEF_PROTOCOL = "aef_annual_2025_contextual"
XUANNV_PROTOCOL = "v5_osm_assisted"
AEF_MATRIX_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/eval/rse_aef_annual_2025_contextual_comparison_matrix.json"
)
PAIRING_KEYS = (
    "label_sha256",
    "spatial_split_sha256",
    "manifest_sha256",
    "statistics_registry_sha256",
    "shot_manifest_sha256",
    "test_patch_ids_sha256",
)
EXPECTED_CELLS = {
    (task, shot, fold, seed)
    for task in ("building", "road", "water")
    for shot in ("5", "10")
    for fold in range(5)
    for seed in (42, 43, 44)
}
AEF_IDENTITY = {
    "baseline_id": "aef_annual_2025",
    "baseline_output_identifier": "annual_2025",
    "time_inequivalent_contextual": True,
    "evidence_scope": "osm_assisted_spatial_readout",
}
FROZEN_PROBE = {
    "head": "conv3x3_64_128_64",
    "epochs": 80,
    "batch_size": 8,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "final_epoch_only": True,
}
FROZEN_THRESHOLD_GRID = {
    "grid_start": 0.001,
    "grid_stop_exclusive": 1.0,
    "grid_step": 0.001,
    "candidate_count": 999,
}
METRICS = ("f1_at_threshold", "ap", "auc_roc", "miou", "precision", "recall")


def contextual_matrix_sha256() -> str:
    """Return the committed AEF contextual-matrix identity used by every baseline cell."""
    if not AEF_MATRIX_PATH.is_file():
        raise FileNotFoundError(f"Missing AEF contextual matrix: {AEF_MATRIX_PATH}")
    verify_git_head_file(AEF_MATRIX_PATH)
    return sha256_file(AEF_MATRIX_PATH)


def _contextual_matrix() -> Mapping[str, object]:
    """Load the Git-pinned comparator declaration used to freeze shared assets."""
    contextual_matrix_sha256()
    matrix = json.loads(AEF_MATRIX_PATH.read_text(encoding="utf-8"))
    if not isinstance(matrix, Mapping):
        raise ValueError("Contextual comparison matrix must be an object")
    return matrix


def _payload(record: Mapping[str, object]) -> Mapping[str, object]:
    payload = record.get("metric_provenance")
    if not isinstance(payload, Mapping):
        raise ValueError("Contextual comparison record lacks metric_provenance")
    return payload


def _cell_from_payload(payload: Mapping[str, object]) -> CellKey:
    task = payload.get("task")
    shot = payload.get("shot")
    fold = payload.get("fold")
    seed = payload.get("shot_seed")
    if (
        not isinstance(task, str)
        or not isinstance(shot, str)
        or not isinstance(fold, int)
        or not isinstance(seed, int)
    ):
        raise ValueError("Contextual comparison payload lacks a valid registered cell identity")
    return task, shot, fold, seed


def _verify_test_confusion(payload: Mapping[str, object]) -> None:
    """Bind each recorded patch confusion map to the declared test patch support."""
    confusion = payload.get("per_patch_confusion")
    expected_hash = payload.get("test_patch_ids_sha256")
    if not isinstance(confusion, Mapping) or not isinstance(expected_hash, str):
        raise ValueError("Contextual comparison requires per-patch test confusion provenance")
    patch_ids = sorted(confusion)
    if not patch_ids or any(not isinstance(patch_id, str) or not patch_id for patch_id in patch_ids):
        raise ValueError("Contextual comparison has invalid per-patch test IDs")
    actual_hash = hashlib.sha256("\n".join(patch_ids).encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Contextual comparison test patch IDs differ from per-patch confusion")
    for patch_id in patch_ids:
        counts = confusion[patch_id]
        if not isinstance(counts, Mapping):
            raise ValueError("Contextual comparison has invalid patch confusion counts")
        for key in ("tp", "fp", "fn", "tn"):
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Contextual comparison has invalid patch confusion counts")


def verify_prediction_target_support(payload: Mapping[str, object]) -> None:
    """Recompute per-patch support hashes from the sealed test prediction archive."""
    prediction_value = payload.get("prediction_file")
    support = payload.get("per_patch_target_support_sha256")
    if not isinstance(prediction_value, str) or not isinstance(support, Mapping):
        raise ValueError("Contextual comparison lacks target-support archive provenance")
    prediction_path = Path(prediction_value)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing contextual test prediction archive: {prediction_path}")
    with np.load(prediction_path, allow_pickle=True) as archive:
        patch_ids = [str(item) for item in archive["patch_ids"].tolist()]
        targets = np.asarray(archive["targets"])
    if len(patch_ids) != len(set(patch_ids)) or targets.shape[0] != len(patch_ids):
        raise ValueError("Contextual prediction archive has invalid patch target support")
    observed = {patch_id: target_support_sha256(targets[index]) for index, patch_id in enumerate(patch_ids)}
    if observed != dict(support):
        raise ValueError("Contextual prediction archive target support differs from metric provenance")


def _verify_fixed_protocol_bindings(
    aef_payload: Mapping[str, object], xuannv_payload: Mapping[str, object], *, task: str
) -> None:
    """Reject a pair that agrees with itself but not with the registered assets."""
    matrix = _contextual_matrix()
    labels = matrix.get("labels")
    label = labels.get(task) if isinstance(labels, Mapping) else None
    expected = {
        "spatial_split_sha256": matrix.get("spatial_split_sha256"),
        "manifest_sha256": matrix.get("manifest_sha256"),
        "statistics_registry_sha256": matrix.get("statistics_registry_sha256"),
        "label_sha256": label.get("tree_sha256") if isinstance(label, Mapping) else None,
    }
    for key, value in expected.items():
        if not isinstance(value, str) or any(
            payload.get(key) != value for payload in (aef_payload, xuannv_payload)
        ):
            raise ValueError(f"Contextual comparison has an unregistered {key}")
    # The descriptor validates shared assets; the matrix adds the immutable
    # family/fold contract and its pinned file hash.
    resolve_registered_protocol(XUANNV_PROTOCOL)
    load_registered_v5_matrix()


def load_verified_contextual_records(
    registry_path: Path, *, expected_protocol: str
) -> dict[CellKey, dict[str, object]]:
    """Load only artifact-bound records for one side of the contextual comparison."""
    if expected_protocol not in (AEF_PROTOCOL, XUANNV_PROTOCOL):
        raise ValueError("Contextual comparison has an unsupported protocol")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing contextual result registry: {registry_path}")
    records: dict[CellKey, dict[str, object]] = {}
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Contextual result registry has an invalid record")
        payload = _payload(record)
        if payload.get("protocol_id") != expected_protocol:
            continue
        metrics_path = Path(str(record.get("metrics_path", "")))
        artifact_path = metrics_path.parent / "artifact_manifest.json"
        if (
            not isinstance(record.get("result_id"), str)
            or not metrics_path.is_file()
            or sha256_file(metrics_path) != record.get("metrics_sha256")
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != record.get("artifact_sha256")
        ):
            raise ValueError("Contextual result record has an invalid sealed artifact reference")
        verify_artifact_registry_binding(artifact_path, registry_path)
        if json.loads(metrics_path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Contextual result metric provenance differs from its sealed metric file")
        verify_prediction_target_support(payload)
        cell = _cell_from_payload(payload)
        if cell in records:
            raise ValueError(f"Duplicate contextual result cell: {cell}")
        records[cell] = record
    if set(records) != EXPECTED_CELLS:
        raise ValueError("Contextual registry lacks the complete registered 90-cell matrix")
    return records


def verify_contextual_pairing(
    aef: Mapping[CellKey, Mapping[str, object]],
    xuannv: Mapping[CellKey, Mapping[str, object]],
) -> None:
    """Accept exactly the registered AEF/Xuannv pair with shared readout provenance."""
    if set(aef) != set(xuannv):
        raise ValueError("Contextual comparison requires identical registered result cells")
    if set(aef) != EXPECTED_CELLS:
        raise ValueError("Contextual comparison requires the complete registered 90-cell matrix")
    for cell in sorted(aef):
        aef_payload = _payload(aef[cell])
        xuannv_payload = _payload(xuannv[cell])
        payload_cell = _cell_from_payload(aef_payload)
        if payload_cell != cell:
            raise ValueError(f"Contextual baseline payload does not match its cell key: {cell}")
        payload_cell = _cell_from_payload(xuannv_payload)
        if payload_cell != cell:
            raise ValueError(f"Contextual candidate payload does not match its cell key: {cell}")
        if aef_payload.get("protocol_id") != AEF_PROTOCOL:
            raise ValueError(f"Contextual baseline has invalid protocol for result cell {cell}")
        if xuannv_payload.get("protocol_id") != XUANNV_PROTOCOL:
            raise ValueError(f"Contextual candidate has invalid protocol for result cell {cell}")
        if any(aef_payload.get(key) != value for key, value in AEF_IDENTITY.items()):
            raise ValueError(f"Contextual baseline has invalid baseline identity for result cell {cell}")
        if aef_payload.get("comparison_matrix_sha256") != contextual_matrix_sha256():
            raise ValueError("Contextual baseline has an invalid comparison-matrix hash")
        if xuannv_payload.get("family") != "full_150":
            raise ValueError(f"Contextual candidate has invalid full_150 identity for result cell {cell}")
        export = xuannv_payload.get("embedding_export")
        provenance = xuannv_payload.get("provenance")
        if (
            not isinstance(export, Mapping)
            or export.get("month") != "202604"
            or export.get("protocol_id") != XUANNV_PROTOCOL
            or not isinstance(provenance, Mapping)
            or provenance.get("family") != "full_150"
            or not isinstance(provenance.get("config_sha256"), str)
            or not isinstance(provenance.get("checkpoint_sha256"), str)
        ):
            raise ValueError(f"Contextual candidate has invalid registered V5 encoder identity for {cell}")
        registry = xuannv_payload.get("embedding_registry")
        if (
            not isinstance(registry, Mapping)
            or registry.get("family") != "full_150"
            or registry.get("encoder_fold") != cell[2]
            or registry.get("month") != "202604"
            or registry.get("protocol_id") != XUANNV_PROTOCOL
            or registry.get("patch_count") != 320
            or registry.get("config_sha256") != provenance.get("config_sha256")
            or registry.get("checkpoint_sha256") != provenance.get("checkpoint_sha256")
            or registry.get("manifest_sha256") != xuannv_payload.get("manifest_sha256")
            or not isinstance(registry.get("embedding_file_index_sha256"), str)
            or not isinstance(registry.get("canonical_export_provenance_sha256"), str)
        ):
            raise ValueError(f"Contextual candidate has invalid sealed V5 embedding registry for {cell}")
        _verify_fixed_protocol_bindings(aef_payload, xuannv_payload, task=cell[0])
        for key in PAIRING_KEYS:
            left = aef_payload.get(key)
            right = xuannv_payload.get(key)
            if not isinstance(left, str) or not left or left != right:
                raise ValueError(
                    f"Contextual comparison requires identical {key} for result cell {cell}"
                )
        _verify_test_confusion(aef_payload)
        _verify_test_confusion(xuannv_payload)
        left_confusion = aef_payload.get("per_patch_confusion")
        right_confusion = xuannv_payload.get("per_patch_confusion")
        if not isinstance(left_confusion, Mapping) or not isinstance(right_confusion, Mapping):
            raise ValueError("Contextual comparison requires per-patch test confusion provenance")
        if set(left_confusion) != set(right_confusion):
            raise ValueError(f"Contextual comparison requires identical test patch IDs for {cell}")
        left_support = aef_payload.get("per_patch_target_support_sha256")
        right_support = xuannv_payload.get("per_patch_target_support_sha256")
        if (
            not isinstance(left_support, Mapping)
            or not isinstance(right_support, Mapping)
            or set(left_support) != set(left_confusion)
            or set(right_support) != set(right_confusion)
            or any(
                not isinstance(left_support.get(patch_id), str)
                or not left_support.get(patch_id)
                or left_support.get(patch_id) != right_support.get(patch_id)
                for patch_id in left_confusion
            )
        ):
            raise ValueError(
                f"Contextual comparison requires identical per-patch target support hashes for {cell}"
            )
        for patch_id in left_confusion:
            left_counts = left_confusion[patch_id]
            right_counts = right_confusion[patch_id]
            assert isinstance(left_counts, Mapping) and isinstance(right_counts, Mapping)
            if (
                left_counts["tp"] + left_counts["fn"]
                != right_counts["tp"] + right_counts["fn"]
                or left_counts["fp"] + left_counts["tn"]
                != right_counts["fp"] + right_counts["tn"]
            ):
                raise ValueError(
                    f"Contextual comparison requires identical valid pixel support for {cell} {patch_id}"
                )
        aef_probe = aef_payload.get("probe")
        xuannv_probe = xuannv_payload.get("probe")
        if not isinstance(aef_probe, Mapping) or not isinstance(xuannv_probe, Mapping):
            raise ValueError("Contextual comparison requires frozen probe provenance")
        if (
            any(aef_probe.get(key) != value for key, value in FROZEN_PROBE.items())
            or any(xuannv_probe.get(key) != value for key, value in FROZEN_PROBE.items())
        ):
            raise ValueError(f"Contextual comparison requires the registered probe contract for {cell}")
        for payload in (aef_payload, xuannv_payload):
            selection = payload.get("validation_threshold_selection")
            if not isinstance(selection, Mapping) or any(
                selection.get(key) != value for key, value in FROZEN_THRESHOLD_GRID.items()
            ):
                raise ValueError(
                    f"Contextual comparison requires the registered validation threshold grid for {cell}"
                )


def summarize_contextual_records(
    records: Mapping[CellKey, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize the complete 90-cell matrix by seed-level five-fold means."""
    if set(records) != EXPECTED_CELLS:
        raise ValueError("Contextual summary requires the complete registered 90-cell matrix")
    summary: dict[str, dict[str, object]] = {}
    for task in ("building", "road", "water"):
        for shot in ("5", "10"):
            metric_summary: dict[str, object] = {}
            for metric in METRICS:
                per_seed: dict[str, float] = {}
                pooled: list[float] = []
                for seed in (42, 43, 44):
                    values: list[float] = []
                    for fold in range(5):
                        value = _payload(records[(task, shot, fold, seed)]).get(metric)
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                        ):
                            raise ValueError(f"Contextual result lacks a finite numeric {metric}")
                        values.append(float(value))
                    per_seed[str(seed)] = mean(values)
                    pooled.extend(values)
                seed_values = list(per_seed.values())
                metric_summary[metric] = {
                    "per_seed_fold_means": per_seed,
                    "mean": mean(seed_values),
                    "std": stdev(seed_values),
                    "n_seeds": len(seed_values),
                    "pooled_fold_mean": mean(pooled),
                    "pooled_fold_std": stdev(pooled),
                    "n_fold_seed_runs": len(pooled),
                }
            summary[f"{task}|{shot}"] = {
                "task": task,
                "shot": shot,
                "metrics": metric_summary,
            }
    return summary
