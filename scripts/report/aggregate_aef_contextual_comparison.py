#!/usr/bin/env python3
"""Fail-closed checks for the annual-AEF versus monthly-Xuannv contextual comparison.

The two products deliberately use different protocol identifiers.  This module
permits only that registered pair while requiring identical downstream split,
labels, support schedule, and held-out patch identities in every paired cell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.eval.run_registered_paper_downstream import (
    sha256_file,
    verify_git_head_file,
    verify_artifact_registry_binding,
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


def contextual_matrix_sha256() -> str:
    """Return the committed AEF contextual-matrix identity used by every baseline cell."""
    if not AEF_MATRIX_PATH.is_file():
        raise FileNotFoundError(f"Missing AEF contextual matrix: {AEF_MATRIX_PATH}")
    verify_git_head_file(AEF_MATRIX_PATH)
    return sha256_file(AEF_MATRIX_PATH)


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
        for key in PAIRING_KEYS:
            left = aef_payload.get(key)
            right = xuannv_payload.get(key)
            if not isinstance(left, str) or not left or left != right:
                raise ValueError(
                    f"Contextual comparison requires identical {key} for result cell {cell}"
                )
        left_confusion = aef_payload.get("per_patch_confusion")
        right_confusion = xuannv_payload.get("per_patch_confusion")
        if not isinstance(left_confusion, Mapping) or not isinstance(right_confusion, Mapping):
            raise ValueError("Contextual comparison requires per-patch test confusion provenance")
        if set(left_confusion) != set(right_confusion):
            raise ValueError(f"Contextual comparison requires identical test patch IDs for {cell}")
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
