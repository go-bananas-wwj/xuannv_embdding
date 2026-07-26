#!/usr/bin/env python3
"""Build immutable release-admission identities for sealed downstream results.

The upstream probe registry is deliberately preliminary and hash-bound.  This
module never changes those files: it only reads them and emits identities that
later release-admission steps can bind to an external archive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.eval import run_registered_paper_downstream as registered


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return payload


def collect_sealed_result_identities(registry_path: Path) -> list[dict[str, str]]:
    """Read preliminary result identities without changing any sealed artifact.

    This narrow primitive deliberately accepts only the preliminary status.
    The future release-admission writer will add archive, matrix-completeness,
    and report-binding checks around these verified identities.
    """
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing result registry: {registry_path}")
    identities: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("Result registry entries must be JSON objects")
        result_id = record.get("result_id")
        if not isinstance(result_id, str) or not result_id or result_id in seen_ids:
            raise ValueError(f"Duplicate or invalid result_id: {result_id}")
        seen_ids.add(result_id)
        if record.get("paper_eligible") is not False or record.get("admission_status") != (
            "registered_preliminary_pending_external_gates"
        ):
            raise ValueError(f"Result is not a sealed preliminary record: {result_id}")
        metrics_value = record.get("metrics_path")
        if not isinstance(metrics_value, str):
            raise ValueError(f"Result lacks metrics_path: {result_id}")
        metrics_path = Path(metrics_value)
        artifact_path = metrics_path.parent / "artifact_manifest.json"
        # This establishes every sealed relationship: registry entry content,
        # artifact sidecar, metrics, predictions, probe, and sidecar paths.
        registered.verify_artifact_registry_binding(artifact_path, registry_path)
        artifact = _load_json(artifact_path, description="artifact manifest")
        metrics = _load_json(metrics_path, description="metrics file")
        if artifact.get("paper_eligible") is not False or artifact.get("admission_status") != (
            "registered_preliminary_pending_external_gates"
        ):
            raise ValueError(f"Artifact is not sealed preliminary evidence: {result_id}")
        if metrics.get("paper_eligible") is not False or metrics.get("admission_status") != (
            "registered_preliminary_pending_external_gates"
        ):
            raise ValueError(f"Metrics are not sealed preliminary evidence: {result_id}")
        for field, value in registered.result_evidence("v5_osm_assisted").items():
            if (
                record.get(field) != value
                or metrics.get(field) != value
                or artifact.get(field) != value
            ):
                raise ValueError(f"V5 evidence descriptor differs across sealed files: {result_id}")
        entry_sha = record.get("registry_entry_sha256")
        if not isinstance(entry_sha, str) or not entry_sha:
            raise ValueError(f"Result lacks registry_entry_sha256: {result_id}")
        identities.append(
            {
                "result_id": result_id,
                "registry_entry_sha256": entry_sha,
                "artifact_sha256": registered.sha256_file(artifact_path),
                "metrics_sha256": registered.sha256_file(metrics_path),
            }
        )
    if not identities:
        raise ValueError("Result registry contains no sealed records")
    return identities
