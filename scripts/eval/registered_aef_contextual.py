"""Fail-closed protocol checks for the annual-AEF contextual comparator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.eval.run_registered_paper_downstream import verify_git_head_file

CONTEXTUAL_IDENTITY = {
    "baseline_id": "aef_annual_2025",
    "baseline_protocol_id": "aef_annual_2025_contextual",
    "candidate_family": "full_150",
    "candidate_protocol_id": "v5_osm_assisted",
    "baseline_output_identifier": "annual_2025",
    "candidate_month": "202604",
    "temporal_statement": (
        "Annual AEF 2025 versus monthly Xuannv 2026-04 "
        "under an aligned spatial downstream protocol."
    ),
    "evidence_scope": "osm_assisted_spatial_readout",
}
REQUIRED_PROHIBITED_CLAIMS = {
    "matched_temporal_information",
    "matched_inputs",
    "labelled_patch_efficiency_against_information_matched_baseline",
    "independent_transfer",
    "geographically_unseen_pretraining",
    "contemporaneous_ground_truth",
    "monthly_change_evidence",
    "global_superiority",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable protocol input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schedule_set_digest(schedule_root: Path, index_file: str) -> tuple[int, str]:
    files = sorted(path for path in schedule_root.glob("*.json") if path.name != index_file)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\n")
    return len(files), digest.hexdigest()


def load_contextual_matrix(path: Path) -> dict[str, Any]:
    """Load a Git-pinned matrix only when it declares the strict contextual contract."""
    verify_git_head_file(path)
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1:
        raise ValueError("AEF contextual matrix has an invalid schema")
    if any(matrix.get(key) != value for key, value in CONTEXTUAL_IDENTITY.items()):
        raise ValueError("AEF contextual matrix has an invalid fixed identity")
    if matrix.get("time_inequivalent_contextual") is not True:
        raise ValueError("AEF comparison must remain explicitly time-inequivalent contextual")
    if matrix.get("expected_job_count") != 90:
        raise ValueError("AEF contextual matrix must freeze exactly 90 probe jobs")
    canonical_cells = {
        "candidate_family": "full_150",
        "tasks": ["building", "road", "water"],
        "shots": ["5", "10"],
        "folds": [0, 1, 2, 3, 4],
        "seeds": [42, 43, 44],
    }
    if any(matrix.get(key) != value for key, value in canonical_cells.items()):
        raise ValueError("AEF contextual matrix differs from the canonical 90-cell contract")
    if not REQUIRED_PROHIBITED_CLAIMS.issubset(set(matrix.get("prohibited_claims", []))):
        raise ValueError("AEF contextual matrix lacks required prohibited claims")
    if not isinstance(matrix.get("shot_schedule"), dict):
        raise ValueError("AEF contextual matrix lacks a frozen shot schedule")
    return matrix


def verify_frozen_shot_schedules(matrix: dict[str, Any]) -> None:
    """Reject any schedule root that differs from the matrix's 45-file snapshot."""
    spec = matrix["shot_schedule"]
    root = Path(str(spec["root"]))
    index = root / str(spec["index_file"])
    if not index.is_file() or sha256_file(index) != spec.get("index_sha256"):
        raise ValueError("AEF contextual shot schedule index does not match the frozen hash")
    payload = json.loads(index.read_text(encoding="utf-8"))
    schedules = payload.get("schedules")
    if not isinstance(schedules, list) or len(schedules) != spec.get("schedule_file_count"):
        raise ValueError("AEF contextual shot schedule index has an invalid file count")
    if payload.get("split_sha256") != matrix.get("spatial_split_sha256"):
        raise ValueError("AEF contextual shot schedule index split does not match the matrix")
    expected_cells = {
        (task, fold, seed)
        for task in matrix["tasks"]
        for fold in matrix["folds"]
        for seed in matrix["seeds"]
    }
    observed_cells: set[tuple[str, int, int]] = set()
    for item in schedules:
        if not isinstance(item, dict):
            raise ValueError("AEF contextual shot schedule index contains an invalid item")
        cell = (str(item.get("task")), item.get("fold"), item.get("seed"))
        if not isinstance(cell[1], int) or not isinstance(cell[2], int):
            raise ValueError("AEF contextual shot schedule index has an invalid cell identity")
        observed_cells.add(cell)
        path = root / str(item.get("path", ""))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError("AEF contextual shot schedule file does not match the frozen index")
        schedule = json.loads(path.read_text(encoding="utf-8"))
        if schedule.get("schema_version") != 3:
            raise ValueError("AEF contextual shot schedule file has an invalid schema")
        if schedule.get("protocol_id") != spec.get("protocol_id"):
            raise ValueError("AEF contextual shot schedule file protocol does not match the matrix")
        if (
            schedule.get("task"),
            schedule.get("fold"),
            schedule.get("seed"),
        ) != cell:
            raise ValueError("AEF contextual shot schedule file has an invalid cell identity")
        if schedule.get("split_sha256") != matrix.get("spatial_split_sha256"):
            raise ValueError("AEF contextual shot schedule file split does not match the matrix")
        label_digest = matrix["labels"].get(cell[0], {}).get("tree_sha256")
        if schedule.get("label_sha256") != label_digest:
            raise ValueError("AEF contextual shot schedule file label does not match the matrix")
        rule = schedule.get("rule")
        if not isinstance(rule, dict) or rule.get("support_unit") != "mixed_class_labeled_patch":
            raise ValueError("AEF contextual shot schedule file has an invalid support rule")
        if rule.get("selection") != "deterministic_nested_prefix":
            raise ValueError("AEF contextual shot schedule file has an invalid support selection")
        sets = schedule.get("sets")
        if not isinstance(sets, dict):
            raise ValueError("AEF contextual shot schedule file lacks support sets")
        for shot in matrix["shots"]:
            item_set = sets.get(shot)
            if not isinstance(item_set, dict) or len(item_set.get("train_patch_ids", [])) != int(
                shot
            ):
                raise ValueError("AEF contextual shot schedule file lacks a required shot set")
    if observed_cells != expected_cells:
        raise ValueError("AEF contextual shot schedule index has an invalid cell identity set")
    count, digest = _schedule_set_digest(root, str(spec["index_file"]))
    if count != spec.get("schedule_file_count") or digest != spec.get(
        "ordered_filename_sha256_set_sha256"
    ):
        raise ValueError("AEF contextual shot schedule set does not match the frozen hash")


def verify_runtime_source_hashes(matrix: dict[str, Any], repo_root: Path) -> None:
    """Require the exact probe readers and schedule generator recorded by the matrix."""
    reader_sources = matrix.get("probe", {}).get("reader_sources")
    if not isinstance(reader_sources, dict) or not reader_sources:
        raise ValueError("AEF contextual matrix lacks runtime source hashes")
    expected_sources = {
        **reader_sources,
        str(matrix["shot_schedule"].get("generator")): matrix["shot_schedule"].get(
            "generator_sha256"
        ),
    }
    for relative, expected_digest in expected_sources.items():
        source = repo_root / relative
        if not source.is_file() or sha256_file(source) != expected_digest:
            raise ValueError(f"AEF contextual runtime source hash mismatch: {relative}")
