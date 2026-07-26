#!/usr/bin/env python3
"""Build immutable release-admission identities for sealed downstream results.

The upstream probe registry is deliberately preliminary and hash-bound.  This
module never changes those files: it only reads them and emits identities that
later release-admission steps can bind to an external archive.
"""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from scripts.eval import run_registered_paper_downstream as registered
from scripts.report import aggregate_registered_paper_results as aggregate


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return payload


def verify_release_anchor(registry_path: Path, anchor_path: Path) -> dict[str, Any]:
    """Verify a Git-sealed anchor against its versioned Zenodo registry bytes."""
    registered.verify_git_head_file(anchor_path)
    anchor = _load_json(anchor_path, description="release anchor")
    expected_fields = {
        "schema_version",
        "protocol_id",
        "registry_path",
        "registry_sha256",
        "zenodo_doi",
        "registry_url",
    }
    unexpected_fields = set(anchor).difference(expected_fields)
    if unexpected_fields:
        raise ValueError(f"Release anchor contains unexpected fields: {sorted(unexpected_fields)}")
    if anchor.get("schema_version") != 1 or anchor.get("protocol_id") != "v5_osm_assisted":
        raise ValueError("Invalid V5 release-anchor schema")
    if anchor.get("registry_path") != str(registry_path.resolve()):
        raise ValueError("Release-anchor registry path does not match the requested registry")
    registry_sha256 = registered.sha256_file(registry_path)
    if anchor.get("registry_sha256") != registry_sha256:
        raise ValueError("Release-anchor registry hash does not match the requested registry")
    doi = anchor.get("zenodo_doi")
    url = anchor.get("registry_url")
    match = (
        re.fullmatch(r"https://doi\.org/10\.5281/zenodo\.(\d+)", doi)
        if isinstance(doi, str)
        else None
    )
    if match is None:
        raise ValueError("Release anchor requires a versioned Zenodo DOI")
    record_id = match.group(1)
    if not isinstance(url, str) or not re.fullmatch(
        rf"https://zenodo\.org/records/{record_id}/files/[^?#]+(?:\?[^#]*)?", url
    ):
        raise ValueError("Release anchor URL is not a file in the DOI's Zenodo record")
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - DOI-bound archive URL.
            archived_bytes = response.read()
    except OSError as exc:
        raise ValueError("Could not fetch the external registry archive") from exc
    if sha256(archived_bytes).hexdigest() != registry_sha256:
        raise ValueError("External registry archive hash does not match the release anchor")
    return {
        **anchor,
        "path": str(anchor_path.resolve()),
        "sha256": registered.sha256_file(anchor_path),
    }


def collect_sealed_result_identities(
    registry_path: Path, *, selected_result_ids: set[str] | None = None
) -> list[dict[str, str]]:
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
        if selected_result_ids is not None and result_id not in selected_result_ids:
            continue
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
        identity = aggregate.sealed_result_identity(record)
        if identity["artifact_sha256"] != registered.sha256_file(artifact_path):
            raise ValueError(f"Artifact identity differs from sealed sidecar: {result_id}")
        if identity["metrics_sha256"] != registered.sha256_file(metrics_path):
            raise ValueError(f"Metric identity differs from sealed sidecar: {result_id}")
        identities.append(identity)
    if (
        selected_result_ids is not None
        and seen_ids.intersection(selected_result_ids) != selected_result_ids
    ):
        raise ValueError("Selected release-admission result IDs are missing from the registry")
    if not identities:
        raise ValueError("Result registry contains no sealed records")
    return identities


def validate_complete_v5_family_records(records: list[dict[str, Any]], family: str) -> None:
    """Require the exact preregistered 3 x 2 x 5 x 3 V5 probe matrix."""
    expected = {
        (task, shot, fold, seed)
        for task in aggregate.PRIMARY_TASKS
        for shot in aggregate.PRIMARY_SHOTS
        for fold in aggregate.PRIMARY_FOLDS
        for seed in aggregate.PRIMARY_SEEDS
    }
    observed = {
        (
            str(aggregate._metric_payload(record)["task"]),
            str(aggregate._metric_payload(record)["shot"]),
            int(aggregate._metric_payload(record)["fold"]),
            int(aggregate._metric_payload(record)["shot_seed"]),
        )
        for record in records
    }
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if missing or unexpected or len(observed) != len(records):
        raise ValueError(
            "V5 probe matrix is incomplete or contains duplicate/unregistered cells: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, records={len(records)}"
        )
    aggregate.validate_report_evidence(records, "weak_supervision")
    aggregate.validate_registered_v5_matrix_records(records, family)
    aggregate.summarize_records(
        records,
        tasks=aggregate.PRIMARY_TASKS,
        shots=aggregate.PRIMARY_SHOTS,
        folds=aggregate.PRIMARY_FOLDS,
        seeds=aggregate.PRIMARY_SEEDS,
    )


def verify_preliminary_aggregate_report(
    report_path: Path,
    *,
    registry_path: Path,
    registry_sha256: str,
    family: str,
    selected_results_sha256: str,
    expected_summary: dict[str, Any],
    expected_record_count: int,
) -> dict[str, str]:
    """Bind an admission to the exact preliminary aggregate used for its claims."""
    report = _load_json(report_path, description="preliminary aggregate report")
    if report.get("preliminary") is not True:
        raise ValueError("Aggregate report must be preliminary before external admission")
    if report.get("family") != family or report.get("protocol_id") != "v5_osm_assisted":
        raise ValueError("Aggregate report family or protocol differs from release admission")
    if report.get("report_kind") != "weak_supervision" or report.get("primary_matrix") is not True:
        raise ValueError("Aggregate report is not the preregistered weak-supervision matrix")
    if report.get("record_count") != expected_record_count:
        raise ValueError("Aggregate report record count differs from sealed results")
    if report.get("summary") != expected_summary:
        raise ValueError("Aggregate summary differs from sealed-result recomputation")
    if (
        report.get("evidence_class") != "osm_assisted_spatial_readout"
        or report.get("label_independence_status") != "osm_overlapping_not_independent"
    ):
        raise ValueError("Aggregate report evidence descriptor differs from release admission")
    if report.get("registry") != str(registry_path.resolve()):
        raise ValueError("Aggregate report registry path differs from release admission")
    if report.get("registry_sha256") != registry_sha256:
        raise ValueError("Aggregate report registry hash differs from release admission")
    if report.get("selected_results_sha256") != selected_results_sha256:
        raise ValueError("Aggregate selected-result identity differs from release admission")
    return {"path": str(report_path.resolve()), "sha256": registered.sha256_file(report_path)}


def verify_hash_bound_report(
    expected: dict[str, str], verified: dict[str, str], description: str
) -> None:
    """Reject a report whose current location or bytes differ from its admission record."""
    if expected.get("path") != verified.get("path"):
        raise ValueError(f"{description} path differs from the release admission")
    if expected.get("sha256") != verified.get("sha256"):
        raise ValueError(f"{description} hash differs from the release admission")


def build_release_admission(
    registry_path: Path, anchor_path: Path, *, family: str, aggregate_report_path: Path
) -> dict[str, Any]:
    """Build the external overlay that admits a verified set of preliminary results.

    The caller is responsible for writing this payload once, committing it, and
    later requiring that committed byte sequence in every paper-facing consumer.
    """
    anchor = verify_release_anchor(registry_path, anchor_path)
    records = aggregate.load_verified_records(registry_path, family=family, allow_preliminary=True)
    validate_complete_v5_family_records(records, family)
    selected_ids = {str(record["result_id"]) for record in records}
    identities = sorted(
        collect_sealed_result_identities(registry_path, selected_result_ids=selected_ids),
        key=lambda item: item["result_id"],
    )
    if registered.sha256_file(registry_path) != anchor["registry_sha256"]:
        raise ValueError("Result registry changed after release-anchor verification")
    selected_results_sha256 = registered._canonical_sha256({"selected_results": identities})
    aggregate_report = verify_preliminary_aggregate_report(
        aggregate_report_path,
        registry_path=registry_path,
        registry_sha256=anchor["registry_sha256"],
        family=family,
        selected_results_sha256=selected_results_sha256,
        expected_summary=aggregate.summarize_records(
            records,
            tasks=aggregate.PRIMARY_TASKS,
            shots=aggregate.PRIMARY_SHOTS,
            folds=aggregate.PRIMARY_FOLDS,
            seeds=aggregate.PRIMARY_SEEDS,
        ),
        expected_record_count=len(records),
    )
    return {
        "schema_version": 1,
        "protocol_id": "v5_osm_assisted",
        **registered.result_evidence("v5_osm_assisted"),
        "paper_eligible": True,
        "admission_status": "external_release_anchor_verified",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": anchor["registry_sha256"],
        "family": family,
        "release_anchor": anchor,
        "aggregate_report": aggregate_report,
        "selected_results": identities,
        "selected_results_sha256": selected_results_sha256,
    }


def load_admitted_records(
    registry_path: Path,
    *,
    family: str,
    release_admission_path: Path,
) -> list[dict[str, Any]]:
    """Verify a Git-sealed admission overlay and return exactly its raw records."""
    validate_release_admission_output_path(release_admission_path, family)
    registered.verify_git_head_file(release_admission_path)
    payload = _load_json(release_admission_path, description="release admission")
    if payload.get("schema_version") != 1 or payload.get("protocol_id") != "v5_osm_assisted":
        raise ValueError("Invalid release-admission schema")
    if payload.get("paper_eligible") is not True or payload.get("family") != family:
        raise ValueError("Release admission does not authorize the requested family")
    if payload.get("registry_path") != str(registry_path.resolve()):
        raise ValueError("Release admission registry path differs from requested registry")
    if payload.get("registry_sha256") != registered.sha256_file(registry_path):
        raise ValueError("Release admission registry hash differs from requested registry")
    anchor = payload.get("release_anchor")
    if not isinstance(anchor, dict) or not isinstance(anchor.get("path"), str):
        raise ValueError("Release admission lacks a sealed release anchor")
    anchor_path = Path(anchor["path"])
    verified_anchor = verify_release_anchor(registry_path, anchor_path)
    if verified_anchor != anchor:
        raise ValueError("Release admission anchor differs from its verified Git artifact")
    records = aggregate.load_verified_records(registry_path, family=family, allow_preliminary=True)
    validate_complete_v5_family_records(records, family)
    selected_ids = {str(record["result_id"]) for record in records}
    identities = sorted(
        collect_sealed_result_identities(registry_path, selected_result_ids=selected_ids),
        key=lambda item: item["result_id"],
    )
    if identities != payload.get("selected_results") or registered._canonical_sha256(
        {"selected_results": identities}
    ) != payload.get("selected_results_sha256"):
        raise ValueError("Release admission selected-result identities differ from sealed records")
    verified_report = verify_preliminary_aggregate_report(
        Path(payload.get("aggregate_report", {}).get("path", "")),
        registry_path=registry_path,
        registry_sha256=str(payload["registry_sha256"]),
        family=family,
        selected_results_sha256=str(payload["selected_results_sha256"]),
        expected_summary=aggregate.summarize_records(
            records,
            tasks=aggregate.PRIMARY_TASKS,
            shots=aggregate.PRIMARY_SHOTS,
            folds=aggregate.PRIMARY_FOLDS,
            seeds=aggregate.PRIMARY_SEEDS,
        ),
        expected_record_count=len(records),
    )
    expected_report = payload.get("aggregate_report")
    if not isinstance(expected_report, dict):
        raise ValueError("Release admission lacks a hash-bound aggregate report")
    verify_hash_bound_report(expected_report, verified_report, "Aggregate report")
    return records


def write_release_admission(
    registry_path: Path,
    anchor_path: Path,
    output_path: Path,
    *,
    family: str,
    aggregate_report_path: Path,
) -> dict[str, Any]:
    """Write a new admission record once; an existing admission is immutable."""
    payload = build_release_admission(
        registry_path,
        anchor_path,
        family=family,
        aggregate_report_path=aggregate_report_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Release admission exists; refusing to overwrite: {output_path}"
        ) from exc
    return payload


def _comparison_input_admission(
    path: Path, *, family: str, role: str
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Read one Git-sealed family admission as an input to a paired comparison."""
    registered.verify_git_head_file(path)
    payload = _load_json(path, description=f"{role} release-admission")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "v5_osm_assisted"
        or payload.get("paper_eligible") is not True
        or payload.get("family") != family
    ):
        raise ValueError(f"Invalid {role} release-admission for family {family}")
    registry_value = payload.get("registry_path")
    if not isinstance(registry_value, str):
        raise ValueError(f"{role.capitalize()} release-admission lacks its registry path")
    records = load_admitted_records(
        Path(registry_value), family=family, release_admission_path=path
    )
    identities = sorted(
        (aggregate.sealed_result_identity(record) for record in records),
        key=lambda item: item["result_id"],
    )
    return {"path": str(path.resolve()), "sha256": registered.sha256_file(path)}, identities


def _validate_comparison_snapshot(
    snapshot_path: Path,
    *,
    baseline_identities: list[dict[str, str]],
    candidate_identities: list[dict[str, str]],
    baseline_family: str,
    candidate_family: str,
) -> dict[str, str]:
    """Verify that the bootstrap's snapshot is exactly its two admitted inputs."""
    validate_paired_bootstrap_snapshot_path(
        snapshot_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    registered.verify_git_head_file(snapshot_path)
    snapshot = _load_json(snapshot_path, description="paired bootstrap input identity snapshot")
    expected = {
        "schema_version": 1,
        "baseline_input_results": baseline_identities,
        "candidate_input_results": candidate_identities,
        "baseline_result_count": len(baseline_identities),
        "candidate_result_count": len(candidate_identities),
        "baseline_input_results_sha256": registered._canonical_sha256(
            {"results": baseline_identities}
        ),
        "candidate_input_results_sha256": registered._canonical_sha256(
            {"results": candidate_identities}
        ),
    }
    expected["sha256"] = registered._canonical_sha256(expected)
    if snapshot != expected:
        raise ValueError("Paired bootstrap input identities differ from its admitted families")
    return {
        "path": str(snapshot_path.resolve()),
        "sha256": registered.sha256_file(snapshot_path),
        "identity_sha256": expected["sha256"],
        "baseline_result_count": len(baseline_identities),
        "candidate_result_count": len(candidate_identities),
    }


def _verify_bootstrap_comparisons(
    bootstrap: dict[str, Any],
    *,
    baseline_release_admission_path: Path,
    candidate_release_admission_path: Path,
    baseline_family: str,
    candidate_family: str,
) -> None:
    """Recompute all registered intervals before promoting a derived report."""
    comparisons = bootstrap.get("comparisons")
    registry_value = bootstrap.get("source_registry_path")
    patch_metadata_value = bootstrap.get("patch_metadata_path")
    if (
        not isinstance(comparisons, dict)
        or not isinstance(registry_value, str)
        or not isinstance(patch_metadata_value, str)
    ):
        raise ValueError("Paired bootstrap lacks recomputable comparison inputs")
    patch_metadata_path = Path(patch_metadata_value)
    if bootstrap.get("patch_metadata_sha256") != registered.sha256_file(patch_metadata_path):
        raise ValueError(
            "Paired bootstrap patch metadata hash differs from its declared provenance"
        )
    from scripts.report import paired_spatial_bootstrap as bootstrap_runner

    recomputed = bootstrap_runner.compare_families(
        registry_path=Path(registry_value),
        baseline_family=baseline_family,
        candidate_family=candidate_family,
        tasks=aggregate.PRIMARY_TASKS,
        shots=aggregate.PRIMARY_SHOTS,
        allow_preliminary=False,
        n_resamples=bootstrap_runner.PROTOCOL_N_RESAMPLES,
        seed=bootstrap_runner.PROTOCOL_RANDOM_SEED,
        patch_metadata_path=patch_metadata_path,
        protocol="v5_osm_assisted",
        baseline_release_admission_path=baseline_release_admission_path,
        candidate_release_admission_path=candidate_release_admission_path,
    )
    if comparisons != recomputed.get("comparisons"):
        raise ValueError("Paired bootstrap comparison values differ from recomputation")


def build_comparison_admission(
    *,
    baseline_release_admission_path: Path,
    candidate_release_admission_path: Path,
    bootstrap_report_path: Path,
    baseline_family: str,
    candidate_family: str,
) -> dict[str, Any]:
    """Bind one formal paired-bootstrap report to its two family admissions."""
    validate_paired_bootstrap_output_path(
        bootstrap_report_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    registered.verify_git_head_file(bootstrap_report_path)
    baseline, baseline_identities = _comparison_input_admission(
        baseline_release_admission_path, family=baseline_family, role="baseline"
    )
    candidate, candidate_identities = _comparison_input_admission(
        candidate_release_admission_path, family=candidate_family, role="candidate"
    )
    bootstrap = _load_json(bootstrap_report_path, description="paired bootstrap report")
    if (
        bootstrap.get("protocol_id") != "v5_osm_assisted"
        or bootstrap.get("preliminary") is not True
        or bootstrap.get("paper_eligible") is not False
        or bootstrap.get("admission_status") != "derived_statistic_pending_external_admission"
    ):
        raise ValueError("Paired bootstrap is not a preliminary V5 derived report")
    if (
        bootstrap.get("baseline_family") != baseline_family
        or bootstrap.get("candidate_family") != candidate_family
        or bootstrap.get("comparison_direction") != "candidate_minus_baseline"
        or bootstrap.get("tasks") != list(aggregate.PRIMARY_TASKS)
        or bootstrap.get("shots") != list(aggregate.PRIMARY_SHOTS)
        or bootstrap.get("n_resamples") != 10000
        or bootstrap.get("random_seed") != 20260725
    ):
        raise ValueError(
            "Paired bootstrap direction or matrix differs from the preregistered protocol"
        )
    report_admissions = bootstrap.get("release_admissions")
    if not isinstance(report_admissions, dict):
        raise ValueError("Paired bootstrap lacks its two release-admission identities")
    for role, expected in (("baseline", baseline), ("candidate", candidate)):
        observed = report_admissions.get(role)
        if not isinstance(observed, dict) or observed != expected:
            raise ValueError(f"Paired bootstrap {role} release-admission differs from input")
    snapshot = bootstrap.get("input_identity_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
        raise ValueError("Paired bootstrap lacks a hash-bound input identity snapshot")
    snapshot_path = Path(snapshot["path"])
    expected_snapshot = _validate_comparison_snapshot(
        snapshot_path,
        baseline_identities=baseline_identities,
        candidate_identities=candidate_identities,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    if snapshot != expected_snapshot:
        raise ValueError("Paired bootstrap input identity snapshot differs from report")
    _verify_bootstrap_comparisons(
        bootstrap,
        baseline_release_admission_path=baseline_release_admission_path,
        candidate_release_admission_path=candidate_release_admission_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    return {
        "schema_version": 1,
        "protocol_id": "v5_osm_assisted",
        "paper_eligible": True,
        "admission_status": "paired_bootstrap_external_admission_verified",
        "baseline_family": baseline_family,
        "candidate_family": candidate_family,
        "baseline_release_admission": baseline,
        "candidate_release_admission": candidate,
        "bootstrap_report": {
            "path": str(bootstrap_report_path.resolve()),
            "sha256": registered.sha256_file(bootstrap_report_path),
        },
        "input_identity_snapshot": expected_snapshot,
    }


def write_comparison_admission(
    *,
    output_path: Path,
    baseline_release_admission_path: Path,
    candidate_release_admission_path: Path,
    bootstrap_report_path: Path,
    baseline_family: str,
    candidate_family: str,
) -> dict[str, Any]:
    """Write one paired-comparison admission once; existing evidence is immutable."""
    payload = build_comparison_admission(
        baseline_release_admission_path=baseline_release_admission_path,
        candidate_release_admission_path=candidate_release_admission_path,
        bootstrap_report_path=bootstrap_report_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Comparison admission exists; refusing to overwrite: {output_path}"
        ) from exc
    return payload


def load_admitted_comparison(
    comparison_admission_path: Path,
    *,
    baseline_family: str,
    candidate_family: str,
) -> dict[str, Any]:
    """Load a paper-facing paired comparison only through its Git-sealed admission."""
    validate_comparison_admission_output_path(
        comparison_admission_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    registered.verify_git_head_file(comparison_admission_path)
    payload = _load_json(comparison_admission_path, description="comparison admission")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != "v5_osm_assisted"
        or payload.get("paper_eligible") is not True
        or payload.get("admission_status") != "paired_bootstrap_external_admission_verified"
        or payload.get("baseline_family") != baseline_family
        or payload.get("candidate_family") != candidate_family
    ):
        raise ValueError("Comparison admission does not authorize the requested paper comparison")
    report = payload.get("bootstrap_report")
    if not isinstance(report, dict) or not isinstance(report.get("path"), str):
        raise ValueError("Comparison admission lacks its bootstrap report identity")
    if report.get("sha256") != registered.sha256_file(Path(report["path"])):
        raise ValueError("Comparison admission bootstrap report hash differs from current bytes")
    bootstrap_report_path = Path(report["path"])
    validate_paired_bootstrap_output_path(
        bootstrap_report_path,
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    registered.verify_git_head_file(bootstrap_report_path)
    input_identities: dict[str, list[dict[str, str]]] = {}
    for role, family in (("baseline", baseline_family), ("candidate", candidate_family)):
        release = payload.get(f"{role}_release_admission")
        if not isinstance(release, dict) or not isinstance(release.get("path"), str):
            raise ValueError(f"Comparison admission lacks its {role} release-admission")
        verified_release, identities = _comparison_input_admission(
            Path(release["path"]), family=family, role=role
        )
        if release != verified_release:
            raise ValueError(
                f"Comparison admission {role} release-admission differs from sealed input"
            )
        input_identities[role] = identities
    snapshot = payload.get("input_identity_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("path"), str):
        raise ValueError("Comparison admission lacks its input identity snapshot")
    verified_snapshot = _validate_comparison_snapshot(
        Path(snapshot["path"]),
        baseline_identities=input_identities["baseline"],
        candidate_identities=input_identities["candidate"],
        baseline_family=baseline_family,
        candidate_family=candidate_family,
    )
    if snapshot != verified_snapshot:
        raise ValueError("Comparison admission input identity snapshot differs from sealed inputs")
    bootstrap = _load_json(bootstrap_report_path, description="paired bootstrap report")
    if (
        bootstrap.get("release_admissions")
        != {
            "baseline": payload["baseline_release_admission"],
            "candidate": payload["candidate_release_admission"],
        }
        or bootstrap.get("input_identity_snapshot") != verified_snapshot
    ):
        raise ValueError("Comparison admission bootstrap bindings differ from sealed inputs")
    return payload


def validate_comparison_admission_output_path(
    output_path: Path, *, baseline_family: str, candidate_family: str
) -> None:
    """Reserve one repository path for every ordered V5 family comparison."""
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        repo_root
        / "configs"
        / "eval"
        / "release_admissions"
        / f"rse_v5_osm_assisted_{baseline_family}_vs_{candidate_family}_bootstrap.json"
    )
    if output_path.resolve() != expected.resolve():
        raise ValueError(
            f"Comparison admission must use canonical comparison-admission path: {expected}"
        )


def validate_paired_bootstrap_output_path(
    output_path: Path, *, baseline_family: str, candidate_family: str
) -> None:
    """Reserve one Git-tracked source path for the paired bootstrap report."""
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


def validate_paired_bootstrap_snapshot_path(
    snapshot_path: Path, *, baseline_family: str, candidate_family: str
) -> None:
    """Reserve the Git-tracked input identity snapshot beside its paired bootstrap."""
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        repo_root
        / "configs"
        / "eval"
        / "release_admissions"
        / f"rse_v5_osm_assisted_{baseline_family}_vs_{candidate_family}_paired_bootstrap"
        "_input_identity_snapshot.json"
    )
    if snapshot_path.resolve() != expected.resolve():
        raise ValueError(
            "Paired bootstrap snapshot must use canonical paired-bootstrap snapshot path: "
            f"{expected}"
        )


def validate_release_admission_output_path(output_path: Path, family: str) -> None:
    """Reserve one repository path for each protocol/family admission decision."""
    repo_root = Path(__file__).resolve().parents[2]
    expected = (
        repo_root / "configs" / "eval" / "release_admissions" / f"rse_v5_osm_assisted_{family}.json"
    )
    if output_path.resolve() != expected.resolve():
        raise ValueError(f"Release admission must use canonical release-admission path: {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--release-anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--aggregate-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_release_admission_output_path(args.output, args.family)
    payload = write_release_admission(
        args.registry,
        args.release_anchor,
        args.output,
        family=args.family,
        aggregate_report_path=args.aggregate_report,
    )
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "result_count": len(payload["selected_results"])}
        )
    )


if __name__ == "__main__":
    main()
