from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.data import build_paper_manifests, compute_statistics
from scripts.eval import register_registered_v5_embedding_export as registrar
from scripts.eval import run_registered_paper_downstream as registered
from scripts.experiments import build_clean_paper_configs, build_paper_subset_registry
from scripts.report import admit_registered_comparison as comparison_admission
from scripts.report import admit_registered_results as admission
from scripts.report import aggregate_registered_paper_results as aggregate
from scripts.report import paired_spatial_bootstrap as bootstrap
from scripts.train import train


def _result(protocol_id: str) -> dict[str, object]:
    evidence = registered.result_evidence(protocol_id)
    return {
        "label_sha256": "labels",
        "metric_provenance": {
            **evidence,
            "task": "building",
            "shot": "5",
            "fold": 0,
            "shot_seed": 42,
            "spatial_split_sha256": "split",
            "manifest_sha256": "manifest",
            "shot_manifest_sha256": "shots",
            "shot_manifest": {"label_sha256": "labels", "split_sha256": "split"},
            "per_patch_confusion": {
                "patch-a": {"tp": 1, "fp": 0, "fn": 0, "tn": 1},
            },
        },
    }


def test_v5_osm_assisted_result_evidence_is_explicit_and_not_independent_transfer() -> None:
    evidence = registered.result_evidence("v5_osm_assisted")

    assert evidence == {
        "protocol_id": "v5_osm_assisted",
        "evidence_class": "osm_assisted_spatial_readout",
        "label_independence_status": "osm_overlapping_not_independent",
    }
    with pytest.raises(ValueError, match="independent-transfer"):
        aggregate.validate_report_evidence([_result("v5_osm_assisted")], "independent_transfer")


def test_result_evidence_rejects_a_record_without_protocol_id() -> None:
    with pytest.raises(ValueError, match="protocol_id"):
        registered.validate_result_evidence(
            {
                "evidence_class": "osm_assisted_spatial_readout",
                "label_independence_status": "osm_overlapping_not_independent",
            }
        )


def test_paired_bootstrap_rejects_mixed_protocol_ids() -> None:
    baseline = {("building", "5", 0, 42): _result("v4_diagnostic")}
    candidate = {("building", "5", 0, 42): _result("v5_osm_assisted")}

    with pytest.raises(ValueError, match="protocol_id"):
        bootstrap.verify_paired_record_provenance(baseline, candidate)


def test_aggregation_normalizes_pre_descriptor_results_to_v4_diagnostic() -> None:
    legacy_record = {
        "metric_provenance": {
            "task": "building",
            "shot": "5",
            "fold": 0,
            "shot_seed": 42,
        }
    }

    assert aggregate.validate_report_evidence([legacy_record], "diagnostic") == {
        "protocol_id": "v4_diagnostic",
        "evidence_class": "diagnostic_spatial_readout",
        "label_independence_status": "not_independent_transfer_evidence",
    }


def test_protocol_bearing_metric_requires_complete_artifact_evidence(tmp_path) -> None:
    output = tmp_path / "probe"
    output.mkdir()
    predictions = output / "predictions_test.npz"
    predictions.write_bytes(b"predictions")
    (output / "final_probe.pt").write_bytes(b"probe")
    metric_payload = {
        "paper_eligible": False,
        "admission_status": "registered_preliminary_pending_external_gates",
        **registered.result_evidence("v5_osm_assisted"),
    }
    (output / "metrics.json").write_text(json.dumps(metric_payload), encoding="utf-8")
    artifact = registered.build_artifact_manifest(
        metric_payload=metric_payload,
        output=output,
        predictions=predictions,
        result_id="result-v5",
        registry_path=tmp_path / "results.jsonl",
        label_sha256="labels",
        patch_count=1,
    )

    assert {key: artifact[key] for key in registered.result_evidence("v5_osm_assisted")} == (
        registered.result_evidence("v5_osm_assisted")
    )
    incomplete_artifact = dict(artifact)
    for key in registered.result_evidence("v5_osm_assisted"):
        incomplete_artifact.pop(key)
    with pytest.raises(ValueError, match="evidence"):
        registered.registry_entry_core_sha256("result-v5", incomplete_artifact)

    artifact_path = output / "artifact_manifest.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    registry_record = {
        "result_id": "result-v5",
        "artifact_sha256": registered.sha256_file(artifact_path),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    registry_record.pop("label_independence_status")
    registry_record["registry_entry_sha256"] = registered._canonical_sha256(registry_record)
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text(json.dumps(registry_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        registered.verify_artifact_registry_binding(artifact_path, registry_path)


def test_v5_report_outputs_include_evidence_classification(tmp_path) -> None:
    report = {
        "family": "full_150",
        "preliminary": True,
        "primary_matrix": True,
        "registry_sha256": "registry",
        "selected_results_sha256": "selected",
        "report_kind": "weak_supervision",
        **registered.result_evidence("v5_osm_assisted"),
        "summary": {
            "cells": {
                "building|5": {
                    "task": "building",
                    "shot": "5",
                    "metrics": {
                        metric: {
                            "mean": 0.5,
                            "std": 0.1,
                            "n_seeds": 3,
                            "pooled_fold_mean": 0.5,
                            "n_fold_seed_runs": 15,
                        }
                        for metric in aggregate.METRICS
                    },
                }
            }
        },
    }
    csv_path = tmp_path / "report.csv"
    markdown_path = tmp_path / "report.md"

    aggregate._write_csv(csv_path, report)
    aggregate._write_markdown(markdown_path, report)

    csv_text = csv_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    for value in (
        "report_kind",
        "protocol_id",
        "evidence_class",
        "label_independence_status",
        "weak_supervision",
        "v5_osm_assisted",
        "osm_assisted_spatial_readout",
        "osm_overlapping_not_independent",
    ):
        assert value in csv_text
        assert value in markdown_text


def test_legacy_aggregation_cli_defaults_to_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pre-v5 aggregation invocations compatible with the diagnostic evidence class."""
    from scripts.report import aggregate_registered_paper_results as aggregate

    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_registered_paper_results.py",
            "--registry",
            "results.jsonl",
            "--family",
            "family",
            "--output",
            "report.json",
        ],
    )
    assert aggregate.parse_args().report_kind == "diagnostic"


def test_release_admission_preserves_preliminary_record_bytes_and_binds_hashes(
    tmp_path: Path,
) -> None:
    """A release decision is an overlay; sealed preliminary records stay immutable."""
    registry_path = tmp_path / "results.jsonl"
    output = tmp_path / "probe"
    output.mkdir()
    predictions = output / "predictions_test.npz"
    predictions.write_bytes(b"predictions")
    (output / "final_probe.pt").write_bytes(b"probe")
    metrics = {
        "paper_eligible": False,
        "admission_status": "registered_preliminary_pending_external_gates",
        **registered.result_evidence("v5_osm_assisted"),
    }
    _write_json(output / "metrics.json", metrics)
    artifact = registered.build_artifact_manifest(
        metric_payload=metrics,
        output=output,
        predictions=predictions,
        result_id="v5-result",
        registry_path=registry_path,
        label_sha256="labels",
        patch_count=1,
    )
    artifact_path = output / "artifact_manifest.json"
    _write_json(artifact_path, artifact)
    registry_record = {
        "result_id": "v5-result",
        "artifact_sha256": _sha256(artifact_path),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    registry_record["registry_entry_sha256"] = registered._canonical_sha256(registry_record)
    registry_path.write_text(json.dumps(registry_record) + "\n", encoding="utf-8")
    original_registry = registry_path.read_bytes()
    original_metrics = (output / "metrics.json").read_bytes()
    original_artifact = artifact_path.read_bytes()

    identities = admission.collect_sealed_result_identities(registry_path)

    assert identities == [
        {
            "result_id": "v5-result",
            "registry_entry_sha256": registry_record["registry_entry_sha256"],
            "artifact_sha256": _sha256(artifact_path),
            "metrics_sha256": _sha256(output / "metrics.json"),
        }
    ]
    assert registry_path.read_bytes() == original_registry
    assert (output / "metrics.json").read_bytes() == original_metrics
    assert artifact_path.read_bytes() == original_artifact


@pytest.mark.parametrize("tampered", ("registry", "metrics", "artifact"))
def test_release_admission_rejects_tampered_sealed_result(tmp_path: Path, tampered: str) -> None:
    """Every result identity must pass the existing full sealed-artifact verifier."""
    registry_path = tmp_path / "results.jsonl"
    output = tmp_path / "probe"
    output.mkdir()
    predictions = output / "predictions_test.npz"
    predictions.write_bytes(b"predictions")
    (output / "final_probe.pt").write_bytes(b"probe")
    metrics = {
        "paper_eligible": False,
        "admission_status": "registered_preliminary_pending_external_gates",
        **registered.result_evidence("v5_osm_assisted"),
    }
    _write_json(output / "metrics.json", metrics)
    artifact = registered.build_artifact_manifest(
        metric_payload=metrics,
        output=output,
        predictions=predictions,
        result_id="v5-result",
        registry_path=registry_path,
        label_sha256="labels",
        patch_count=1,
    )
    artifact_path = output / "artifact_manifest.json"
    _write_json(artifact_path, artifact)
    record = {
        "result_id": "v5-result",
        "artifact_sha256": _sha256(artifact_path),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    record["registry_entry_sha256"] = registered._canonical_sha256(record)
    registry_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    if tampered == "registry":
        record["registry_entry_sha256"] = "forged"
        registry_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    elif tampered == "metrics":
        _write_json(output / "metrics.json", {**metrics, "tampered": True})
    else:
        _write_json(artifact_path, {**artifact, "patch_count": 2})

    with pytest.raises(ValueError):
        admission.collect_sealed_result_identities(registry_path)


def test_release_anchor_requires_doi_bound_registry_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Git anchor must bind the exact bytes published in its Zenodo record."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    anchor = {
        "schema_version": 1,
        "protocol_id": "v5_osm_assisted",
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
        "registry_url": "https://zenodo.org/records/1234567/files/results.jsonl",
    }
    _write_json(anchor_path, anchor)
    monkeypatch.setattr(registered, "verify_git_head_file", lambda path: None)
    monkeypatch.setattr(
        admission,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(registry_path.read_bytes()),
        raising=False,
    )

    verified = admission.verify_release_anchor(registry_path, anchor_path)

    assert verified["registry_sha256"] == _sha256(registry_path)
    assert verified["sha256"] == _sha256(anchor_path)


def test_release_anchor_rejects_wrong_external_registry_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DOI-shaped URL is insufficient when the published registry bytes differ."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    _write_json(
        anchor_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": _sha256(registry_path),
            "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
            "registry_url": "https://zenodo.org/records/1234567/files/results.jsonl",
        },
    )
    monkeypatch.setattr(registered, "verify_git_head_file", lambda path: None)
    monkeypatch.setattr(admission, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))

    with pytest.raises(ValueError, match="External registry archive hash"):
        admission.verify_release_anchor(registry_path, anchor_path)


def test_release_anchor_rejects_unexpected_hash_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anchor-controlled fields must not override computed release identity hashes."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    _write_json(
        anchor_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": _sha256(registry_path),
            "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
            "registry_url": "https://zenodo.org/records/1234567/files/results.jsonl",
            "sha256": "forged",
        },
    )
    monkeypatch.setattr(registered, "verify_git_head_file", lambda path: None)
    monkeypatch.setattr(
        admission, "urlopen", lambda *args, **kwargs: io.BytesIO(registry_path.read_bytes())
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        admission.verify_release_anchor(registry_path, anchor_path)


def test_release_anchor_rejects_untracked_file_before_network_fetch(tmp_path: Path) -> None:
    """A release anchor must be a tracked, byte-identical Git HEAD artifact."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    _write_json(anchor_path, {"schema_version": 1})

    with pytest.raises(ValueError, match="Git repository"):
        admission.verify_release_anchor(registry_path, anchor_path)


def test_release_anchor_rejects_worktree_bytes_that_differ_from_git_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tracked anchor is still rejected after any uncommitted byte-level change."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "tests@example.invalid"),
        ("git", "config", "user.name", "Tests"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    registry_path = repo / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = repo / "anchor.json"
    _write_json(
        anchor_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "registry_path": str(registry_path.resolve()),
            "registry_sha256": _sha256(registry_path),
            "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
            "registry_url": "https://zenodo.org/records/1234567/files/results.jsonl",
        },
    )
    subprocess.run(("git", "add", "."), cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ("git", "commit", "-m", "seal anchor"),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    anchor_path.write_text(anchor_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(admission, "urlopen", lambda *args, **kwargs: io.BytesIO(b"unused"))

    with pytest.raises(ValueError, match="current Git HEAD"):
        admission.verify_release_anchor(registry_path, anchor_path)


def test_build_release_admission_creates_an_immutable_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission records identify sealed results without changing their preliminary status."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text("{}\n", encoding="utf-8")
    identities = [
        {
            "result_id": "sealed",
            "registry_entry_sha256": "entry",
            "artifact_sha256": "artifact",
            "metrics_sha256": "metrics",
        }
    ]
    monkeypatch.setattr(
        admission,
        "verify_release_anchor",
        lambda registry, anchor: {
            "path": str(anchor.resolve()),
            "sha256": _sha256(anchor),
            "registry_sha256": _sha256(registry),
        },
    )
    monkeypatch.setattr(
        admission, "collect_sealed_result_identities", lambda registry, **kwargs: identities
    )
    monkeypatch.setattr(admission.aggregate, "load_verified_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(admission, "validate_complete_v5_family_records", lambda *args: None)
    monkeypatch.setattr(admission.aggregate, "summarize_records", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        admission,
        "verify_preliminary_aggregate_report",
        lambda *args, **kwargs: {"path": "aggregate.json", "sha256": "aggregate"},
    )

    record = admission.build_release_admission(
        registry_path,
        anchor_path,
        family="full_150",
        aggregate_report_path=tmp_path / "aggregate.json",
    )

    assert record["paper_eligible"] is True
    assert record["admission_status"] == "external_release_anchor_verified"
    assert record["selected_results"] == identities
    assert record["selected_results_sha256"] == registered._canonical_sha256(
        {"selected_results": identities}
    )
    assert registry_path.read_text(encoding="utf-8") == '{"result_id":"sealed"}\n'


def test_write_release_admission_refuses_to_overwrite_an_existing_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission output is write-once so its future Git identity stays auditable."""
    output_path = tmp_path / "admission.json"
    output_path.write_text('{"old":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        admission, "build_release_admission", lambda registry, anchor, **kwargs: {"new": True}
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        admission.write_release_admission(
            tmp_path / "results.jsonl",
            tmp_path / "anchor.json",
            output_path,
            family="full_150",
            aggregate_report_path=tmp_path / "aggregate.json",
        )


def test_release_admission_rejects_an_incomplete_v5_probe_matrix() -> None:
    """No partial fold/task/shot/seed matrix may receive paper-eligible status."""
    with pytest.raises(ValueError, match="V5 probe matrix is incomplete"):
        admission.validate_complete_v5_family_records([], "full_150")


def test_build_release_admission_rejects_partial_family_before_marking_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer must validate all 90 registered cells before creating an overlay."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text("\n", encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(admission.aggregate, "load_verified_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        admission,
        "verify_release_anchor",
        lambda registry, anchor: {"registry_sha256": _sha256(registry)},
    )

    with pytest.raises(ValueError, match="V5 probe matrix is incomplete"):
        admission.build_release_admission(
            registry_path,
            anchor_path,
            family="full_150",
            aggregate_report_path=tmp_path / "aggregate.json",
        )


def test_release_admission_rejects_aggregate_report_for_another_result_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release overlay must bind the exact preliminary aggregate it later cites."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text('{"result_id":"sealed"}\n', encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "aggregate.json"
    _write_json(
        report_path,
        {
            "family": "full_150",
            "preliminary": True,
            "protocol_id": "v5_osm_assisted",
            "primary_matrix": True,
            "report_kind": "weak_supervision",
            **registered.result_evidence("v5_osm_assisted"),
            "registry": str(registry_path.resolve()),
            "registry_sha256": _sha256(registry_path),
            "selected_results_sha256": "wrong",
            "record_count": 0,
            "summary": {},
        },
    )
    identities = [
        {
            "result_id": "sealed",
            "registry_entry_sha256": "entry",
            "artifact_sha256": "artifact",
            "metrics_sha256": "metrics",
        }
    ]
    monkeypatch.setattr(
        admission,
        "verify_release_anchor",
        lambda registry, anchor: {"registry_sha256": _sha256(registry)},
    )
    monkeypatch.setattr(admission.aggregate, "load_verified_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(admission, "validate_complete_v5_family_records", lambda *args: None)
    monkeypatch.setattr(admission.aggregate, "summarize_records", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        admission, "collect_sealed_result_identities", lambda registry, **kwargs: identities
    )

    with pytest.raises(ValueError, match="Aggregate selected-result identity"):
        admission.build_release_admission(
            registry_path, anchor_path, family="full_150", aggregate_report_path=report_path
        )


def test_release_admission_rejects_a_forged_aggregate_summary(tmp_path: Path) -> None:
    """Matching identity fields do not admit a report with fabricated metrics."""
    report_path = tmp_path / "aggregate.json"
    _write_json(
        report_path,
        {
            "family": "full_150",
            "preliminary": True,
            "primary_matrix": True,
            "report_kind": "weak_supervision",
            "protocol_id": "v5_osm_assisted",
            **registered.result_evidence("v5_osm_assisted"),
            "registry": "/tmp/results.jsonl",
            "registry_sha256": "registry",
            "selected_results_sha256": "selected",
            "record_count": 90,
            "summary": {"forged": True},
        },
    )

    with pytest.raises(ValueError, match="Aggregate summary differs"):
        admission.verify_preliminary_aggregate_report(
            report_path,
            registry_path=Path("/tmp/results.jsonl"),
            registry_sha256="registry",
            family="full_150",
            selected_results_sha256="selected",
            expected_summary={"real": True},
            expected_record_count=90,
        )


def test_release_admission_cli_path_is_pinned_to_one_protocol_family_location(
    tmp_path: Path,
) -> None:
    """A second arbitrary output path cannot become a competing formal admission."""
    with pytest.raises(ValueError, match="canonical release-admission path"):
        admission.validate_release_admission_output_path(tmp_path / "other.json", "full_150")


def test_formal_aggregation_requires_an_immutable_release_admission(tmp_path: Path) -> None:
    """Raw preliminary records cannot enter paper aggregation without its overlay."""
    with pytest.raises(ValueError, match="requires --release-admission"):
        aggregate.load_admitted_records(
            tmp_path / "results.jsonl", family="full_150", release_admission_path=None
        )


def test_aggregation_selected_result_identity_includes_the_metrics_hash() -> None:
    """Admission and aggregation share one four-hash identity contract."""
    assert aggregate.sealed_result_identity(
        {
            "result_id": "result",
            "artifact_sha256": "artifact",
            "registry_entry_sha256": "entry",
            "metrics_sha256": "metrics",
        }
    ) == {
        "result_id": "result",
        "artifact_sha256": "artifact",
        "registry_entry_sha256": "entry",
        "metrics_sha256": "metrics",
    }


def test_release_admission_rejects_a_changed_bound_aggregate_report() -> None:
    """The report bytes named by an admission are part of its sealed evidence set."""
    with pytest.raises(ValueError, match="Aggregate report hash differs"):
        admission.verify_hash_bound_report(
            {"path": "/tmp/report.json", "sha256": "sealed"},
            {"path": "/tmp/report.json", "sha256": "changed"},
            "Aggregate report",
        )


def test_load_admitted_comparison_requires_a_git_sealed_matching_bootstrap_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paper prose may only load the exact admitted paired-bootstrap bytes."""
    admission_path = tmp_path / "comparison_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    baseline_release = {"path": str(tmp_path / "baseline.json"), "sha256": "baseline"}
    candidate_release = {"path": str(tmp_path / "candidate.json"), "sha256": "candidate"}
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_payload = bootstrap.build_input_identity_snapshot([], [])
    _write_json(snapshot_path, snapshot_payload)
    snapshot = {
        "path": str(snapshot_path),
        "sha256": _sha256(snapshot_path),
        "identity_sha256": snapshot_payload["sha256"],
        "baseline_result_count": snapshot_payload["baseline_result_count"],
        "candidate_result_count": snapshot_payload["candidate_result_count"],
    }
    _write_json(
        bootstrap_path,
        {
            "comparisons": {},
            "release_admissions": {
                "baseline": baseline_release,
                "candidate": candidate_release,
            },
            "input_identity_snapshot": snapshot,
        },
    )
    payload = {
        "schema_version": 1,
        "protocol_id": "v5_osm_assisted",
        "paper_eligible": True,
        "admission_status": "paired_bootstrap_external_admission_verified",
        "baseline_family": "aef_v5",
        "candidate_family": "full_150",
        "bootstrap_report": {
            "path": str(bootstrap_path),
            "sha256": _sha256(bootstrap_path),
        },
        "baseline_release_admission": baseline_release,
        "candidate_release_admission": candidate_release,
        "input_identity_snapshot": snapshot,
    }
    admission_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        admission, "validate_comparison_admission_output_path", lambda *_args, **_kwargs: None
    )
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(registered, "verify_git_head_file", lambda _path: None)
    monkeypatch.setattr(
        admission,
        "_comparison_input_admission",
        lambda path, *, family, role: (
            baseline_release if role == "baseline" else candidate_release,
            [],
        ),
    )

    assert admission.load_admitted_comparison(
        admission_path, baseline_family="aef_v5", candidate_family="full_150"
    ) == payload


def test_load_admitted_comparison_revalidates_the_two_release_admissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed comparison shell cannot bypass its two family evidence chains."""
    admission_path = tmp_path / "comparison_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text('{"comparisons":{}}\n', encoding="utf-8")
    _write_json(
        admission_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "admission_status": "paired_bootstrap_external_admission_verified",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "bootstrap_report": {
                "path": str(bootstrap_path),
                "sha256": _sha256(bootstrap_path),
            },
        },
    )
    monkeypatch.setattr(
        admission, "validate_comparison_admission_output_path", lambda *_args, **_kwargs: None
    )
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(registered, "verify_git_head_file", lambda _path: None)

    with pytest.raises(ValueError, match="release-admission"):
        admission.load_admitted_comparison(
            admission_path, baseline_family="aef_v5", candidate_family="full_150"
        )


def test_load_admitted_comparison_revalidates_the_input_identity_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap and comparison cannot share an arbitrary stale input snapshot."""
    admission_path = tmp_path / "comparison_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    baseline_release = {"path": str(tmp_path / "baseline.json"), "sha256": "baseline"}
    candidate_release = {"path": str(tmp_path / "candidate.json"), "sha256": "candidate"}
    stale_snapshot_path = tmp_path / "stale_snapshot.json"
    _write_json(stale_snapshot_path, {"schema_version": 1, "wrong": True})
    stale_snapshot = {"path": str(stale_snapshot_path), "sha256": _sha256(stale_snapshot_path)}
    _write_json(
        bootstrap_path,
        {
            "release_admissions": {
                "baseline": baseline_release,
                "candidate": candidate_release,
            },
            "input_identity_snapshot": stale_snapshot,
        },
    )
    _write_json(
        admission_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "admission_status": "paired_bootstrap_external_admission_verified",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "baseline_release_admission": baseline_release,
            "candidate_release_admission": candidate_release,
            "bootstrap_report": {
                "path": str(bootstrap_path),
                "sha256": _sha256(bootstrap_path),
            },
            "input_identity_snapshot": stale_snapshot,
        },
    )
    monkeypatch.setattr(
        admission, "validate_comparison_admission_output_path", lambda *_args, **_kwargs: None
    )
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(registered, "verify_git_head_file", lambda _path: None)
    monkeypatch.setattr(
        admission,
        "_comparison_input_admission",
        lambda path, *, family, role: (
            baseline_release if role == "baseline" else candidate_release,
            [{"result_id": role}],
        ),
    )

    with pytest.raises(ValueError, match="input identit"):
        admission.load_admitted_comparison(
            admission_path, baseline_family="aef_v5", candidate_family="full_150"
        )


def test_load_admitted_records_rejects_a_noncanonical_release_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Formal consumers have one canonical release-admission path per family."""
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(registered, "verify_git_head_file", lambda _path: None)

    with pytest.raises(ValueError, match="canonical release-admission path"):
        admission.load_admitted_records(
            registry_path,
            family="full_150",
            release_admission_path=tmp_path / "competing_admission.json",
        )


def test_comparison_admission_binds_two_family_admissions_and_bootstrap_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one immutable record may authorize a paired paper comparison."""
    baseline_path = tmp_path / "baseline_admission.json"
    candidate_path = tmp_path / "candidate_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    snapshot_path = tmp_path / "bootstrap_snapshot.json"
    patch_metadata_path = tmp_path / "patches.json"
    patch_metadata_path.write_text("[]\n", encoding="utf-8")
    baseline_results = [
        {
            "result_id": "aef-result",
            "artifact_sha256": "aef-artifact",
            "registry_entry_sha256": "aef-entry",
            "metrics_sha256": "aef-metrics",
        }
    ]
    candidate_results = [
        {
            "result_id": "xuannv-result",
            "artifact_sha256": "xuannv-artifact",
            "registry_entry_sha256": "xuannv-entry",
            "metrics_sha256": "xuannv-metrics",
        }
    ]
    snapshot_payload = bootstrap.build_input_identity_snapshot(baseline_results, candidate_results)
    _write_json(snapshot_path, snapshot_payload)
    _write_json(
        baseline_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "aef_v5",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(
        candidate_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "full_150",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(
        bootstrap_path,
        {
            "protocol_id": "v5_osm_assisted",
            "preliminary": True,
            "paper_eligible": False,
            "admission_status": "derived_statistic_pending_external_admission",
            "comparison_direction": "candidate_minus_baseline",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "tasks": ["building", "road", "water"],
            "shots": ["5", "10"],
            "n_resamples": 10000,
            "random_seed": 20260725,
            "source_registry_path": str((tmp_path / "results.jsonl").resolve()),
            "patch_metadata_path": str(patch_metadata_path.resolve()),
            "patch_metadata_sha256": _sha256(patch_metadata_path),
            "comparisons": {"building|5": {"f1": "verified"}},
            "release_admissions": {
                "baseline": {
                    "path": str(baseline_path.resolve()),
                    "sha256": _sha256(baseline_path),
                },
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            },
            "input_identity_snapshot": _bootstrap_snapshot_reference(
                snapshot_path, snapshot_payload
            ),
        },
    )
    monkeypatch.setattr(admission.registered, "verify_git_head_file", lambda path: None)
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(
        admission,
        "_load_json",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        admission,
        "load_admitted_records",
        lambda _registry, *, family, **_kwargs: (
            baseline_results if family == "aef_v5" else candidate_results
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "compare_families",
        lambda **_kwargs: {"comparisons": {"building|5": {"f1": "verified"}}},
    )

    payload = admission.build_comparison_admission(
        baseline_release_admission_path=baseline_path,
        candidate_release_admission_path=candidate_path,
        bootstrap_report_path=bootstrap_path,
        baseline_family="aef_v5",
        candidate_family="full_150",
    )

    assert payload["paper_eligible"] is True
    assert payload["admission_status"] == "paired_bootstrap_external_admission_verified"
    assert payload["bootstrap_report"]["sha256"] == _sha256(bootstrap_path)


def test_comparison_admission_rejects_a_forged_bootstrap_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching input snapshot cannot substitute for recomputed CI values."""
    baseline_path = tmp_path / "baseline_admission.json"
    candidate_path = tmp_path / "candidate_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    snapshot_path = tmp_path / "snapshot.json"
    patch_metadata_path = tmp_path / "patches.json"
    patch_metadata_path.write_text("[]\n", encoding="utf-8")
    baseline_results = [
        {
            "result_id": "aef-result",
            "artifact_sha256": "aef-artifact",
            "registry_entry_sha256": "aef-entry",
            "metrics_sha256": "aef-metrics",
        }
    ]
    candidate_results = [
        {
            "result_id": "xuannv-result",
            "artifact_sha256": "xuannv-artifact",
            "registry_entry_sha256": "xuannv-entry",
            "metrics_sha256": "xuannv-metrics",
        }
    ]
    snapshot_payload = bootstrap.build_input_identity_snapshot(baseline_results, candidate_results)
    _write_json(snapshot_path, snapshot_payload)
    for path, family in ((baseline_path, "aef_v5"), (candidate_path, "full_150")):
        _write_json(
            path,
            {
                "schema_version": 1,
                "protocol_id": "v5_osm_assisted",
                "paper_eligible": True,
                "family": family,
                "registry_path": str((tmp_path / "results.jsonl").resolve()),
            },
        )
    _write_json(
        bootstrap_path,
        {
            "protocol_id": "v5_osm_assisted",
            "preliminary": True,
            "paper_eligible": False,
            "admission_status": "derived_statistic_pending_external_admission",
            "comparison_direction": "candidate_minus_baseline",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "tasks": ["building", "road", "water"],
            "shots": ["5", "10"],
            "n_resamples": 10000,
            "random_seed": 20260725,
            "source_registry_path": str((tmp_path / "results.jsonl").resolve()),
            "patch_metadata_path": str(patch_metadata_path.resolve()),
            "patch_metadata_sha256": _sha256(patch_metadata_path),
            "release_admissions": {
                "baseline": {
                    "path": str(baseline_path.resolve()),
                    "sha256": _sha256(baseline_path),
                },
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            },
            "input_identity_snapshot": _bootstrap_snapshot_reference(
                snapshot_path, snapshot_payload
            ),
            "comparisons": {"building|5": {"f1": "forged"}},
        },
    )
    monkeypatch.setattr(admission.registered, "verify_git_head_file", lambda path: None)
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(
        admission,
        "load_admitted_records",
        lambda _registry, *, family, **_kwargs: (
            baseline_results if family == "aef_v5" else candidate_results
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "compare_families",
        lambda **_kwargs: {"comparisons": {"building|5": {"f1": "verified"}}},
    )

    with pytest.raises(ValueError, match="comparison values differ"):
        admission.build_comparison_admission(
            baseline_release_admission_path=baseline_path,
            candidate_release_admission_path=candidate_path,
            bootstrap_report_path=bootstrap_path,
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_comparison_admission_rejects_a_reversed_comparison_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admitted interval sign must remain candidate minus baseline."""
    baseline_path = tmp_path / "baseline_admission.json"
    candidate_path = tmp_path / "candidate_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    snapshot_path = tmp_path / "snapshot.json"
    records = {
        "aef_v5": [
            {
                "result_id": "aef-result",
                "artifact_sha256": "aef-artifact",
                "registry_entry_sha256": "aef-entry",
                "metrics_sha256": "aef-metrics",
            }
        ],
        "full_150": [
            {
                "result_id": "xuannv-result",
                "artifact_sha256": "xuannv-artifact",
                "registry_entry_sha256": "xuannv-entry",
                "metrics_sha256": "xuannv-metrics",
            }
        ],
    }
    _write_json(snapshot_path, bootstrap.build_input_identity_snapshot(*records.values()))
    for path, family in ((baseline_path, "aef_v5"), (candidate_path, "full_150")):
        _write_json(
            path,
            {
                "schema_version": 1,
                "protocol_id": "v5_osm_assisted",
                "paper_eligible": True,
                "family": family,
                "registry_path": str((tmp_path / "results.jsonl").resolve()),
            },
        )
    _write_json(
        bootstrap_path,
        {
            "protocol_id": "v5_osm_assisted",
            "preliminary": True,
            "paper_eligible": False,
            "admission_status": "derived_statistic_pending_external_admission",
            "comparison_direction": "baseline_minus_candidate",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "tasks": ["building", "road", "water"],
            "shots": ["5", "10"],
            "n_resamples": 10000,
            "random_seed": 20260725,
            "source_registry_path": str((tmp_path / "results.jsonl").resolve()),
            "patch_metadata_path": str((tmp_path / "patches.json").resolve()),
            "release_admissions": {
                "baseline": {
                    "path": str(baseline_path.resolve()),
                    "sha256": _sha256(baseline_path),
                },
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            },
            "input_identity_snapshot": {
                "path": str(snapshot_path.resolve()),
                "sha256": _sha256(snapshot_path),
            },
            "comparisons": {"building|5": {"f1": "verified"}},
        },
    )
    monkeypatch.setattr(admission.registered, "verify_git_head_file", lambda path: None)
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(
        admission, "load_admitted_records", lambda _registry, *, family, **_kwargs: records[family]
    )

    with pytest.raises(ValueError, match="direction or matrix"):
        admission.build_comparison_admission(
            baseline_release_admission_path=baseline_path,
            candidate_release_admission_path=candidate_path,
            bootstrap_report_path=bootstrap_path,
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_comparison_admission_rejects_bootstrap_with_different_admission_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bootstrap result cannot be relabelled with unrelated family admissions."""
    baseline_path = tmp_path / "baseline_admission.json"
    candidate_path = tmp_path / "candidate_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    _write_json(
        baseline_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "aef_v5",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(
        candidate_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "full_150",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(
        bootstrap_path,
        {
            "protocol_id": "v5_osm_assisted",
            "preliminary": True,
            "paper_eligible": False,
            "admission_status": "derived_statistic_pending_external_admission",
            "comparison_direction": "candidate_minus_baseline",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "tasks": ["building", "road", "water"],
            "shots": ["5", "10"],
            "n_resamples": 10000,
            "random_seed": 20260725,
            "release_admissions": {
                "baseline": {"path": "other", "sha256": "wrong"},
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            },
        },
    )
    monkeypatch.setattr(admission.registered, "verify_git_head_file", lambda path: None)
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(
        admission,
        "load_admitted_records",
        lambda _registry, *, family, **_kwargs: [
            {
                "result_id": f"{family}-result",
                "artifact_sha256": "artifact",
                "registry_entry_sha256": "entry",
                "metrics_sha256": "metrics",
            }
        ],
    )

    with pytest.raises(ValueError, match="baseline release-admission"):
        admission.build_comparison_admission(
            baseline_release_admission_path=baseline_path,
            candidate_release_admission_path=candidate_path,
            bootstrap_report_path=bootstrap_path,
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_comparison_admission_rejects_snapshot_from_another_result_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap snapshot must equal both families' admitted result identities."""
    baseline_path = tmp_path / "baseline_admission.json"
    candidate_path = tmp_path / "candidate_admission.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    snapshot_path = tmp_path / "bootstrap_snapshot.json"
    baseline_results = [
        {
            "result_id": "aef-result",
            "artifact_sha256": "aef-artifact",
            "registry_entry_sha256": "aef-entry",
            "metrics_sha256": "aef-metrics",
        }
    ]
    candidate_results = [
        {
            "result_id": "xuannv-result",
            "artifact_sha256": "xuannv-artifact",
            "registry_entry_sha256": "xuannv-entry",
            "metrics_sha256": "xuannv-metrics",
        }
    ]
    _write_json(
        baseline_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "aef_v5",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(
        candidate_path,
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "paper_eligible": True,
            "family": "full_150",
            "registry_path": str((tmp_path / "results.jsonl").resolve()),
        },
    )
    _write_json(snapshot_path, bootstrap.build_input_identity_snapshot([], candidate_results))
    _write_json(
        bootstrap_path,
        {
            "protocol_id": "v5_osm_assisted",
            "preliminary": True,
            "paper_eligible": False,
            "admission_status": "derived_statistic_pending_external_admission",
            "comparison_direction": "candidate_minus_baseline",
            "baseline_family": "aef_v5",
            "candidate_family": "full_150",
            "tasks": ["building", "road", "water"],
            "shots": ["5", "10"],
            "n_resamples": 10000,
            "random_seed": 20260725,
            "release_admissions": {
                "baseline": {
                    "path": str(baseline_path.resolve()),
                    "sha256": _sha256(baseline_path),
                },
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
            },
            "input_identity_snapshot": {
                "path": str(snapshot_path.resolve()),
                "sha256": _sha256(snapshot_path),
            },
        },
    )
    monkeypatch.setattr(admission.registered, "verify_git_head_file", lambda path: None)
    _allow_test_only_paired_bootstrap_path(monkeypatch)
    monkeypatch.setattr(
        admission,
        "load_admitted_records",
        lambda _registry, *, family, **_kwargs: (
            baseline_results if family == "aef_v5" else candidate_results
        ),
    )

    with pytest.raises(ValueError, match="input identities differ"):
        admission.build_comparison_admission(
            baseline_release_admission_path=baseline_path,
            candidate_release_admission_path=candidate_path,
            bootstrap_report_path=bootstrap_path,
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_write_comparison_admission_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paired-comparison admission is a write-once paper evidence record."""
    output_path = tmp_path / "comparison_admission.json"
    output_path.write_text('{"old":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        admission,
        "build_comparison_admission",
        lambda **_kwargs: {"new": True},
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        admission.write_comparison_admission(
            output_path=output_path,
            baseline_release_admission_path=tmp_path / "baseline.json",
            candidate_release_admission_path=tmp_path / "candidate.json",
            bootstrap_report_path=tmp_path / "bootstrap.json",
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_comparison_admission_cli_exposes_all_sealed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired-bootstrap admission must be reproducible without an ad-hoc Python call."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "admit_registered_comparison.py",
            "--output",
            "comparison.json",
            "--baseline-release-admission",
            "aef.json",
            "--candidate-release-admission",
            "xuannv.json",
            "--bootstrap-report",
            "bootstrap.json",
            "--baseline-family",
            "aef_v5",
            "--candidate-family",
            "full_150",
        ],
    )

    args = comparison_admission.parse_args()

    assert args.baseline_release_admission == Path("aef.json")
    assert args.candidate_release_admission == Path("xuannv.json")
    assert args.bootstrap_report == Path("bootstrap.json")


def test_comparison_admission_output_path_is_unique_per_family_pair(tmp_path: Path) -> None:
    """One family comparison has one canonical write-once admission location."""
    with pytest.raises(ValueError, match="canonical comparison-admission path"):
        admission.validate_comparison_admission_output_path(
            tmp_path / "other.json", baseline_family="aef_v5", candidate_family="full_150"
        )


def test_comparison_admission_rejects_a_noncanonical_paired_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A formal comparison cannot bind an arbitrary bootstrap file outside Git."""
    bootstrap_path = tmp_path / "external_bootstrap.json"
    _write_json(bootstrap_path, {})
    monkeypatch.setattr(
        admission,
        "_comparison_input_admission",
        lambda _path, *, family, role: ({"path": f"/{role}", "sha256": role}, []),
    )

    with pytest.raises(ValueError, match="canonical paired-bootstrap path"):
        admission.build_comparison_admission(
            baseline_release_admission_path=tmp_path / "baseline.json",
            candidate_release_admission_path=tmp_path / "candidate.json",
            bootstrap_report_path=bootstrap_path,
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_comparison_admission_rejects_a_noncanonical_paired_bootstrap_snapshot(
    tmp_path: Path,
) -> None:
    """The bootstrap input identities must be committed beside the bootstrap report."""
    with pytest.raises(ValueError, match="canonical paired-bootstrap snapshot path"):
        admission.validate_paired_bootstrap_snapshot_path(
            tmp_path / "external_snapshot.json",
            baseline_family="aef_v5",
            candidate_family="full_150",
        )


def test_aggregation_cli_accepts_release_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formal aggregation entrypoint exposes the immutable admission input."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_registered_paper_results.py",
            "--registry",
            "results.jsonl",
            "--family",
            "full_150",
            "--output",
            "report.json",
            "--release-admission",
            "admission.json",
        ],
    )
    assert aggregate.parse_args().release_admission == Path("admission.json")


def test_bootstrap_cli_accepts_two_release_admissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A formal comparison binds each encoder family to its own sealed admission."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "paired_spatial_bootstrap.py",
            "--registry",
            "results.jsonl",
            "--baseline-family",
            "aef_v5",
            "--candidate-family",
            "full_150",
            "--output",
            "bootstrap.json",
            "--baseline-release-admission",
            "aef_admission.json",
            "--candidate-release-admission",
            "xuannv_admission.json",
        ],
    )
    args = bootstrap.parse_args()
    assert args.baseline_release_admission == Path("aef_admission.json")
    assert args.candidate_release_admission == Path("xuannv_admission.json")


def test_admitted_records_reject_an_untracked_release_admission(tmp_path: Path) -> None:
    """Formal consumers must reject a competing path before reading its contents."""
    admission_path = tmp_path / "admission.json"
    _write_json(admission_path, {"schema_version": 1})

    with pytest.raises(ValueError, match="canonical release-admission path"):
        admission.load_admitted_records(
            tmp_path / "results.jsonl", family="full_150", release_admission_path=admission_path
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_snapshot_reference(
    snapshot_path: Path, snapshot: dict[str, object]
) -> dict[str, object]:
    return {
        "path": str(snapshot_path.resolve()),
        "sha256": _sha256(snapshot_path),
        "identity_sha256": snapshot["sha256"],
        "baseline_result_count": snapshot["baseline_result_count"],
        "candidate_result_count": snapshot["candidate_result_count"],
    }


def _allow_test_only_paired_bootstrap_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests focused on admission payload checks, not repository layout."""
    monkeypatch.setattr(
        admission, "validate_paired_bootstrap_output_path", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        admission, "validate_paired_bootstrap_snapshot_path", lambda *_args, **_kwargs: None
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _v5_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    split_path = tmp_path / "split.json"
    metadata_path = tmp_path / "patches.json"
    source_manifest_path = tmp_path / "source.json"
    ids = [f"patch_{index:06d}" for index in range(8)]
    _write_json(
        split_path,
        {
            "folds": [
                {
                    "fold": 0,
                    "train": ids[:4],
                    "val": ids[4:5],
                    "test": ids[5:7],
                    "buffer": ids[7:],
                }
            ]
        },
    )
    _write_json(
        metadata_path,
        [
            {"patch_id": patch_id, "bounds": [index, 0, index + 1, 1]}
            for index, patch_id in enumerate(ids)
        ],
    )
    _write_json(source_manifest_path, [{"patch_id": patch_id} for patch_id in ids])
    return split_path, metadata_path, source_manifest_path


def test_v5_subset_registry_rejects_a_different_committed_split(tmp_path: Path) -> None:
    split_path, metadata_path, _ = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    registry["source_split_sha256"] = "not-the-committed-split"

    with pytest.raises(ValueError, match="source split"):
        build_paper_subset_registry.validate_registry_split(registry, split_path)


def test_v5_manifest_sidecar_and_statistics_registry_reject_mismatched_source_hashes(
    tmp_path: Path,
) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    registry_path = tmp_path / "subsets.json"
    _write_json(registry_path, registry)
    output_dir = tmp_path / "manifests"
    audit_path = build_paper_manifests.materialize_fold_manifests(
        source_manifest_path=source_manifest_path,
        spatial_split_path=split_path,
        subset_registry_path=registry_path,
        output_dir=output_dir,
        fold_id=0,
        sizes=(2, 3),
        prefix="paper_v5",
        protocol_name="rse_v5_registered_20260726",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_manifest_sha256"] = "wrong"
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="source manifest"):
        build_paper_manifests.validate_manifest_audit_sidecar(
            audit_path, source_manifest_path, split_path, registry_path
        )

    statistics_registry = {
        "protocol_name": "rse_v5_registered_20260726",
        "source_split_sha256": _sha256(split_path),
        "folds": {
            "0": {
                "source_manifest": str(output_dir / "paper_v5_train_pool_fold0.json"),
                "source_manifest_sha256": "wrong",
                "manifest_audit": str(audit_path),
                "manifest_audit_sha256": _sha256(audit_path),
            }
        },
    }
    with pytest.raises(ValueError, match="source manifest"):
        compute_statistics.validate_fold_statistics_registry(
            statistics_registry, 0, audit_path, split_path
        )


def test_v5_manifest_rejects_duplicate_selected_ids(tmp_path: Path) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    duplicate = registry["folds"]["0"]["2"][0]
    registry["folds"]["0"]["2"] = [duplicate, duplicate]
    registry_path = tmp_path / "subsets.json"
    _write_json(registry_path, registry)

    with pytest.raises(ValueError, match="duplicate"):
        build_paper_manifests.materialize_fold_manifests(
            source_manifest_path=source_manifest_path,
            spatial_split_path=split_path,
            subset_registry_path=registry_path,
            output_dir=tmp_path / "manifests",
            fold_id=0,
            sizes=(2, 3),
            prefix="paper_v5",
            protocol_name="rse_v5_registered_20260726",
        )


@pytest.mark.parametrize(
    ("duplicate_source", "duplicate_fold"),
    [(True, False), (False, True)],
    ids=("source_manifest", "split_fold_membership"),
)
def test_v5_manifest_rejects_duplicate_source_or_fold_patch_ids(
    tmp_path: Path,
    duplicate_source: bool,
    duplicate_fold: bool,
) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    if duplicate_source:
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source.append(dict(source[0]))
        _write_json(source_manifest_path, source)
    if duplicate_fold:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["folds"][0]["train"][1] = split["folds"][0]["train"][0]
        _write_json(split_path, split)
        registry["source_split_sha256"] = _sha256(split_path)

    registry_path = tmp_path / "subsets.json"
    _write_json(registry_path, registry)
    with pytest.raises(ValueError, match="duplicate patch IDs"):
        build_paper_manifests.materialize_fold_manifests(
            source_manifest_path=source_manifest_path,
            spatial_split_path=split_path,
            subset_registry_path=registry_path,
            output_dir=tmp_path / "manifests",
            fold_id=0,
            sizes=(2, 3),
            prefix="paper_v5",
            protocol_name="rse_v5_registered_20260726",
        )


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("train", "val"),
        ("train", "test"),
        ("train", "buffer"),
        ("val", "test"),
        ("val", "buffer"),
        ("test", "buffer"),
    ),
)
def test_v5_manifest_rejects_all_fold_membership_overlaps(
    tmp_path: Path,
    left: str,
    right: str,
) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["folds"][0][right][0] = split["folds"][0][left][0]
    _write_json(split_path, split)
    registry["source_split_sha256"] = _sha256(split_path)
    registry_path = tmp_path / "subsets.json"
    _write_json(registry_path, registry)

    with pytest.raises(ValueError, match="membership overlap"):
        build_paper_manifests.materialize_fold_manifests(
            source_manifest_path=source_manifest_path,
            spatial_split_path=split_path,
            subset_registry_path=registry_path,
            output_dir=tmp_path / "manifests",
            fold_id=0,
            sizes=(2, 3),
            prefix="paper_v5",
            protocol_name="rse_v5_registered_20260726",
        )


def test_manifest_source_files_ignores_null_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "train_pool.json"
    _write_json(
        manifest_path,
        [
            {"patch_id": "patch_000000", "s2": [None, "patches/s2/one.tif"]},
            {"patch_id": "patch_000001", "s2": None},
            {"patch_id": "patch_000002"},
        ],
    )

    assert compute_statistics.manifest_source_files(tmp_path, manifest_path, "s2") == [
        tmp_path / "patches/s2/one.tif"
    ]


def test_fold_statistics_registry_records_materialized_audit_hash(tmp_path: Path) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    subsets = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    subsets_path = tmp_path / "subsets.json"
    _write_json(subsets_path, subsets)
    manifest_audit = build_paper_manifests.materialize_fold_manifests(
        source_manifest_path=source_manifest_path,
        spatial_split_path=split_path,
        subset_registry_path=subsets_path,
        output_dir=tmp_path / "manifests",
        fold_id=0,
        sizes=(2, 3),
        prefix="paper_v5",
        protocol_name="rse_v5_registered_20260726",
    )
    registry = compute_statistics.build_fold_statistics_registry(
        split_path=split_path,
        manifest_audits=[manifest_audit],
        statistics_root=tmp_path / "stats",
        protocol_name="rse_v5_registered_20260726",
    )
    statistics_entry = registry["folds"]["0"]
    statistics_file = Path(statistics_entry["statistics_dir"]) / "s2_stats.json"
    statistics_file.parent.mkdir(parents=True)
    statistics_file.write_text('{"mean": [0.0], "std": [1.0]}\n', encoding="utf-8")
    statistics_audit = Path(statistics_entry["statistics_audit"])
    compute_statistics.write_fold_statistics_audit(
        statistics_audit,
        Path(statistics_entry["source_manifest"]),
        {"s2": statistics_file},
        "rse_v5_registered_20260726",
    )

    compute_statistics.mark_fold_statistics_materialized(registry, 0, statistics_audit)

    s2_only_audit = statistics_audit.with_name("fold_statistics_s2_audit.json")
    compute_statistics.write_fold_statistics_audit(
        s2_only_audit,
        Path(statistics_entry["source_manifest"]),
        {"s2": statistics_file},
        "rse_v5_registered_20260726",
    )
    compute_statistics.mark_fold_statistics_materialized(registry, 0, s2_only_audit)

    assert registry["folds"]["0"]["status"] == "materialized"
    assert registry["folds"]["0"]["statistics_audit_sha256"] == _sha256(statistics_audit)
    assert registry["folds"]["0"]["statistics_audits"]["s2"]["sha256"] == _sha256(s2_only_audit)


def test_train_main_runs_v5_sidecar_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def fail_after_legacy_validation(_: object) -> None:
        observed.append("legacy")
        raise RuntimeError("stop after legacy validation")

    monkeypatch.setattr(train, "parse_args", lambda: SimpleNamespace(config="encoder.yaml"))
    monkeypatch.setattr(train, "setup_distributed", lambda: (False, 0))
    monkeypatch.setattr(train, "reject_base_config", lambda _: None)
    monkeypatch.setattr(
        train,
        "validate_registered_paper_v5_sidecars",
        lambda path: observed.append(f"v5:{path}"),
    )
    monkeypatch.setattr(
        train,
        "validate_registered_paper_manifests",
        fail_after_legacy_validation,
    )
    monkeypatch.setattr(
        train.Config,
        "from_yaml",
        lambda _: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="stop after legacy validation"):
        train.main()
    assert observed == ["v5:encoder.yaml", "legacy"]


def test_v5_generated_config_binds_sealed_sidecars_and_excludes_held_out_geography(
    tmp_path: Path,
) -> None:
    split_path, metadata_path, source_manifest_path = _v5_inputs(tmp_path)
    registry = build_paper_subset_registry.build_registry(
        split_path=split_path,
        patch_metadata_path=metadata_path,
        budgets=(2, 3),
        protocol_name="rse_v5_registered_20260726",
    )
    registry_path = tmp_path / "subsets.json"
    _write_json(registry_path, registry)
    manifest_dir = tmp_path / "manifests"
    audit_path = build_paper_manifests.materialize_fold_manifests(
        source_manifest_path=source_manifest_path,
        spatial_split_path=split_path,
        subset_registry_path=registry_path,
        output_dir=manifest_dir,
        fold_id=0,
        sizes=(2, 3),
        prefix="paper_v5",
        protocol_name="rse_v5_registered_20260726",
    )
    statistics_registry_path = tmp_path / "statistics.json"
    statistics_registry = compute_statistics.build_fold_statistics_registry(
        split_path=split_path,
        manifest_audits=[audit_path],
        statistics_root=tmp_path / "stats",
        protocol_name="rse_v5_registered_20260726",
    )
    _write_json(statistics_registry_path, statistics_registry)

    config = build_clean_paper_configs.base_config(
        name="paper_v5_full_2_fold0",
        train_size=2,
        fold=0,
        data_dir=manifest_dir,
        output_root=tmp_path / "outputs",
        split_path=split_path,
        subset_registry_path=registry_path,
        statistics_registry_path=statistics_registry_path,
        manifest_prefix="paper_v5",
        protocol_name="rse_v5_registered_20260726",
    )
    config_path = tmp_path / "encoder.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert config["data"]["paper_spatial_split_sha256"] == _sha256(split_path)
    assert config["data"]["paper_subset_registry_sha256"] == _sha256(registry_path)
    assert config["data"]["paper_manifest_audit_sha256"] == _sha256(audit_path)
    assert config["data"]["paper_normalization_statistics_sha256"] == _sha256(
        statistics_registry_path
    )
    config["data"]["sources"] = ["s2", "s1"]
    config["data"]["paper_normalization_statistics_audit_sha256"] = "pending"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="statistics audit"):
        train.validate_registered_paper_v5_sidecars(config_path)
    statistics_entry = statistics_registry["folds"]["0"]
    statistics_audit_path = Path(statistics_entry["statistics_audit"])
    statistics_audit_path.parent.mkdir(parents=True)
    statistics_file = statistics_audit_path.parent / "s2_stats.json"
    statistics_file.write_text('{"mean": [0.0], "std": [1.0]}\n', encoding="utf-8")
    _write_json(
        statistics_audit_path,
        {
            "source_manifest": statistics_entry["source_manifest"],
            "source_manifest_sha256": statistics_entry["source_manifest_sha256"],
            "statistics_files": {
                "s2": {"path": str(statistics_file), "sha256": _sha256(statistics_file)}
            },
        },
    )
    with pytest.raises(ValueError, match="configured sources"):
        train.validate_registered_paper_v5_sidecars(config_path)
    s1_statistics_file = statistics_audit_path.parent / "s1_stats.json"
    s1_statistics_file.write_text('{"mean": [0.0], "std": [1.0]}\n', encoding="utf-8")
    _write_json(
        statistics_audit_path,
        {
            "source_manifest": statistics_entry["source_manifest"],
            "source_manifest_sha256": statistics_entry["source_manifest_sha256"],
            "statistics_files": {
                "s2": {"path": str(statistics_file), "sha256": _sha256(statistics_file)},
                "s1": {"path": str(s1_statistics_file), "sha256": _sha256(s1_statistics_file)},
            },
        },
    )
    compute_statistics.mark_fold_statistics_materialized(
        statistics_registry, 0, statistics_audit_path
    )
    _write_json(statistics_registry_path, statistics_registry)
    config["data"]["paper_normalization_statistics_sha256"] = _sha256(statistics_registry_path)
    config["data"]["paper_normalization_statistics_audit"] = str(statistics_audit_path)
    config["data"]["paper_normalization_statistics_audit_sha256"] = _sha256(statistics_audit_path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    train.validate_registered_paper_v5_sidecars(config_path)
    statistics_file.write_text('{"mean": [1.0], "std": [1.0]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="statistics file"):
        train.validate_registered_paper_v5_sidecars(config_path)
    s1_statistics_file.write_text('{"mean": [0.0], "std": [1.0]}\n', encoding="utf-8")
    outside_statistics_file = tmp_path / "outside_stats.json"
    outside_statistics_file.write_text('{"mean": [0.0], "std": [1.0]}\n', encoding="utf-8")
    _write_json(
        statistics_audit_path,
        {
            "source_manifest": statistics_entry["source_manifest"],
            "source_manifest_sha256": statistics_entry["source_manifest_sha256"],
            "statistics_files": {
                "s2": {"path": str(statistics_file), "sha256": _sha256(statistics_file)},
                "s1": {
                    "path": str(outside_statistics_file),
                    "sha256": _sha256(outside_statistics_file),
                },
            },
        },
    )
    with pytest.raises(ValueError, match="statistics_dir"):
        train.validate_registered_paper_v5_sidecars(config_path)
    _write_json(
        statistics_audit_path,
        {
            "source_manifest": statistics_entry["source_manifest"],
            "source_manifest_sha256": statistics_entry["source_manifest_sha256"],
            "statistics_files": {
                "s2": {"path": str(statistics_file), "sha256": _sha256(statistics_file)},
                "s1": {"path": str(s1_statistics_file), "sha256": _sha256(s1_statistics_file)},
                "unexpected": {
                    "path": str(s1_statistics_file),
                    "sha256": _sha256(s1_statistics_file),
                },
            },
        },
    )
    with pytest.raises(ValueError, match="configured sources"):
        train.validate_registered_paper_v5_sidecars(config_path)
    held_out = {"patch_000004", "patch_000005", "patch_000006", "patch_000007"}
    train_ids = {
        record["patch_id"]
        for record in json.loads(Path(config["data"]["train_manifest_path"]).read_text())
    }
    assert not train_ids & held_out


def _v5_protocol_bindings() -> dict[str, str]:
    descriptor = registered.resolve_registered_protocol("v5_osm_assisted")
    return {
        "spatial_split": str(descriptor["split_path"]),
        "spatial_split_sha256": str(descriptor["split_sha256"]),
        "manifest_sha256": str(descriptor["manifest_sha256"]),
        "statistics_registry_sha256": str(descriptor["statistics_registry_sha256"]),
    }


def _sealed_v5_matrix(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "sealed-matrix-repo"
    matrix_path = repo / "configs/eval/rse_v5_osm_assisted_matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        (
            Path(__file__).resolve().parents[1] / "configs/eval/rse_v5_osm_assisted_matrix.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Protocol Test"],
        ["git", "add", "configs/eval/rse_v5_osm_assisted_matrix.json"],
        ["git", "commit", "-qm", "seal matrix"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    return matrix_path, _sha256(matrix_path)


def _sealed_v5_protocol_assets(tmp_path: Path) -> tuple[Path, str, Path]:
    matrix_path, matrix_sha256 = _sealed_v5_matrix(tmp_path)
    registry_path = matrix_path.with_name("registered_embedding_exports_v5_20260726.json")
    registry_path.write_text(
        (
            Path(__file__).resolve().parents[1]
            / "configs/eval/registered_embedding_exports_v5_20260726.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", str(registry_path.relative_to(matrix_path.parents[2]))],
        cwd=matrix_path.parents[2],
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "seal registry"], cwd=matrix_path.parents[2], check=True
    )
    return matrix_path, matrix_sha256, registry_path


def test_v5_protocol_resolver_rejects_v4_split_and_mismatched_hashes() -> None:
    bindings = _v5_protocol_bindings()

    with pytest.raises(ValueError, match="spatial split"):
        registered.validate_protocol_bindings(
            "v5_osm_assisted",
            {**bindings, "spatial_split": str(registered.FROZEN_SPLIT)},
        )
    with pytest.raises(ValueError, match="statistics registry"):
        registered.validate_protocol_bindings(
            "v5_osm_assisted",
            {**bindings, "statistics_registry_sha256": "mismatch"},
        )


def test_v5_schedule_and_probe_reject_another_protocol() -> None:
    bindings = _v5_protocol_bindings()
    schedule = {
        "protocol_id": "v4_diagnostic",
        "split_sha256": bindings["spatial_split_sha256"],
    }

    with pytest.raises(ValueError, match="shot schedule"):
        registered.validate_registered_shot_schedule(schedule, "v5_osm_assisted")
    with pytest.raises(ValueError, match="protocol_id"):
        registered.validate_v5_probe_request(
            protocol_id="v4_diagnostic",
            bindings=bindings,
            cell={
                "family": "full_150",
                "head": "conv3x3_64_128_64",
                "task": "building",
                "shot": "5",
                "fold": 0,
                "seed": 42,
            },
        )


def test_v5_matrix_rejects_undeclared_result_cell_and_bootstrap_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, matrix_sha256 = _sealed_v5_matrix(tmp_path)
    monkeypatch.setattr(registered, "V5_MATRIX", matrix_path)
    monkeypatch.setattr(registered, "V5_MATRIX_SHA256", matrix_sha256)
    cell = {
        "family": "full_150",
        "head": "conv3x3_64_128_64",
        "task": "building",
        "shot": "5",
        "fold": 0,
        "seed": 42,
    }
    registered.validate_registered_matrix_cell("v5_osm_assisted", cell)
    with pytest.raises(ValueError, match="matrix"):
        registered.validate_registered_matrix_cell("v5_osm_assisted", {**cell, "seed": 999})
    baseline = {("building", "5", 0, 42): _result("v5_osm_assisted")}
    candidate = {("building", "5", 0, 42): _result("v5_osm_assisted")}
    candidate[("building", "5", 0, 42)]["metric_provenance"]["test_patch_ids_sha256"] = "other"
    with pytest.raises(ValueError, match="test_patch_ids_sha256"):
        bootstrap.verify_paired_record_provenance(baseline, candidate)


def test_v5_matrix_rejects_uncommitted_working_copy(tmp_path: Path) -> None:
    matrix_path, matrix_sha256 = _sealed_v5_matrix(tmp_path)
    matrix_path.write_text(matrix_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current Git HEAD"):
        registered.load_registered_v5_matrix(matrix_path, matrix_sha256)


def test_v5_protocol_assets_are_commit_ready_only_at_git_head(tmp_path: Path) -> None:
    matrix_path, matrix_sha256, registry_path = _sealed_v5_protocol_assets(tmp_path)

    registered.validate_v5_protocol_assets_for_commit(matrix_path, matrix_sha256, registry_path)
    registry_path.write_text(registry_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current Git HEAD"):
        registered.validate_v5_protocol_assets_for_commit(matrix_path, matrix_sha256, registry_path)


def test_v5_encoder_config_requires_head_bytes_and_matching_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs/paper_registered_v5_20260726/paper_registered_v5_full_150_fold0_20260726.yaml"
    )
    config_sha256 = _sha256(config_path)

    registered.validate_v5_encoder_config_provenance(config_path, config_sha256)
    with pytest.raises(ValueError, match="hash"):
        registered.validate_v5_encoder_config_provenance(config_path, "mismatch")


def test_v5_family_name_matches_probe_and_aggregation_provenance(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs/paper_registered_v5_20260726/paper_registered_v5_full_150_fold0_20260726.yaml"
    )
    record = {
        "metric_provenance": {
            "protocol_id": "v5_osm_assisted",
            "fold": 0,
            "embedding_export": {"config_path": str(config_path)},
            "provenance": {"config_sha256": _sha256(config_path)},
        }
    }

    assert registered.registered_family_from_experiment(config_path.stem, 0) == "full_150"
    assert aggregate._family_from_record(record) == "full_150"
    legacy_config = tmp_path / "paper_registered_full_150_fold0_20260716.yaml"
    legacy_config.write_text(
        yaml.safe_dump({"experiment": {"name": legacy_config.stem}, "data": {"paper_fold": 0}}),
        encoding="utf-8",
    )
    legacy_record = {
        "metric_provenance": {
            "protocol_id": "v5_osm_assisted",
            "fold": 0,
            "embedding_export": {"config_path": str(legacy_config)},
            "provenance": {"config_sha256": _sha256(legacy_config)},
        }
    }
    with pytest.raises(ValueError, match="namespace"):
        aggregate._family_from_record(legacy_record)


def test_v5_bootstrap_requires_the_matrix_declared_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, matrix_sha256 = _sealed_v5_matrix(tmp_path)
    monkeypatch.setattr(registered, "V5_MATRIX", matrix_path)
    monkeypatch.setattr(registered, "V5_MATRIX_SHA256", matrix_sha256)

    registered.validate_v5_matrix_comparator("full_150", "full_40")
    with pytest.raises(ValueError, match="declared comparator"):
        registered.validate_v5_matrix_comparator("no_osm_150", "full_40")


def test_v5_embedding_registry_requires_versioned_protocol_and_bindings(tmp_path: Path) -> None:
    bindings = _v5_protocol_bindings()
    unversioned = tmp_path / "registry.json"
    _write_json(unversioned, {"exports": []})
    with pytest.raises(ValueError, match="schema_version"):
        registered.validate_v5_embedding_registry(unversioned, bindings)

    mismatched = tmp_path / "mismatched.json"
    _write_json(
        mismatched,
        {
            "schema_version": 2,
            "protocol_id": "v5_osm_assisted",
            **{**bindings, "manifest_sha256": "mismatch"},
            "exports": [],
        },
    )
    with pytest.raises(ValueError, match="manifest"):
        registered.validate_v5_embedding_registry(mismatched, bindings)


def _sealed_v5_export_root(tmp_path: Path, *, fold: int = 0) -> tuple[Path, dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / f"configs/paper_registered_v5_20260726/paper_registered_v5_full_150_fold{fold}_20260726.yaml"
    )
    checkpoint_path = tmp_path / f"best_fold{fold}.pt"
    checkpoint_path.write_bytes(b"sealed checkpoint")
    export_root = tmp_path / f"export_fold{fold}"
    region_root = export_root / "haidian"
    patch_ids = [f"patch_{index:06d}" for index in range(320)]
    for index, patch_id in enumerate(patch_ids):
        feature = region_root / patch_id / "202604_embedding_map.pt"
        feature.parent.mkdir(parents=True, exist_ok=True)
        feature.write_bytes(f"embedding-{index}".encode("utf-8"))
    commands = {shard: f"export shard {shard}" for shard in range(6)}
    for shard in commands:
        shard_ids = patch_ids[shard::6]
        (region_root / f"produced_patch_ids_shard_{shard}.json").write_text(
            json.dumps(shard_ids), encoding="utf-8"
        )
    _write_json(
        export_root / "meta.json",
        {
            "protocol_id": "v5_osm_assisted",
            **_v5_protocol_bindings(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "manifest": {
                "manifest_sha256": registered.resolve_registered_protocol("v5_osm_assisted")[
                    "manifest_sha256"
                ],
                "patch_count": 320,
            },
        },
    )
    registered.canonicalize_embedding_export(
        export_root,
        "haidian",
        "202604",
        expected_patch_ids=set(patch_ids),
        shard_commands=commands,
    )
    registered.seal_embedding_file_index(export_root, "haidian", "202604")
    return export_root, {"family": "full_150", "fold": fold, "region": "haidian", "month": "202604"}


def test_v5_registrar_rebuilds_export_entry_and_rejects_forged_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, matrix_sha256, registry_path = _sealed_v5_protocol_assets(tmp_path)
    monkeypatch.setattr(registered, "V5_MATRIX", matrix_path)
    monkeypatch.setattr(registered, "V5_MATRIX_SHA256", matrix_sha256)
    export_root, identity = _sealed_v5_export_root(tmp_path)
    entry = {
        **registered.build_v5_embedding_registry_entry(
            export_root,
            family=str(identity["family"]),
            fold=int(identity["fold"]),
            region=str(identity["region"]),
            month=str(identity["month"]),
        ),
        "embedding_root": str(export_root.resolve()),
    }
    pending_path = tmp_path / "pending_registry_entry.json"
    pending = {
        "schema_version": 1,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "entry": entry,
        "entry_sha256": registered._canonical_sha256(entry),
    }
    _write_json(pending_path, pending)
    monkeypatch.setattr(
        "sys.argv",
        [
            "register_registered_v5_embedding_export.py",
            "--registry",
            str(registry_path),
            "--pending-entry",
            str(pending_path),
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="exactly folds 0 through 4"):
        registrar.main()

    forged = dict(entry)
    forged["checkpoint_sha256"] = "forged"
    pending["entry"] = forged
    pending["entry_sha256"] = registered._canonical_sha256(forged)
    with pytest.raises(ValueError, match="does not match the sealed export"):
        registrar.rebuild_entry_from_export(forged)


def test_v5_registrar_appends_multiple_exports_from_one_registry_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five fold exports created from one sealed registry must register atomically."""
    matrix_path, matrix_sha256, registry_path = _sealed_v5_protocol_assets(tmp_path)
    monkeypatch.setattr(registered, "V5_MATRIX", matrix_path)
    monkeypatch.setattr(registered, "V5_MATRIX_SHA256", matrix_sha256)
    pending_paths: list[Path] = []
    for fold in range(5):
        export_root, identity = _sealed_v5_export_root(tmp_path, fold=fold)
        entry = {
            **registered.build_v5_embedding_registry_entry(
                export_root,
                family=str(identity["family"]),
                fold=int(identity["fold"]),
                region=str(identity["region"]),
                month=str(identity["month"]),
            ),
            "embedding_root": str(export_root.resolve()),
        }
        pending_path = tmp_path / f"pending_fold{fold}.json"
        _write_json(
            pending_path,
            {
                "schema_version": 1,
                "registry_path": str(registry_path.resolve()),
                "registry_sha256": _sha256(registry_path),
                "entry": entry,
                "entry_sha256": registered._canonical_sha256(entry),
            },
        )
        pending_paths.append(pending_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "register_registered_v5_embedding_export.py",
            "--registry",
            str(registry_path),
            "--pending-entry",
            *(str(path) for path in pending_paths[:-1]),
        ],
    )
    with pytest.raises(ValueError, match="exactly folds 0 through 4"):
        registrar.main()

    before = registry_path.read_bytes()
    monkeypatch.setattr(
        "sys.argv",
        [
            "register_registered_v5_embedding_export.py",
            "--registry",
            str(registry_path),
            "--pending-entry",
            *(str(path) for path in pending_paths),
            "--dry-run",
        ],
    )
    registrar.main()
    assert registry_path.read_bytes() == before

    monkeypatch.setattr(
        "sys.argv",
        [
            "register_registered_v5_embedding_export.py",
            "--registry",
            str(registry_path),
            "--pending-entry",
            *(str(path) for path in pending_paths),
        ],
    )
    registrar.main()

    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert {entry["encoder_fold"] for entry in raw["exports"]} == set(range(5))


def test_v5_registrar_rejects_pending_entries_from_mixed_registry_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch must not combine exports sealed against different registry bytes."""
    _, _, registry_path = _sealed_v5_protocol_assets(tmp_path)
    pending_paths = [tmp_path / f"pending_{fold}.json" for fold in range(5)]
    entries = [
        {
            "id": f"v5-full_150-fold{fold}-haidian-202604",
            "family": "full_150",
            "encoder_fold": fold,
            "protocol_id": "v5_osm_assisted",
            "checkpoint_sha256": "checkpoint",
            "config_sha256": "config",
            "manifest_sha256": "manifest",
            "embedding_file_index_sha256": "index",
            "canonical_export_provenance_sha256": "provenance",
            "region": "haidian",
            "month": "202604",
            "patch_count": 320,
            "embedding_root": str(tmp_path / f"export_{fold}"),
        }
        for fold in range(5)
    ]
    snapshots = ["snapshot-a", "snapshot-a", "snapshot-b", "snapshot-a", "snapshot-a"]
    monkeypatch.setattr(
        registrar,
        "load_pending_entry",
        lambda path, registry: (entries[pending_paths.index(path)], snapshots[pending_paths.index(path)]),
    )
    monkeypatch.setattr(registrar, "rebuild_entry_from_export", lambda entry: entry)
    monkeypatch.setattr(
        "sys.argv",
        [
            "register_registered_v5_embedding_export.py",
            "--registry",
            str(registry_path),
            "--pending-entry",
            *(str(path) for path in pending_paths),
        ],
    )

    with pytest.raises(ValueError, match="same registry snapshot"):
        registrar.main()


def test_v5_rejects_aef_family_before_task5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path, matrix_sha256 = _sealed_v5_matrix(tmp_path)
    monkeypatch.setattr(registered, "V5_MATRIX", matrix_path)
    monkeypatch.setattr(registered, "V5_MATRIX_SHA256", matrix_sha256)
    aef_cell = {
        "family": "matched_aef_v5",
        "head": "conv3x3_64_128_64",
        "task": "building",
        "shot": "5",
        "fold": 0,
        "seed": 42,
    }

    with pytest.raises(ValueError, match="only full_150"):
        registered.validate_registered_matrix_cell("v5_osm_assisted", aef_cell)
    with pytest.raises(ValueError, match="baseline is outside"):
        registered.validate_v5_matrix_comparator("matched_aef_v5", "full_150")


def test_v5_canonical_checkpoint_resolver_uses_the_queue_sealed_pointer(tmp_path: Path) -> None:
    """Export and probe stages must consume the queue's verified checkpoint snapshot."""
    from scripts.eval import registered_v5_encoder_checkpoint as checkpoint_resolver

    job_root = tmp_path / "paper_registered_v5_full_150_fold2_20260726"
    attempt = job_root / "attempt_1"
    checkpoint = attempt / "verified_checkpoints" / "best_sealed.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed checkpoint")
    manifest = attempt / "attempt_manifest.json"
    _write_json(manifest, {"schema_version": 1, "attempt_dir": str(attempt.resolve())})
    pointer = {
        "schema_version": 1,
        "attempt_dir": str(attempt.resolve()),
        "attempt_manifest_sha256": _sha256(manifest),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    pointer["sha256"] = hashlib.sha256(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pointer_path = job_root / "canonical_attempt.json"
    _write_json(pointer_path, pointer)

    with pytest.raises(ValueError, match="verification record"):
        checkpoint_resolver.resolve_registered_v5_checkpoint(tmp_path, family="full_150", fold=2)
    verification = attempt / "checkpoint_verifications" / f"{checkpoint.name}.json"
    verification.parent.mkdir()
    _write_json(
        verification,
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
        },
    )
    assert checkpoint_resolver.resolve_registered_v5_checkpoint(
        tmp_path, family="full_150", fold=2
    ) == checkpoint.resolve()
    checkpoint_resolver.validate_registered_v5_checkpoint_path(
        checkpoint.resolve(), expected_job_name=job_root.name
    )
    pointer["sha256"] = "forged"
    _write_json(pointer_path, pointer)
    with pytest.raises(ValueError, match="self-hash"):
        checkpoint_resolver.resolve_registered_v5_checkpoint(tmp_path, family="full_150", fold=2)


def test_v5_canonical_checkpoint_resolver_requires_the_sealed_source_config(
    tmp_path: Path,
) -> None:
    """An export cannot relabel a sealed checkpoint with another fold config."""
    from scripts.eval import registered_v5_encoder_checkpoint as checkpoint_resolver

    attempt = tmp_path / "paper_registered_v5_full_150_fold0_20260726" / "attempt_1"
    checkpoint = attempt / "verified_checkpoints" / "best_sealed.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed checkpoint")
    source_config = tmp_path / "source_fold0.yaml"
    requested_config = tmp_path / "requested_fold0.yaml"
    source_config.write_text("experiment: {name: source}\n", encoding="utf-8")
    requested_config.write_text("experiment: {name: requested}\n", encoding="utf-8")
    manifest = attempt / "attempt_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "source_config": {"path": str(source_config.resolve()), "sha256": _sha256(source_config)},
        },
    )
    verification = attempt / "checkpoint_verifications" / f"{checkpoint.name}.json"
    verification.parent.mkdir()
    _write_json(
        verification,
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
        },
    )
    pointer = {
        "schema_version": 1,
        "attempt_dir": str(attempt.resolve()),
        "attempt_manifest_sha256": _sha256(manifest),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    pointer["sha256"] = checkpoint_resolver._canonical_pointer_sha256(pointer)
    _write_json(attempt.parent / "canonical_attempt.json", pointer)

    checkpoint_resolver.validate_registered_v5_checkpoint_config_binding(
        checkpoint, expected_config_path=source_config
    )
    with pytest.raises(ValueError, match="source config"):
        checkpoint_resolver.validate_registered_v5_checkpoint_config_binding(
            checkpoint, expected_config_path=requested_config
        )


def test_v5_shot_schedule_generator_resolves_repo_imports_when_run_directly(tmp_path: Path) -> None:
    """The schedule preparation command must not import the shadowing downstream package."""
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": f"{root / 'src'}:{root / 'downstreams'}",
    }

    result = subprocess.run(
        [sys.executable, str(root / "scripts/eval/prepare_registered_shot_manifests.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Generate immutable shared few-shot schedules" in result.stdout


def _sealed_v5_wrapper_repo(tmp_path: Path) -> Path:
    """Create a minimal committed repository for V5 wrapper integration tests."""
    source_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "wrapper-repo"
    for relative in (
        Path("scripts/__init__.py"),
        Path("scripts/eval/export_registered_v5_paper_encoders.sh"),
        Path("scripts/eval/launch_registered_v5_downstream.sh"),
        Path("scripts/eval/registered_v5_encoder_checkpoint.py"),
        Path("scripts/eval/registered_v5_matrix.py"),
        Path("scripts/eval/run_registered_paper_downstream.py"),
        Path("downstreams/scripts/precompute_embeddings.py"),
        Path("downstreams/scripts/export_paths.py"),
        Path("downstreams/downstreams/inference.py"),
        Path("src/xuannv_embedding/models/model.py"),
        Path("configs/eval/rse_v5_osm_assisted_matrix.json"),
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "wrapper-test@example.invalid"],
        ["git", "config", "user.name", "Wrapper Test"],
        ["git", "add", "."],
        ["git", "commit", "--quiet", "-m", "seal wrapper fixture"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    return repo


def test_v5_export_and_launcher_dry_runs_admit_declared_family_only(tmp_path: Path) -> None:
    root = _sealed_v5_wrapper_repo(tmp_path)
    exporter = root / "scripts/eval/export_registered_v5_paper_encoders.sh"
    launcher = root / "scripts/eval/launch_registered_v5_downstream.sh"
    for script in (exporter, launcher):
        missing_protocol = subprocess.run(
            [str(script), "--family", "full_150", "--dry-run"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert missing_protocol.returncode == 2
        accepted = subprocess.run(
            [
                str(script),
                "--protocol",
                "v5_osm_assisted",
                "--family",
                "full_150",
                "--dry-run",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert accepted.returncode == 0, accepted.stderr
        assert all(line.startswith(("EXPORT ", "PROBE ")) for line in accepted.stdout.splitlines())
        if script == exporter:
            assert sum("phase=finalize" in line for line in accepted.stdout.splitlines()) == 5
        for rejected_family in ("not_declared", "matched_aef_v5", "full_80", "no_osm_150"):
            rejected = subprocess.run(
                [
                    str(script),
                    "--protocol",
                    "v5_osm_assisted",
                    "--family",
                    rejected_family,
                    "--dry-run",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            assert rejected.returncode == 2


def test_v5_exporter_refuses_to_contend_with_an_active_registered_trainer(tmp_path: Path) -> None:
    """The exporter must fail before any NPU work if a V5 trainer is live."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    pgrep.chmod(0o755)

    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "registered V5 encoder training is active" in result.stderr


def test_v5_exporter_preflights_every_fold_and_reserves_exit_five_for_busy_lanes() -> None:
    """All-fold checkpoint binding must precede writes and shard failure must not look busy."""
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/eval/export_registered_v5_paper_encoders.sh"
    ).read_text(encoding="utf-8")

    assert "preflight_registered_v5_folds()" in source
    assert source.index("  preflight_registered_v5_folds") < source.index('RUN_ID="$(date -u')
    assert "one or more v5 export shards failed for ${suffix}\" >&2\n    exit 6" in source


def test_v5_exporter_invalid_preflight_creates_no_export_root(tmp_path: Path) -> None:
    """One invalid canonical fold fails closed before any export directory or shard starts."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    for fold in range(5):
        config = root / "configs/paper_registered_v5_20260726" / (
            f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"fold: {fold}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (bin_dir / "pgrep").chmod(0o755)
    marker = tmp_path / "python-invoked"
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$*\" >> \"$V5_PYTHON_MARKER\"\n"
        "if [[ \"$1\" == *registered_v5_encoder_checkpoint.py ]]; then\n"
        "  count=0; [[ -f \"$V5_RESOLVER_COUNT\" ]] && count=$(<\"$V5_RESOLVER_COUNT\")\n"
        "  count=$((count + 1)); echo \"$count\" > \"$V5_RESOLVER_COUNT\"\n"
        "  if [[ $count -eq 5 ]]; then echo 'invalid canonical fold' >&2; exit 3; fi\n"
        "  echo \"$V5_TEST_CHECKPOINT\"; exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"-\" ]]; then exit 0; fi\n"
        "exec \"$V5_REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)
    embed_root = tmp_path / "embeddings"
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "V5_PYTHON_MARKER": str(marker),
            "V5_RESOLVER_COUNT": str(tmp_path / "resolver-count"),
            "V5_REAL_PYTHON": sys.executable,
            "V5_TEST_CHECKPOINT": str(checkpoint),
            "V5_EMBED_ROOT": str(embed_root),
            "V5_LOG_ROOT": str(tmp_path / "logs"),
        },
    )

    assert result.returncode == 3
    assert not embed_root.exists()
    assert "precompute_embeddings.py" not in marker.read_text(encoding="utf-8")


def test_v5_exporter_rechecks_for_training_after_leases(tmp_path: Path) -> None:
    """A trainer that appears after the first scan aborts export with the retryable busy code."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count = tmp_path / "pgrep-count"
    (bin_dir / "pgrep").write_text(
        "#!/usr/bin/env bash\n"
        "count=0; [[ -f \"$V5_PGREP_COUNT\" ]] && count=$(<\"$V5_PGREP_COUNT\")\n"
        "count=$((count + 1)); echo \"$count\" > \"$V5_PGREP_COUNT\"\n"
        "[[ $count -ge 6 ]] && exit 0\n"
        "exit 1\n",
        encoding="utf-8",
    )
    (bin_dir / "pgrep").chmod(0o755)
    embed_root = tmp_path / "embeddings"
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "V5_PGREP_COUNT": str(count),
            "V5_EMBED_ROOT": str(embed_root),
        },
    )

    assert result.returncode == 5
    assert "registered V5 encoder training is active" in result.stderr
    assert not embed_root.exists()


def test_v5_exporter_rejects_path_override_outside_pytest_fixture(tmp_path: Path) -> None:
    """Production invocations cannot redirect registered exports through environment overrides."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTEST_CURRENT_TEST": "",
            "V5_EMBED_ROOT": str(tmp_path / "redirected"),
        },
    )

    assert result.returncode == 2
    assert "path overrides are restricted to pytest fixtures" in result.stderr


def test_v5_exporter_rejects_a_snapshot_changed_after_shard_launch(tmp_path: Path) -> None:
    """Finalization must reject a config snapshot modified after the workers consumed it."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    for fold in range(5):
        config = root / "configs/paper_registered_v5_20260726" / (
            f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("fold: 0\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_sha256 = hashlib.sha256(b"fold: 0\n").hexdigest()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (bin_dir / "pgrep").chmod(0o755)
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == *registered_v5_encoder_checkpoint.py ]]; then\n"
        "  echo \"$V5_TEST_CHECKPOINT\"; exit 0\n"
        "fi\n"
        "if [[ \"$1\" == *precompute_embeddings.py ]]; then\n"
        "  for ((i=1; i<=$#; i++)); do\n"
        "    [[ \"${!i}\" == --config ]] && next=$((i + 1)) && printf tampered > \"${!next}\"\n"
        "  done\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"-\" ]]; then\n"
        "  payload=$(cat)\n"
        "  if [[ \"$payload\" == *validate_registered_v5_checkpoint_config_binding* ]]; then\n"
        "    echo \"$V5_TEST_CONFIG_SHA\"; exit 0\n"
        "  fi\n"
        "  if [[ \"$payload\" == *training_config_snapshot* ]]; then\n"
        "    printf '%s' \"$payload\" | exec \"$V5_REAL_PYTHON\" \"$@\"\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "exec \"$V5_REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)
    embed_root = tmp_path / "embeddings"
    embed_root.mkdir()
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "V5_REAL_PYTHON": sys.executable,
            "V5_TEST_CHECKPOINT": str(checkpoint),
            "V5_TEST_CONFIG_SHA": config_sha256,
            "V5_EMBED_ROOT": str(embed_root),
            "V5_LOG_ROOT": str(tmp_path / "logs"),
        },
    )

    assert result.returncode != 0
    assert "training config snapshot hash changed before finalization" in result.stderr
    assert not list(embed_root.rglob("meta.json"))
    assert not list(embed_root.rglob("pending_registry_entry.json"))


def test_v5_exporter_shard_failure_exits_six(tmp_path: Path) -> None:
    """A failed export shard is terminal, rather than being mistaken for a busy lane."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    for fold in range(5):
        config = root / "configs/paper_registered_v5_20260726" / (
            f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"fold: {fold}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config_sha256 = hashlib.sha256(b"fold: 0\n").hexdigest()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (bin_dir / "pgrep").chmod(0o755)
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == *registered_v5_encoder_checkpoint.py ]]; then\n"
        "  echo \"$V5_TEST_CHECKPOINT\"\n"
        "  exit 0\n"
        "fi\n"
        "if [[ \"$1\" == \"-\" ]]; then echo \"$V5_TEST_CONFIG_SHA\"; exit 0; fi\n"
        "if [[ \"$1\" == *precompute_embeddings.py ]]; then exit 1; fi\n"
        "exec \"$V5_REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)
    embed_root = tmp_path / "embeddings"
    embed_root.mkdir()
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "V5_REAL_PYTHON": sys.executable,
            "V5_TEST_CHECKPOINT": str(checkpoint),
            "V5_TEST_CONFIG_SHA": config_sha256,
            "V5_EMBED_ROOT": str(embed_root),
            "V5_LOG_ROOT": str(tmp_path / "logs"),
        },
    )

    assert result.returncode == 6
    assert "one or more v5 export shards failed" in result.stderr


def test_v5_exporter_refuses_a_shared_registered_lane_lease(tmp_path: Path) -> None:
    """A queue lease acquired after pgrep still prevents all-NPU export contention."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    pgrep.chmod(0o755)
    lock_root = root / ".pytest_registered_v5_locks"
    lock_root.mkdir()
    ready = tmp_path / "lease-ready"
    holder = subprocess.Popen(
        [
            "flock",
            str(lock_root / "lane_1.lock"),
            "bash",
            "-c",
            "touch \"$1\"; sleep 30",
            "bash",
            str(ready),
        ]
    )
    try:
        for _ in range(100):
            if ready.is_file():
                break
            time.sleep(0.01)
        assert ready.is_file(), "lease holder did not acquire lane_1"
        result = subprocess.run(
            [
                str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
                "--protocol",
                "v5_osm_assisted",
                "--family",
                "full_150",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
            },
        )
    finally:
        holder.terminate()
        holder.wait(timeout=10)

    assert result.returncode != 0
    assert "encoder lane 1 is active" in result.stderr


def test_v5_exporter_holds_all_registered_lane_leases_after_acquisition(tmp_path: Path) -> None:
    """All three leases remain held while the exporter is still in its preflight phase."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    pgrep.chmod(0o755)
    ready = tmp_path / "exporter-after-locks"
    python_wrapper = bin_dir / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ -n \"$V5_LEASE_READY\" ]]; then\n"
        "  touch \"$V5_LEASE_READY\"\n"
        "  sleep 30\n"
        "  exit 0\n"
        "fi\n"
        "exec \"$V5_LEASE_REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    exporter = subprocess.Popen(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "V5_LEASE_READY": str(ready),
            "V5_LEASE_REAL_PYTHON": sys.executable,
        },
    )
    try:
        for _ in range(100):
            if ready.is_file():
                break
            time.sleep(0.01)
        assert ready.is_file(), "exporter did not reach the post-lease preflight"
        lock_root = root / ".pytest_registered_v5_locks"
        for lane in range(3):
            contender = subprocess.run(
                ["flock", "-n", str(lock_root / f"lane_{lane}.lock"), "true"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert contender.returncode != 0, f"lane {lane} was released during export"
    finally:
        exporter.terminate()
        exporter.wait(timeout=10)


def test_v5_probe_request_rejects_a_declared_but_not_yet_admitted_encoder_family() -> None:
    """Direct runner calls cannot bypass the full_150-only release boundary."""
    cell = {
        "family": "full_80",
        "head": "conv3x3_64_128_64",
        "task": "building",
        "shot": "5",
        "fold": 0,
        "seed": 42,
    }

    with pytest.raises(ValueError, match="only full_150"):
        registered.validate_registered_matrix_cell("v5_osm_assisted", cell)


def test_v5_full_matrix_preflight_rejects_a_missing_fold_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No probe wave may start when even one full_150 export is absent."""
    config_root = tmp_path / "configs"
    for fold in range(5):
        (config_root / f"paper_registered_v5_full_150_fold{fold}_20260726.yaml").parent.mkdir(
            parents=True, exist_ok=True
        )
        (config_root / f"paper_registered_v5_full_150_fold{fold}_20260726.yaml").write_text(
            f"fold: {fold}\n", encoding="utf-8"
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]\n", encoding="utf-8")
    entries = [
        {"family": "full_150", "encoder_fold": fold, "embedding_root": str(tmp_path / str(fold))}
        for fold in range(1, 5)
    ]
    resolver = __import__(
        "scripts.eval.registered_v5_encoder_checkpoint", fromlist=["placeholder"]
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(resolver, "resolve_registered_v5_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(registered, "load_registered_v5_matrix", lambda: {"families": {"full_150": {}}})
    monkeypatch.setattr(registered, "validate_v5_embedding_registry", lambda *_args: {"exports": entries})
    monkeypatch.setattr(
        registered,
        "verify_encoder_provenance",
        lambda *_args, **_kwargs: {"config_sha256": "config", "checkpoint_sha256": "checkpoint"},
    )
    with pytest.raises(ValueError, match="exactly one full_150 embedding export for fold 0"):
        registered.validate_v5_full_150_matrix_readiness(
            encoder_root=tmp_path / "encoders",
            registry_path=tmp_path / "registry.json",
            manifest_path=manifest,
            config_root=config_root,
        )


def test_v5_exporter_dry_run_can_plan_an_atomic_registry_batch(tmp_path: Path) -> None:
    """All five export entries must be registered from their shared registry snapshot."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
            "--register-pending",
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("REGISTER protocol=v5_osm_assisted") == 1


def test_v5_exporter_finalization_heredoc_is_valid_python() -> None:
    """The post-shard finalization program must compile before a long export is launched."""
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/eval/export_registered_v5_paper_encoders.sh").read_text(
        encoding="utf-8"
    )
    finalizer = source.rsplit("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]

    compile(finalizer, "registered_v5_export_finalizer", "exec")
    assert "import hashlib" in finalizer


def test_v5_export_and_launcher_dry_runs_work_from_outside_repository(tmp_path: Path) -> None:
    """V5 wrappers must resolve their registered relative assets from the repository root."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_cwds = tmp_path / "python-cwds.txt"
    python_wrapper = bin_dir / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$PWD" >> "$V5_TEST_PYTHON_CWDS"\n'
        'exec "$V5_TEST_REAL_PYTHON" "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "V5_TEST_PYTHON_CWDS": str(python_cwds),
        "V5_TEST_REAL_PYTHON": sys.executable,
    }
    for script in (
        root / "scripts/eval/export_registered_v5_paper_encoders.sh",
        root / "scripts/eval/launch_registered_v5_downstream.sh",
    ):
        result = subprocess.run(
            [
                str(script),
                "--protocol",
                "v5_osm_assisted",
                "--family",
                "full_150",
                "--dry-run",
            ],
            cwd=outside,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
    assert python_cwds.read_text(encoding="utf-8").splitlines() == [str(root), str(root)]


def test_v5_exporter_rejects_a_dirty_runtime_source(tmp_path: Path) -> None:
    """A V5 export must not begin with modified feature-generation code."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    precompute = root / "downstreams/scripts/precompute_embeddings.py"
    precompute.write_text(precompute.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "runtime source tree differs from Git HEAD" in result.stderr


def test_v5_exporter_rejects_a_dirty_model_dependency(tmp_path: Path) -> None:
    """The export gate covers model/data code, not only the top-level exporter."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    model = root / "src/xuannv_embedding/models/model.py"
    model.write_text(model.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "runtime source tree differs from Git HEAD" in result.stderr


def test_v5_exporter_rejects_an_untracked_runtime_source(tmp_path: Path) -> None:
    """The runtime-tree gate also rejects untracked source files."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    untracked = root / "src/xuannv_embedding/untracked_runtime.py"
    untracked.write_text("VALUE = 'untracked'\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(root / "scripts/eval/export_registered_v5_paper_encoders.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "runtime source tree differs from Git HEAD" in result.stderr


def test_v5_launcher_rejects_a_dirty_runner_before_import(tmp_path: Path) -> None:
    """The shell launcher rejects a dirty runner before it starts a Python probe process."""
    root = _sealed_v5_wrapper_repo(tmp_path)
    runner = root / "scripts/eval/run_registered_paper_downstream.py"
    runner.write_text(runner.read_text(encoding="utf-8") + "\n# dirty\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(root / "scripts/eval/launch_registered_v5_downstream.sh"),
            "--protocol",
            "v5_osm_assisted",
            "--family",
            "full_150",
            "--dry-run",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "runtime source tree differs from Git HEAD" in result.stderr


def test_v5_runtime_source_gate_covers_runner_and_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered probe records and verifies every Python source it executes."""
    seen: list[Path] = []
    monkeypatch.setattr(registered, "verify_git_head_file", lambda path: seen.append(Path(path)))

    sources = registered.verify_v5_runtime_sources()

    assert set(sources) == {
        "registered_v5_encoder_checkpoint.py",
        "run_registered_paper_downstream.py",
        "registered_v5_matrix.py",
        "run_strong_downstream_benchmark.py",
        "run_traditional_ml_benchmark.py",
    }
    assert {path.name for path in seen} == set(sources)
    assert all(len(value) == 64 for value in sources.values())


def test_precompute_embeddings_export_name_is_a_safe_child_directory(tmp_path: Path) -> None:
    """The sharded exporter accepts only one literal child directory name."""
    root = Path(__file__).resolve().parents[1]
    export_paths_script = repr(str(root / "downstreams/scripts/export_paths.py"))
    code = (
        "import importlib.util, sys; from pathlib import Path; "
        f"spec = importlib.util.spec_from_file_location('v5_export_paths', {export_paths_script}); "
        "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module); "
        "print(module.resolve_export_root(Path(sys.argv[1]), sys.argv[2], 'fallback'))"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": f"{root / 'downstreams'}:{os.environ.get('PYTHONPATH', '')}",
    }
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; {code}", str(tmp_path), "sealed-export"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "sealed-export")

    traversal = subprocess.run(
        [sys.executable, "-c", f"import sys; {code}", str(tmp_path), ".."],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert traversal.returncode != 0
    assert "literal child directory" in traversal.stderr

    link = tmp_path / "linked-export"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    symlink = subprocess.run(
        [sys.executable, "-c", f"import sys; {code}", str(tmp_path), link.name],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert symlink.returncode != 0
    assert "symbolic link" in symlink.stderr


def test_v5_matrix_preflight_loader_avoids_training_runtime_imports() -> None:
    """Dry-run admission must not import torch, rasterio, or sklearn training dependencies."""
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from scripts.eval.registered_v5_matrix import load_registered_v5_matrix; "
            "from pathlib import Path; "
            "matrix = load_registered_v5_matrix("
            "Path('configs/eval/rse_v5_osm_assisted_matrix.json')); "
            "assert 'full_150' in matrix['families']; "
            "assert not {'torch', 'rasterio', 'sklearn'} & set(sys.modules)"
        ),
    ]

    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def _sealed_v5_queue_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create the small committed repository needed to exercise the queue's Git gate."""
    source_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "queue-repo"
    required_paths = [
        Path("scripts/experiments/run_registered_v5_paper_queue.sh"),
        Path("scripts/eval/registered_v5_encoder_checkpoint.py"),
        *(
            Path("configs/paper_registered_v5_20260726")
            / f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
            for fold in range(5)
        ),
        Path("configs/eval/rse_v5_osm_assisted_matrix.json"),
        Path("configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json"),
        Path("configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json"),
        Path("configs/eval/haidian_paper_v5_normalization_statistics.json"),
        Path("configs/eval/registered_embedding_exports_v5_20260726.json"),
    ]
    for relative_path in required_paths:
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, target)

    for command in (
        ["git", "init", "--quiet"],
        ["git", "config", "user.email", "queue-test@example.invalid"],
        ["git", "config", "user.name", "Queue Test"],
        ["git", "add", "."],
        ["git", "commit", "--quiet", "-m", "seal queue fixture"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    return repo, repo / "scripts/experiments/run_registered_v5_paper_queue.sh"


def _write_verified_resume_attempt(output_root: Path) -> Path:
    attempt = output_root / "paper_registered_v5_full_150_fold0_20260726" / "attempt_1"
    attempt.mkdir(parents=True)
    source = attempt / "recovery.pt"
    source.write_bytes(b"verified recovery checkpoint")
    checkpoint_sha256 = _sha256(source)
    snapshot_dir = attempt / "verified_checkpoints"
    snapshot_dir.mkdir()
    checkpoint = snapshot_dir / f"recovery_{checkpoint_sha256}.pt"
    shutil.copyfile(source, checkpoint)
    _write_json(
        attempt / "attempt_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "family": "full_150",
            "fold": 0,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_hashes": {checkpoint.name: checkpoint_sha256},
        },
    )
    verification_dir = attempt / "checkpoint_verifications"
    verification_dir.mkdir()
    _write_json(
        verification_dir / f"{checkpoint.name}.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "source_checkpoint_path": str(source.resolve()),
            "created_at_ns": time.time_ns(),
        },
    )
    return attempt


def _bind_recovery_parent_to_registered_protocol(repo: Path, attempt: Path, fold: int = 0) -> None:
    config = (
        repo
        / "configs/paper_registered_v5_20260726"
        / f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
    )
    assets = {
        "matrix": repo / "configs/eval/rse_v5_osm_assisted_matrix.json",
        "spatial_split": repo / "configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json",
        "subset_registry": repo / "configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json",
        "statistics_registry": repo / "configs/eval/haidian_paper_v5_normalization_statistics.json",
        "embedding_export_registry": repo / "configs/eval/registered_embedding_exports_v5_20260726.json",
    }
    manifest_path = attempt / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest["source_config"] = {"path": str(config.resolve()), "sha256": _sha256(config)}
    launch = source
    launch["experiment"]["output_dir"] = str(attempt.resolve())
    launch["experiment"]["wandb_run_name"] = f"{launch['experiment']['wandb_run_name']}__{attempt.name}"
    launch_path = attempt / "launch_config.yaml"
    launch_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")
    manifest["launch_config"] = {"path": str(launch_path.resolve()), "sha256": _sha256(launch_path)}
    manifest["protocol_assets"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in assets.items()
    }
    sidecar = Path(source["data"]["paper_manifest_audit"])
    manifest["manifest_sidecar"] = {
        "path": str(sidecar.resolve()),
        "sha256": _sha256(sidecar),
    }
    _write_json(manifest_path, manifest)


def test_v5_encoder_queue_dry_run_requires_committed_queue_and_runs_from_root(
    tmp_path: Path,
) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_cwd = tmp_path / "python-cwds.txt"
    python_wrapper = bin_dir / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$PWD" >> "$QUEUE_TEST_PYTHON_CWDS"\n'
        'exec "$QUEUE_TEST_REAL_PYTHON" "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "QUEUE_TEST_PYTHON_CWDS": str(python_cwd),
        "QUEUE_TEST_REAL_PYTHON": sys.executable,
    }

    result = subprocess.run(
        [str(queue), "--dry-run"],
        cwd=outside,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert python_cwd.read_text(encoding="utf-8").splitlines() == [str(repo)]
    jobs = [line for line in result.stdout.splitlines() if line.startswith("ENCODER ")]
    assert len(jobs) == 5
    assert all("family=full_150" in line for line in jobs)
    assert all("paper_registered_v5_full_150_fold" in line for line in jobs)
    assert "lane=0 devices=0,1 fold=0" in jobs[0]
    assert "lane=1 devices=2,3 fold=1" in jobs[1]
    assert "lane=2 devices=4,5 fold=2" in jobs[2]
    assert "lane=0 devices=0,1 fold=3" in jobs[3]
    assert "lane=1 devices=2,3 fold=4" in jobs[4]

    queue.write_text(queue.read_text(encoding="utf-8") + "\n# dirty queue\n", encoding="utf-8")
    dirty = subprocess.run(
        [str(queue), "--dry-run"],
        cwd=outside,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert dirty.returncode != 0
    assert "queue script differs from Git HEAD" in dirty.stderr


def test_v5_encoder_queue_only_fold_dry_run_isolates_the_selected_lane(tmp_path: Path) -> None:
    """A recovery follow-up can schedule Fold 3 without touching the other active lanes."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)

    result = subprocess.run(
        [str(queue), "--only-fold", "3", "--dry-run", "--output-root", str(tmp_path / "outputs")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("ENCODER ")]
    assert jobs == [
        "ENCODER protocol=v5_osm_assisted family=full_150 lane=0 devices=0,1 fold=3 "
        "config="
        f"{repo}/configs/paper_registered_v5_20260726/"
        "paper_registered_v5_full_150_fold3_20260726.yaml attempt=1 resume=fresh"
    ]


def test_v5_encoder_queue_only_fold_refuses_an_active_shared_lane(tmp_path: Path) -> None:
    """Fold 3 must not launch while Fold 0 still owns its NPU pair and rendezvous port."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    pgrep.chmod(0o755)

    result = subprocess.run(
        [str(queue), "--only-fold", "3", "--output-root", str(output_root)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "active" in result.stderr.lower()
    assert "fold 0" in result.stderr.lower()
    assert not output_root.exists()


def test_v5_encoder_queue_only_fold_refuses_an_existing_lane_lease(tmp_path: Path) -> None:
    """The lane lease closes the race between an idle check and torchrun launch."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, content in {
        "pgrep": "#!/usr/bin/env bash\nexit 1\n",
        "torchrun": "#!/usr/bin/env bash\necho unexpected-launch >&2\nexit 1\n",
    }.items():
        executable = bin_dir / name
        executable.write_text(content, encoding="utf-8")
        executable.chmod(0o755)

    lock_root = repo / ".pytest_registered_v5_locks"
    lock_path = lock_root / "lane_0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    ready = tmp_path / "lane-lock-ready"
    holder = subprocess.Popen(
        ["flock", "--no-fork", "-n", str(lock_path), "sh", "-c", f"touch {ready}; sleep 30"]
    )
    try:
        for _ in range(100):
            if ready.is_file():
                break
            time.sleep(0.01)
        assert ready.is_file(), "lane-lease holder did not acquire its lock"
        result = subprocess.run(
            [str(queue), "--only-fold", "3", "--output-root", str(output_root)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        )
    finally:
        holder.terminate()
        holder.wait(timeout=10)

    assert result.returncode != 0
    assert "lease" in result.stderr.lower()
    assert "unexpected-launch" not in result.stderr


def test_v5_encoder_queue_rejects_only_fold_with_verification_mode(tmp_path: Path) -> None:
    """Verification is read-only and must not silently discard a scheduling argument."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    attempt = _write_verified_resume_attempt(tmp_path / "outputs")

    result = subprocess.run(
        [str(queue), "--only-fold", "3", "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_v5_encoder_queue_rejects_a_dirty_canonical_checkpoint_helper(tmp_path: Path) -> None:
    """Queue admission seals the helper that decides whether a fold is canonical."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    helper = repo / "scripts/eval/registered_v5_encoder_checkpoint.py"
    helper.write_text(helper.read_text(encoding="utf-8") + "\n# dirty helper\n", encoding="utf-8")

    result = subprocess.run(
        [str(queue), "--dry-run"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "registered_v5_encoder_checkpoint.py" in result.stderr
    assert "differs from Git HEAD" in result.stderr


def test_v5_encoder_queue_rejects_an_untracked_queue_script(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    subprocess.run(["git", "rm", "--cached", str(queue.relative_to(repo))], cwd=repo, check=True)

    result = subprocess.run(
        [str(queue), "--dry-run"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "queue script is not Git tracked" in result.stderr


def test_v5_encoder_queue_resume_verification_is_attempt_local(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    checkpoint = next((attempt / "verified_checkpoints").glob("*.pt"))
    verification_dir = attempt / "checkpoint_verifications"

    accepted = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "VERIFIED_RESUME" in accepted.stdout

    resumed = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--resume-fold",
            "0",
            "--resume-attempt",
            "1",
            "--dry-run",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "attempt=1 resume=" in resumed.stdout

    restarted = subprocess.run(
        [str(queue), "--output-root", str(output_root), "--dry-run"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert restarted.returncode == 0, restarted.stderr
    restart_job = next(line for line in restarted.stdout.splitlines() if "fold=0" in line)
    assert "attempt=2 resume=fresh" in restart_job

    manifest_path = attempt / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fold"] = 1
    _write_json(manifest_path, manifest)
    wrong_fold = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--resume-fold",
            "0",
            "--resume-attempt",
            "1",
            "--dry-run",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_fold.returncode != 0
    assert "fold" in wrong_fold.stderr.lower()

    manifest["fold"] = 0
    _write_json(manifest_path, manifest)
    checkpoint.write_bytes(b"modified after verification")
    modified = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert modified.returncode != 0
    assert "hash" in modified.stderr.lower()
    checkpoint.write_bytes((attempt / "recovery.pt").read_bytes())

    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"foreign checkpoint")
    _write_json(
        verification_dir / "foreign.pt.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(foreign.resolve()),
            "checkpoint_sha256": _sha256(foreign),
        },
    )
    foreign_result = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert foreign_result.returncode != 0
    assert "verified_checkpoints" in foreign_result.stderr.lower()


def test_v5_encoder_queue_selects_the_newest_verified_recovery(tmp_path: Path) -> None:
    """Resume selection is chronological, not an accidental checkpoint-filename ordering."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    newer_payload = b"newer verified recovery"
    newer_name = f"recovery_{hashlib.sha256(newer_payload).hexdigest()}.pt"
    newer = attempt / "verified_checkpoints" / newer_name
    newer.write_bytes(newer_payload)
    verification_dir = attempt / "checkpoint_verifications"
    _write_json(
        verification_dir / f"{newer.name}.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(newer.resolve()),
            "checkpoint_sha256": _sha256(newer),
            "source_checkpoint_path": str(attempt / "recovery.pt"),
            "created_at_ns": time.time_ns() + 1,
        },
    )

    result = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(newer.resolve()) in result.stdout


def test_v5_encoder_queue_rejects_a_mutable_or_malformed_verification_record(
    tmp_path: Path,
) -> None:
    """Only queue-sealed snapshots with the complete record contract authorize recovery."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    mutable = attempt / "mutable.pt"
    mutable.write_bytes(b"not an immutable queue snapshot")
    _write_json(
        attempt / "checkpoint_verifications" / "mutable.pt.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(mutable.resolve()),
            "checkpoint_sha256": _sha256(mutable),
            "source_checkpoint_path": str(mutable.resolve()),
            "created_at_ns": time.time_ns() + 1,
        },
    )

    result = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "verified_checkpoints" in result.stderr


def test_v5_encoder_queue_rejects_an_unhashed_snapshot(
    tmp_path: Path,
) -> None:
    """Only content-addressed snapshots can enter the verified recovery set."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    snapshot = attempt / "verified_checkpoints" / "unhashed.pt"
    snapshot.write_bytes(b"not content-addressed")
    _write_json(
        attempt / "checkpoint_verifications" / "unhashed.pt.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(snapshot.resolve()),
            "checkpoint_sha256": _sha256(snapshot),
            "source_checkpoint_path": str(attempt / "recovery.pt"),
            "created_at_ns": time.time_ns(),
        },
    )

    result = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "content-addressed" in result.stderr


def test_v5_encoder_queue_rejects_a_verified_record_without_creation_time(tmp_path: Path) -> None:
    """Chronological selection requires an explicit immutable creation time."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    record_path = next((attempt / "checkpoint_verifications").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("created_at_ns")
    _write_json(record_path, record)

    result = subprocess.run(
        [str(queue), "--verify-attempt", str(attempt)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "creation timestamp" in result.stderr


def test_v5_encoder_queue_recovers_new_attempt_from_named_valid_parent_snapshot(
    tmp_path: Path,
) -> None:
    """A malformed sibling record cannot erase an earlier, fully verified snapshot."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint = next((parent / "verified_checkpoints").glob("*.pt"))
    _write_json(
        parent / "checkpoint_verifications" / "malformed.pt.json",
        {
            "schema_version": 1,
            "attempt_dir": str(parent.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_checkpoint_path": str(parent / "recovery.pt"),
        },
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "config=\n"
        "previous=\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$previous\" == --config ]]; then config=\"$arg\"; fi\n"
        "  previous=\"$arg\"\n"
        "done\n"
        "printf recovered > \"$(dirname \"$config\")/best.pt\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint.name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    recovered = output_root / "paper_registered_v5_full_150_fold0_20260726" / "attempt_2"
    manifest = json.loads((recovered / "attempt_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_attempt"] == str(parent.resolve())
    assert manifest["recovery_parent"]["checkpoint_path"] == str(checkpoint.resolve())
    assert manifest["recovery_parent"]["checkpoint_sha256"] == _sha256(checkpoint)
    record_path = parent / "checkpoint_verifications" / f"{checkpoint.name}.json"
    assert manifest["recovery_parent"]["verification_record_sha256"] == _sha256(record_path)
    assert manifest["recovery_parent"]["attempt_manifest_sha256"] == _sha256(
        parent / "attempt_manifest.json"
    )
    assert manifest["recovery_parent"]["launch_config_sha256"] == _sha256(
        parent / "launch_config.yaml"
    )
    assert manifest["recovery_parent"]["manifest_sidecar_sha256"] == _sha256(
        Path(manifest["recovery_parent"]["manifest_sidecar_path"])
    )
    assert (recovered.parent / "canonical_attempt.json").is_file()


def test_v5_encoder_queue_recovery_rejects_parent_with_wrong_registered_config(
    tmp_path: Path,
) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint = next((parent / "verified_checkpoints").glob("*.pt"))
    manifest_path = parent / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_config"] = {
        "path": str(repo / "configs/paper_registered_v5_20260726/paper_registered_v5_full_150_fold0_20260726.yaml"),
        "sha256": "0" * 64,
    }
    _write_json(manifest_path, manifest)

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint.name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source config" in result.stderr.lower()
    assert not (parent.parent / "attempt_2").exists()


def test_v5_encoder_queue_recovery_rejects_parent_with_changed_launch_config(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint = next((parent / "verified_checkpoints").glob("*.pt"))
    launch_path = parent / "launch_config.yaml"
    launch = yaml.safe_load(launch_path.read_text(encoding="utf-8"))
    launch["data"]["batch_size"] = 99
    launch_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint.name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "launch config" in result.stderr.lower()


def test_v5_encoder_queue_recovery_rejects_parent_with_changed_data_sidecar(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint = next((parent / "verified_checkpoints").glob("*.pt"))
    manifest_path = parent / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_sidecar"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint.name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "manifest sidecar" in result.stderr.lower()


def test_v5_encoder_queue_recovery_rejects_symlinked_parent_attempt(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint_name = next((parent / "verified_checkpoints").glob("*.pt")).name
    external = tmp_path / "external_attempt"
    parent.rename(external)
    parent.symlink_to(external, target_is_directory=True)

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint_name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()


def test_v5_encoder_queue_recovery_rejects_checkpoint_path_traversal(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)

    result = subprocess.run(
        [
            str(queue),
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            "../recovery.pt",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "snapshot filename" in result.stderr.lower()


def test_v5_encoder_queue_recovery_dry_run_does_not_create_an_attempt(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    parent = _write_verified_resume_attempt(output_root)
    _bind_recovery_parent_to_registered_protocol(repo, parent)
    checkpoint = next((parent / "verified_checkpoints").glob("*.pt"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    torchrun_trace = tmp_path / "torchrun_trace.txt"
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text(
        f"#!/usr/bin/env bash\nprintf called > {torchrun_trace}\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    result = subprocess.run(
        [
            str(queue),
            "--dry-run",
            "--output-root",
            str(output_root),
            "--recover-fold",
            "0",
            "--recover-from-attempt",
            "1",
            "--recover-checkpoint",
            checkpoint.name,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    assert "recovery_parent=1" in result.stdout
    assert not (parent.parent / "attempt_2").exists()
    assert not (repo / ".pytest_registered_v5_locks" / "lane_0.lock").exists()
    assert not torchrun_trace.exists()


def test_v5_encoder_queue_resume_writes_immutable_resume_logs(tmp_path: Path) -> None:
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    historical_log = attempt.parent / "logs" / "attempt_1.log"
    historical_log.parent.mkdir()
    historical_log.write_text("original attempt log\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\necho fake resume failure >&2\nexit 1\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    command = [
        str(queue),
        "--output-root",
        str(output_root),
        "--resume-fold",
        "0",
        "--resume-attempt",
        "1",
    ]

    first = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, check=False, env=environment
    )
    second = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, check=False, env=environment
    )

    assert first.returncode != 0
    assert second.returncode != 0
    assert historical_log.read_text(encoding="utf-8") == "original attempt log\n"
    assert (attempt / "logs" / "resume_1.log").read_text(
        encoding="utf-8"
    ) == "fake resume failure\n"
    assert (attempt / "logs" / "resume_2.log").read_text(
        encoding="utf-8"
    ) == "fake resume failure\n"


def test_v5_encoder_queue_prefers_same_attempt_resume_after_fresh_failure(tmp_path: Path) -> None:
    """A verified recovery must be resumed before the queue creates another fresh attempt."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "torchrun_trace.txt"
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"trace={trace}\n"
        "config=\n"
        "previous=\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$previous\" == --config ]]; then config=\"$arg\"; fi\n"
        "  previous=\"$arg\"\n"
        "done\n"
        "if [[ \" $* \" == *\" --resume \"* ]]; then\n"
        "  echo resume >> \"$trace\"\n"
        "  exit 1\n"
        "fi\n"
        "echo fresh >> \"$trace\"\n"
        "printf recovery > \"$(dirname \"$config\")/recovery.pt\"\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MAX_JOB_ATTEMPTS": "1",
    }

    result = subprocess.run(
        [str(queue), "--output-root", str(output_root)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0
    assert "resume" in trace.read_text(encoding="utf-8").splitlines()
    fold0 = output_root / "paper_registered_v5_full_150_fold0_20260726"
    assert (fold0 / "attempt_1" / "logs" / "resume_1.log").is_file()
    assert not (fold0 / "attempt_2").exists()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (fold0 / "attempt_1" / "status").glob("*.json")
    ]
    assert "log_seal_failed" not in {payload["event"] for payload in payloads}
    verification_records = list((fold0 / "attempt_1" / "checkpoint_verifications").glob("*.json"))
    assert len(verification_records) == 1
    verification = json.loads(verification_records[0].read_text(encoding="utf-8"))
    assert isinstance(verification["created_at_ns"], int)
    assert all(
        any(
            detail.startswith("checkpoint=") and detail != "checkpoint="
            for detail in payload["details"]
        )
        for payload in payloads
        if payload["event"] == "failed"
    )


def test_v5_encoder_queue_records_recovery_verification_reason_and_fails_closed(
    tmp_path: Path,
) -> None:
    """A failed verification must be immutable evidence, not a silent fresh retry."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    output_root = tmp_path / "outputs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)

    result = subprocess.run(
        [str(queue), "--output-root", str(output_root)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MAX_JOB_ATTEMPTS": "1"},
    )

    assert result.returncode != 0
    fold0 = output_root / "paper_registered_v5_full_150_fold0_20260726"
    assert not (fold0 / "attempt_2").exists()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (fold0 / "attempt_1" / "status").glob("*.json")
    ]
    verification_failure = next(
        payload for payload in payloads if payload["event"] == "recovery_verification_failed"
    )
    assert any(
        detail.startswith("reason=attempt has no verified same-attempt recovery checkpoint")
        for detail in verification_failure["details"]
    )


def test_v5_encoder_queue_concurrent_resumes_share_one_new_verified_snapshot(
    tmp_path: Path,
) -> None:
    """Two failed resume workers can seal the same newly-written recovery without corruption."""
    repo, queue = _sealed_v5_queue_repo(tmp_path)
    second_repo, second_queue = _sealed_v5_queue_repo(tmp_path / "second")
    output_root = tmp_path / "outputs"
    attempt = _write_verified_resume_attempt(output_root)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_torchrun = bin_dir / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "config=\n"
        "previous=\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$previous\" == --config ]]; then config=\"$arg\"; fi\n"
        "  previous=\"$arg\"\n"
        "done\n"
        "attempt=$(dirname \"$config\")\n"
        "barrier=\"$attempt/.concurrent_resume_barrier\"\n"
        "mkdir -p \"$barrier\"\n"
        "touch \"$barrier/$RANDOM.$RANDOM\"\n"
        "while (( $(find \"$barrier\" -type f | wc -l) < 2 )); do sleep 0.01; done\n"
        "dd if=/dev/zero of=\"$attempt/recovery.pt\" bs=1M count=32 conv=fsync status=none\n"
        "touch \"$barrier/sealed.$RANDOM.$RANDOM\"\n"
        "while (( $(find \"$barrier\" -type f -name 'sealed.*' | wc -l) < 2 )); do sleep 0.01; done\n"
        "if mkdir \"$barrier/prewriter\" 2>/dev/null; then\n"
        "  (\n"
        "  digest=$(sha256sum \"$attempt/recovery.pt\" | awk '{print $1}')\n"
        "  snapshot=\"$attempt/verified_checkpoints/recovery_${digest}.pt\"\n"
        "  mkdir -p \"$(dirname \"$snapshot\")\"\n"
        "  : > \"$snapshot\"\n"
        "  record=\"$attempt/checkpoint_verifications/$(basename \"$snapshot\").json\"\n"
        "  mkdir -p \"$(dirname \"$record\")\"\n"
        "  printf '{' > \"$record\"\n"
        "  touch \"$barrier/prewriter_ready\"\n"
        "  sleep 3\n"
        "  cat \"$attempt/recovery.pt\" > \"$snapshot\"\n"
        "  sleep 3\n"
        "  created_at_ns=$(date +%s%N)\n"
        "  printf '{\"schema_version\":1,\"attempt_dir\":\"%s\",\"checkpoint_path\":\"%s\","
        "\"checkpoint_sha256\":\"%s\",\"source_checkpoint_path\":\"%s\",\"created_at_ns\":%s}\n' "
        "\"$attempt\" \"$snapshot\" \"$digest\" \"$attempt/recovery.pt\" \"$created_at_ns\" > \"$record\"\n"
        "  ) &\n"
        "fi\n"
        "while [[ ! -f \"$barrier/prewriter_ready\" ]]; do sleep 0.01; done\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    command = [
        str(queue),
        "--output-root",
        str(output_root),
        "--resume-fold",
        "0",
        "--resume-attempt",
        "1",
    ]
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    first = subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    second_command = [str(second_queue), *command[1:]]
    second = subprocess.Popen(
        second_command,
        cwd=second_repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    first_stdout, first_stderr = first.communicate(timeout=30)
    second_stdout, second_stderr = second.communicate(timeout=30)

    assert first.returncode != 0
    assert second.returncode != 0
    combined = first_stdout + first_stderr + second_stdout + second_stderr
    assert "snapshot hash mismatch" not in combined
    assert "JSONDecodeError" not in combined
    records = list((attempt / "checkpoint_verifications").glob("*.json"))
    assert len(records) == 2
    newest = max(
        (json.loads(path.read_text(encoding="utf-8")) for path in records),
        key=lambda record: record["created_at_ns"],
    )
    assert newest["checkpoint_sha256"] == _sha256(attempt / "recovery.pt")
