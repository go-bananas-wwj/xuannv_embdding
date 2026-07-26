from __future__ import annotations

import json

import pytest

from scripts.eval import run_registered_paper_downstream as registered
from scripts.report import aggregate_registered_paper_results as aggregate
from scripts.report import paired_spatial_bootstrap as bootstrap


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
