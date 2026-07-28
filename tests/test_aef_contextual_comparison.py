from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.report import aggregate_aef_contextual_comparison as contextual
from scripts.report import admit_aef_contextual_comparison as admission
from scripts.report import compare_aef_contextual_protocols as comparison
from scripts.eval import run_registered_paper_downstream as registered


def _record(
    protocol: str, *, cell: tuple[str, str, int, int], test_ids: str | None = None
) -> dict[str, object]:
    task, shot, fold, seed = cell
    matrix = json.loads(contextual.AEF_MATRIX_PATH.read_text(encoding="utf-8"))
    return {
        "result_id": f"{protocol}-result",
        "metric_provenance": {
            "protocol_id": protocol,
            "task": task,
            "shot": shot,
            "fold": fold,
            "shot_seed": seed,
            "label_sha256": matrix["labels"][task]["tree_sha256"],
            "spatial_split_sha256": matrix["spatial_split_sha256"],
            "manifest_sha256": matrix["manifest_sha256"],
            "statistics_registry_sha256": matrix["statistics_registry_sha256"],
            "shot_manifest_sha256": "schedule",
            "test_patch_ids_sha256": test_ids
            or hashlib.sha256("patch_000000".encode("utf-8")).hexdigest(),
            "per_patch_confusion": {"patch_000000": {"tp": 1, "fp": 0, "fn": 0, "tn": 1}},
            "per_patch_target_support_sha256": {"patch_000000": "target-support"},
            "probe": {
                "head": "conv3x3_64_128_64",
                "epochs": 80,
                "batch_size": 8,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "final_epoch_only": True,
            },
            "validation_threshold_selection": {
                "grid_start": 0.001,
                "grid_stop_exclusive": 1.0,
                "grid_step": 0.001,
                "candidate_count": 999,
            },
            "f1_at_threshold": 0.5,
            "ap": 0.6,
            "auc_roc": 0.7,
            "miou": 0.4,
            "precision": 0.55,
            "recall": 0.45,
        },
    }


def _with_protocol_identity(record: dict[str, object], protocol: str) -> dict[str, object]:
    payload = record["metric_provenance"]
    assert isinstance(payload, dict)
    if protocol == "aef_annual_2025_contextual":
        payload.update(
            {
                "baseline_id": "aef_annual_2025",
                "baseline_output_identifier": "annual_2025",
                "comparison_matrix_sha256": contextual.contextual_matrix_sha256(),
                "time_inequivalent_contextual": True,
                "evidence_scope": "osm_assisted_spatial_readout",
            }
        )
    else:
        payload.update(
            {
                "family": "full_150",
                # The immutable embedding registry, rather than duplicated export metadata,
                # is the canonical owner of the monthly slot.
                "embedding_export": {"protocol_id": "v5_osm_assisted"},
                "provenance": {
                    "family": "full_150",
                    "config_sha256": "config",
                    "checkpoint_sha256": "checkpoint",
                },
                "embedding_registry": {
                    "family": "full_150",
                    "encoder_fold": payload["fold"],
                    "month": "202604",
                    "protocol_id": "v5_osm_assisted",
                    "patch_count": 320,
                    "config_sha256": "config",
                    "checkpoint_sha256": "checkpoint",
                    "manifest_sha256": payload["manifest_sha256"],
                    "embedding_file_index_sha256": "index",
                    "canonical_export_provenance_sha256": "canonical",
                },
            }
        )
    return record


def test_contextual_pairing_resolves_xuannv_statistics_from_sealed_export() -> None:
    """V5 probe records may inherit normalization identity from their sealed export."""
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    payload = xuannv[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict) and isinstance(payload["embedding_export"], dict)
    statistics_sha = payload.pop("statistics_registry_sha256")
    payload["embedding_export"]["statistics_registry_sha256"] = statistics_sha

    contextual.verify_contextual_pairing(aef, xuannv)


def _matrix(protocol: str) -> dict[tuple[str, str, int, int], dict[str, object]]:
    return {
        (task, shot, fold, seed): _with_protocol_identity(
            _record(protocol, cell=(task, shot, fold, seed)), protocol
        )
        for task in ("building", "road", "water")
        for shot in ("5", "10")
        for fold in range(5)
        for seed in (42, 43, 44)
    }


def test_contextual_pairing_accepts_only_the_registered_protocol_pair() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")

    contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_pairing_rejects_mismatched_test_patch_identity() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    xuannv[("building", "5", 0, 42)] = _with_protocol_identity(
        _record("v5_osm_assisted", cell=("building", "5", 0, 42), test_ids="other"),
        "v5_osm_assisted",
    )

    with pytest.raises(ValueError, match="test_patch_ids_sha256"):
        contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_pairing_rejects_same_patch_id_with_different_pixel_support() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    payload = xuannv[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict) and isinstance(payload["per_patch_confusion"], dict)
    payload["per_patch_confusion"]["patch_000000"] = {"tp": 1, "fp": 1, "fn": 1, "tn": 99}

    with pytest.raises(ValueError, match="valid pixel support"):
        contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_pairing_rejects_mismatched_target_support_hash() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    payload = xuannv[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict) and isinstance(payload["per_patch_target_support_sha256"], dict)
    payload["per_patch_target_support_sha256"]["patch_000000"] = "different-support"

    with pytest.raises(ValueError, match="target support hashes"):
        contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_pairing_rejects_incomplete_or_payload_key_mismatch() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    aef.pop(("water", "10", 4, 44))
    xuannv.pop(("water", "10", 4, 44))

    with pytest.raises(ValueError, match="complete registered 90-cell"):
        contextual.verify_contextual_pairing(aef, xuannv)

    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    aef[("building", "5", 0, 42)] = _with_protocol_identity(
        _record("aef_annual_2025_contextual", cell=("road", "5", 0, 42)),
        "aef_annual_2025_contextual",
    )
    with pytest.raises(ValueError, match="cell key"):
        contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_pairing_rejects_wrong_baseline_or_probe_contract() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    payload = aef[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict)
    payload["baseline_id"] = "other"
    with pytest.raises(ValueError, match="baseline identity"):
        contextual.verify_contextual_pairing(aef, xuannv)

    aef = _matrix("aef_annual_2025_contextual")
    payload = xuannv[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict) and isinstance(payload["probe"], dict)
    payload["probe"]["epochs"] = 81
    with pytest.raises(ValueError, match="probe contract"):
        contextual.verify_contextual_pairing(aef, xuannv)


def test_contextual_matrix_identity_requires_git_head(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(contextual, "verify_git_head_file", lambda path: calls.append(path))

    contextual.contextual_matrix_sha256()

    assert calls == [contextual.AEF_MATRIX_PATH]


def test_contextual_summary_uses_seed_level_fold_means() -> None:
    records = _matrix("aef_annual_2025_contextual")
    for cell, record in records.items():
        payload = record["metric_provenance"]
        assert isinstance(payload, dict)
        payload["f1_at_threshold"] = 0.4 + 0.01 * cell[2] + 0.02 * (cell[3] - 42)

    summary = contextual.summarize_contextual_records(records)

    f1 = summary["building|5"]["metrics"]["f1_at_threshold"]
    assert f1["n_fold_seed_runs"] == 15
    assert f1["n_seeds"] == 3
    assert f1["mean"] == pytest.approx(0.44)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True])
def test_contextual_summary_rejects_non_finite_metrics(invalid: object) -> None:
    records = _matrix("aef_annual_2025_contextual")
    payload = records[("building", "5", 0, 42)]["metric_provenance"]
    assert isinstance(payload, dict)
    payload["ap"] = invalid

    with pytest.raises(ValueError, match="finite"):
        contextual.summarize_contextual_records(records)


def test_contextual_comparison_requires_the_registered_bootstrap_contract() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")

    with pytest.raises(ValueError, match="exactly 10000"):
        comparison.compare_contextual_records(aef, xuannv, n_resamples=9999, seed=20260725)

    with pytest.raises(ValueError, match="random seed"):
        comparison.compare_contextual_records(aef, xuannv, n_resamples=10000, seed=7)


def test_contextual_admission_snapshot_seals_each_registered_result_identity() -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    for name, matrix in (("aef", aef), ("xuannv", xuannv)):
        for index, record in enumerate(matrix.values()):
            record.update(
                {
                    "result_id": f"{name}-{index:03d}",
                    "artifact_sha256": f"artifact-{name}-{index}",
                    "metrics_sha256": f"metrics-{name}-{index}",
                    "registry_entry_sha256": f"entry-{name}-{index}",
                }
            )

    snapshot = admission.build_contextual_input_identity_snapshot(aef, xuannv)

    assert snapshot["baseline_result_count"] == 90
    assert snapshot["candidate_result_count"] == 90
    assert snapshot["baseline_input_results"][0]["result_id"].startswith("aef-")
    assert snapshot["candidate_input_results"][0]["result_id"].startswith("xuannv-")


def test_contextual_comparison_uses_the_registered_pairing_and_hierarchy(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    aef = _matrix("aef_annual_2025_contextual")
    xuannv = _matrix("v5_osm_assisted")
    bounds = {
        "patch_000000": (0.0, 0.0, 1.0, 1.0),
        "patch_000001": (1.0, 0.0, 2.0, 1.0),
        "patch_000002": (0.0, 1.0, 1.0, 2.0),
        "patch_000003": (1.0, 1.0, 2.0, 2.0),
    }
    for matrix in (aef, xuannv):
        for record in matrix.values():
            payload = record["metric_provenance"]
            assert isinstance(payload, dict)
            payload["per_patch_confusion"] = {
                patch_id: {"tp": 2, "fp": 1, "fn": 1, "tn": 2} for patch_id in bounds
            }
            payload["per_patch_target_support_sha256"] = {
                patch_id: f"target-{patch_id}" for patch_id in bounds
            }
            payload["test_patch_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(bounds)).encode("utf-8")
            ).hexdigest()
    metadata = tmp_path / "patches.json"
    metadata.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(comparison.bootstrap, "PROTOCOL_N_RESAMPLES", 1)
    monkeypatch.setattr(comparison.bootstrap, "validate_paper_patch_metadata", lambda path: None)
    monkeypatch.setattr(comparison.bootstrap, "load_patch_bounds", lambda path: bounds)

    report = comparison.compare_contextual_records(
        aef, xuannv, n_resamples=1, seed=20260725, patch_metadata_path=metadata
    )

    assert report["comparison_direction"] == "xuannv_minus_aef"
    assert report["bootstrap"]["hierarchy"] == [
        "fold",
        "complete_2x2_geographic_cluster",
        "support_schedule_seed",
    ]
    assert set(report["comparisons"]) == {
        "building|5", "building|10", "road|5", "road|10", "water|5", "water|10"
    }
    cell_summary = report["point_estimates"]["building|5"]
    assert cell_summary["uncertainty_rule"] == "bootstrap_ci_for_confusion_metrics_only"
    assert cell_summary["baseline"]["metrics"]["auc_roc"]["mean"] == pytest.approx(0.7)
    assert cell_summary["candidate_minus_baseline"]["f1_at_threshold"] == pytest.approx(0.0)
