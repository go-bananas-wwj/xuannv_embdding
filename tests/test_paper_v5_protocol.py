from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.data import build_paper_manifests, compute_statistics
from scripts.eval import register_registered_v5_embedding_export as registrar
from scripts.eval import run_registered_paper_downstream as registered
from scripts.experiments import build_clean_paper_configs, build_paper_subset_registry
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _sealed_v5_export_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "configs/paper_registered_v5_20260726/paper_registered_v5_full_150_fold0_20260726.yaml"
    )
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"sealed checkpoint")
    export_root = tmp_path / "export"
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
    return export_root, {"family": "full_150", "fold": 0, "region": "haidian", "month": "202604"}


def test_v5_registrar_rebuilds_export_entry_and_rejects_forged_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    before = registry_path.read_bytes()

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
    registrar.main()
    assert registry_path.read_bytes() == before
    assert "commit required" in capsys.readouterr().out

    forged = dict(entry)
    forged["checkpoint_sha256"] = "forged"
    pending["entry"] = forged
    pending["entry_sha256"] = registered._canonical_sha256(forged)
    _write_json(pending_path, pending)
    with pytest.raises(ValueError, match="does not match the sealed export"):
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

    with pytest.raises(ValueError, match="outside the registered v5 matrix"):
        registered.validate_registered_matrix_cell("v5_osm_assisted", aef_cell)
    with pytest.raises(ValueError, match="baseline is outside"):
        registered.validate_v5_matrix_comparator("matched_aef_v5", "full_150")


def _sealed_v5_wrapper_repo(tmp_path: Path) -> Path:
    """Create a minimal committed repository for V5 wrapper integration tests."""
    source_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "wrapper-repo"
    for relative in (
        Path("scripts/eval/export_registered_v5_paper_encoders.sh"),
        Path("scripts/eval/launch_registered_v5_downstream.sh"),
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
        for rejected_family in ("not_declared", "matched_aef_v5"):
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
    checkpoint = attempt / "verified_resume.pt"
    checkpoint.write_bytes(b"verified recovery checkpoint")
    checkpoint_sha256 = _sha256(checkpoint)
    _write_json(
        attempt / "attempt_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": "v5_osm_assisted",
            "family": "full_150",
            "fold": 0,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_hashes": {"verified_resume.pt": checkpoint_sha256},
        },
    )
    verification_dir = attempt / "checkpoint_verifications"
    verification_dir.mkdir()
    _write_json(
        verification_dir / "verified_resume.pt.json",
        {
            "schema_version": 1,
            "attempt_dir": str(attempt.resolve()),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        },
    )
    return attempt


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
    checkpoint = attempt / "verified_resume.pt"
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

    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"foreign checkpoint")
    _write_json(
        verification_dir / "verified_resume.pt.json",
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
    assert "same attempt" in foreign_result.stderr.lower()


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
        "#!/usr/bin/env bash\n" "echo fake resume failure >&2\n" "exit 1\n",
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
