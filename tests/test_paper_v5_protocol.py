from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.data import build_paper_manifests, compute_statistics
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
