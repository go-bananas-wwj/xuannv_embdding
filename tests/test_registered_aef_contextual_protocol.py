from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_protocol_module():
    path = Path(__file__).parents[1] / "scripts/eval/registered_aef_contextual.py"
    spec = importlib.util.spec_from_file_location("registered_aef_contextual", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_matrix(path: Path, schedule_root: Path) -> None:
    schedules = []
    for task in ("building", "road", "water"):
        for fold in range(5):
            for seed in (42, 43, 44):
                item = schedule_root / f"{task}_fold{fold}_seed{seed}.json"
                item.write_text(
                    json.dumps(
                        {
                            "schema_version": 3,
                            "task": task,
                            "fold": fold,
                            "seed": seed,
                            "protocol_id": "v5_osm_assisted",
                            "split_sha256": "a" * 64,
                            "label_sha256": "d" * 64,
                            "rule": {
                                "support_unit": "mixed_class_labeled_patch",
                                "min_positive_pixels": 64,
                                "min_background_pixels": 64,
                                "selection": "deterministic_nested_prefix",
                            },
                            "sets": {
                                "5": {"train_patch_ids": ["a"] * 5},
                                "10": {"train_patch_ids": ["a"] * 10},
                            },
                        }
                    )
                )
                schedules.append(item)
    index = {
        "split_sha256": "a" * 64,
        "schedules": [
            {
                "path": item.name,
                "sha256": _sha256(item),
                "task": item.name.split("_fold", maxsplit=1)[0],
                "fold": int(item.name.split("_fold", maxsplit=1)[1].split("_seed", maxsplit=1)[0]),
                "seed": int(item.stem.rsplit("_seed", maxsplit=1)[1]),
            }
            for item in sorted(schedules)
        ],
    }
    index_path = schedule_root / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True))
    digest = hashlib.sha256()
    for item in sorted(schedules):
        digest.update(item.name.encode() + b"\0")
        digest.update(_sha256(item).encode() + b"\n")
    matrix = {
        "schema_version": 1,
        "comparison_id": "test",
        "baseline_id": "aef_annual_2025",
        "baseline_protocol_id": "aef_annual_2025_contextual",
        "candidate_family": "full_150",
        "candidate_protocol_id": "v5_osm_assisted",
        "baseline_output_identifier": "annual_2025",
        "candidate_month": "202604",
        "time_inequivalent_contextual": True,
        "temporal_statement": (
            "Annual AEF 2025 versus monthly Xuannv 2026-04 "
            "under an aligned spatial downstream protocol."
        ),
        "spatial_split": "split.json",
        "spatial_split_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "statistics_registry_sha256": "c" * 64,
        "label_root": "/labels",
        "labels": {task: {"tree_sha256": "d" * 64} for task in ("building", "road", "water")},
        "probe": {
            "head": "conv3x3_64_128_64",
            "embedding_channels": 64,
            "epochs": 80,
            "batch_size": 8,
            "optimizer": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "standardization": "support_only",
            "epoch_selection": "final_epoch_only",
            "loss": {"name": "BCEWithLogitsLoss"},
            "threshold": {
                "split": "validation_only",
                "metric": "f1",
                "start": 0.001,
                "stop_exclusive": 1.0,
                "step": 0.001,
                "candidate_count": 999,
                "tie_break": "greatest_threshold",
            },
        },
        "tasks": ["building", "road", "water"],
        "shots": ["5", "10"],
        "folds": [0, 1, 2, 3, 4],
        "seeds": [42, 43, 44],
        "expected_job_count": 90,
        "shot_schedule": {
            "root": str(schedule_root),
            "index_file": "index.json",
            "index_sha256": _sha256(index_path),
            "schedule_file_count": 45,
            "ordered_filename_sha256_set_sha256": digest.hexdigest(),
            "protocol_id": "v5_osm_assisted",
            "generator": "generator.py",
            "generator_sha256": "e" * 64,
            "selection": "registered_mixed_positive_background_patch_schedule",
        },
        "evidence_scope": "osm_assisted_spatial_readout",
        "prohibited_claims": [
            "matched_temporal_information",
            "matched_inputs",
            "labelled_patch_efficiency_against_information_matched_baseline",
            "independent_transfer",
            "geographically_unseen_pretraining",
            "contemporaneous_ground_truth",
            "monthly_change_evidence",
            "global_superiority",
        ],
    }
    path.write_text(json.dumps(matrix, sort_keys=True))


def test_load_matrix_rejects_modified_shot_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)

    matrix = module.load_contextual_matrix(matrix_path)
    assert matrix["expected_job_count"] == 90
    (schedule_root / "road_fold1_seed43.json").write_text("tampered")

    with pytest.raises(ValueError, match="shot schedule"):
        module.verify_frozen_shot_schedules(matrix)


def test_load_matrix_rejects_non_contextual_temporal_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    matrix = json.loads(matrix_path.read_text())
    matrix["time_inequivalent_contextual"] = False
    matrix_path.write_text(json.dumps(matrix))
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)

    with pytest.raises(ValueError, match="time-inequivalent"):
        module.load_contextual_matrix(matrix_path)


def test_load_matrix_rejects_noncanonical_90_cell_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    matrix = json.loads(matrix_path.read_text())
    matrix["candidate_family"] = "unregistered_family"
    matrix_path.write_text(json.dumps(matrix))
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)

    with pytest.raises(ValueError, match="identity"):
        module.load_contextual_matrix(matrix_path)


def test_load_matrix_rejects_changed_contextual_identity_or_claim_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    matrix = json.loads(matrix_path.read_text())
    matrix["candidate_month"] = "202605"
    matrix_path.write_text(json.dumps(matrix))
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)

    with pytest.raises(ValueError, match="identity"):
        module.load_contextual_matrix(matrix_path)


def test_verify_schedules_rejects_wrong_cell_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)
    matrix = module.load_contextual_matrix(matrix_path)
    index_path = schedule_root / "index.json"
    index = json.loads(index_path.read_text())
    index["schedules"][0]["task"] = "wrong_task"
    index_path.write_text(json.dumps(index, sort_keys=True))
    matrix["shot_schedule"]["index_sha256"] = _sha256(index_path)

    with pytest.raises(ValueError, match="cell identity"):
        module.verify_frozen_shot_schedules(matrix)


def test_runtime_source_hashes_reject_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    repo_root = tmp_path / "repo"
    (repo_root / "scripts/eval").mkdir(parents=True)
    reader = repo_root / "scripts/eval/reader.py"
    generator = repo_root / "scripts/eval/generator.py"
    reader.write_text("reader-v1")
    generator.write_text("generator-v1")
    matrix = json.loads(matrix_path.read_text())
    matrix["probe"]["reader_sources"] = {"scripts/eval/reader.py": _sha256(reader)}
    matrix["shot_schedule"]["generator"] = "scripts/eval/generator.py"
    matrix["shot_schedule"]["generator_sha256"] = _sha256(generator)
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)
    loaded = module.load_contextual_matrix(matrix_path)
    module.verify_runtime_source_hashes(loaded, repo_root)
    reader.write_text("reader-v2")

    with pytest.raises(ValueError, match="runtime source"):
        module.verify_runtime_source_hashes(loaded, repo_root)


def test_verify_schedules_rejects_missing_protocol_and_shot_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix_path = tmp_path / "matrix.json"
    schedule_root = tmp_path / "schedules"
    schedule_root.mkdir()
    _write_matrix(matrix_path, schedule_root)
    module = _load_protocol_module()
    monkeypatch.setattr(module, "verify_git_head_file", lambda _: None)
    matrix = module.load_contextual_matrix(matrix_path)
    path = schedule_root / "water_fold4_seed44.json"
    schedule = json.loads(path.read_text())
    schedule.pop("protocol_id")
    schedule["sets"].pop("10")
    path.write_text(json.dumps(schedule))
    index_path = schedule_root / "index.json"
    index = json.loads(index_path.read_text())
    for item in index["schedules"]:
        if item["path"] == path.name:
            item["sha256"] = _sha256(path)
    index_path.write_text(json.dumps(index, sort_keys=True))
    matrix["shot_schedule"]["index_sha256"] = _sha256(index_path)
    digest = hashlib.sha256()
    for item in sorted(schedule_root.glob("*.json")):
        if item.name == "index.json":
            continue
        digest.update(item.name.encode() + b"\0")
        digest.update(_sha256(item).encode() + b"\n")
    matrix["shot_schedule"]["ordered_filename_sha256_set_sha256"] = digest.hexdigest()

    with pytest.raises(ValueError, match="protocol"):
        module.verify_frozen_shot_schedules(matrix)
