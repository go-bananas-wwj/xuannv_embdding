from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import run_registered_paper_downstream as registered
from scripts.eval import run_traditional_ml_benchmark as benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_registered_scaling_subsets_are_nested_and_within_train_folds() -> None:
    split_path = ROOT / "configs/eval/haidian_spatial_5fold_buffer1_seed42.json"
    registry_path = ROOT / "configs/eval/haidian_paper_subsets_40_80_150_seed42.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert registry["source_split_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
    for fold in split["folds"]:
        subsets = registry["folds"][str(fold["fold"])]
        ids40, ids80, ids150 = map(set, (subsets["40"], subsets["80"], subsets["150"]))
        assert len(ids40) == 40
        assert len(ids80) == 80
        assert len(ids150) == 150
        assert ids40 < ids80 < ids150 <= set(fold["train"])


def test_few_shot_selection_requires_exact_positive_and_negative_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {f"p{i}": 64 for i in range(10)} | {f"n{i}": 0 for i in range(10)}
    task = benchmark.TaskSpec("task", [], Path("."))
    monkeypatch.setattr(
        benchmark,
        "positive_pixel_count",
        lambda _task, patch_id: counts[patch_id],
    )

    selected = benchmark.select_train_patch_ids(task, list(counts), "5", 42, 0)
    assert len(selected) == 10
    assert sum(counts[patch_id] >= 64 for patch_id in selected) == 5
    assert sum(counts[patch_id] == 0 for patch_id in selected) == 5

    with pytest.raises(RuntimeError, match=r"Exact 11\+11 shot budget infeasible"):
        benchmark.select_train_patch_ids(task, list(counts), "11", 42, 0)


def test_registered_threshold_uses_validation_probabilities_and_largest_tie() -> None:
    probabilities = registered.sigmoid_probabilities(np.array([-10.0, 0.0, 10.0]))
    assert np.allclose(probabilities, [4.5397868e-05, 0.5, 0.9999546], rtol=1e-5)

    threshold, score = registered.select_validation_threshold(
        probabilities=np.array([0.1, 0.4, 0.6, 0.9]),
        targets=np.array([0, 0, 1, 1]),
    )
    assert threshold == pytest.approx(0.6)
    assert score == pytest.approx(1.0)


def test_registered_provenance_rejects_mismatched_encoder_fold(tmp_path: Path) -> None:
    config_path = tmp_path / "encoder.yaml"
    config_path.write_text("data:\n  paper_fold: 1\n", encoding="utf-8")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="does not match requested evaluation fold"):
        registered.verify_encoder_provenance(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            expected_fold=0,
        )


def test_registered_provenance_requires_validation_selected_best_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "encoder.yaml"
    config_path.write_text("data:\n  paper_fold: 0\n", encoding="utf-8")
    checkpoint_path = tmp_path / "epoch_800.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="best.pt"):
        registered.verify_encoder_provenance(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            expected_fold=0,
        )


def test_registered_threshold_excludes_invalid_pixels_and_rejects_nonfinite_values() -> None:
    threshold, score = registered.select_validation_threshold(
        probabilities=np.array([0.1, 0.4, 0.6, 0.9]),
        targets=np.array([0, -1, 1, 1]),
    )
    assert threshold == pytest.approx(0.6)
    assert score == pytest.approx(1.0)

    with pytest.raises(ValueError, match="finite"):
        registered.select_validation_threshold(
            probabilities=np.array([0.1, np.nan]),
            targets=np.array([0, 1]),
        )


def test_registered_shot_manifest_is_nested_and_rejects_budget_shortage() -> None:
    counts = {f"p{i}": 64 for i in range(60)} | {f"n{i}": 0 for i in range(60)}
    manifest = registered.build_shot_manifest(
        task_name="building",
        train_ids=list(counts),
        fold=0,
        seed=42,
        label_sha256="labels",
        split_sha256="split",
        pixel_count=lambda patch_id: counts[patch_id],
    )
    assert set(manifest["sets"]["5"]["positive_patch_ids"]) < set(
        manifest["sets"]["10"]["positive_patch_ids"]
    )
    assert len(manifest["sets"]["10"]["negative_patch_ids"]) == 10

    limited_counts = {f"p{i}": 64 for i in range(12)} | {f"n{i}": 0 for i in range(12)}
    with pytest.raises(RuntimeError, match=r"Exact 50\+50 shot budget infeasible"):
        registered.build_shot_manifest(
            task_name="building",
            train_ids=list(limited_counts),
            fold=0,
            seed=42,
            label_sha256="labels",
            split_sha256="split",
            pixel_count=lambda patch_id: limited_counts[patch_id],
        )

    with pytest.raises(ValueError, match="unique"):
        registered.build_shot_manifest(
            task_name="building",
            train_ids=["p0", "p0", "n0"],
            fold=0,
            seed=42,
            label_sha256="labels",
            split_sha256="split",
            pixel_count=lambda patch_id: {"p0": 64, "n0": 0}[patch_id],
        )


def test_registered_shot_schedule_records_infeasible_budgets_without_truncating() -> None:
    counts = {f"p{i}": 64 for i in range(12)} | {f"n{i}": 0 for i in range(7)}
    schedule = registered.build_registered_shot_schedule(
        task_name="building",
        train_ids=list(counts),
        fold=0,
        seed=42,
        label_sha256="labels",
        split_sha256="split",
        pixel_count=lambda patch_id: counts[patch_id],
    )
    assert set(schedule["sets"]) == {"5"}
    assert set(schedule["sets"]["5"]["positive_patch_ids"]) <= set(counts)
    assert schedule["unavailable"]["10"]["status"] == "NA"
    assert schedule["unavailable"]["50"]["positive_available"] == 12
    assert schedule["unavailable"]["50"]["negative_available"] == 7
    assert len(schedule["rule_sha256"]) == 64


def test_registered_shot_schedule_is_nested_at_every_feasible_budget() -> None:
    counts = {f"p{i}": 64 for i in range(60)} | {f"n{i}": 0 for i in range(60)}
    schedule = registered.build_registered_shot_schedule(
        task_name="building",
        train_ids=list(counts),
        fold=0,
        seed=42,
        label_sha256="labels",
        split_sha256="split",
        pixel_count=lambda patch_id: counts[patch_id],
    )
    assert not schedule["unavailable"]
    for budget in (5, 10, 50):
        entry = schedule["sets"][str(budget)]
        assert len(entry["positive_patch_ids"]) == budget
        assert len(entry["negative_patch_ids"]) == budget
    assert (
        set(schedule["sets"]["5"]["positive_patch_ids"])
        <= set(schedule["sets"]["10"]["positive_patch_ids"])
        <= set(schedule["sets"]["50"]["positive_patch_ids"])
    )
    assert (
        set(schedule["sets"]["5"]["negative_patch_ids"])
        <= set(schedule["sets"]["10"]["negative_patch_ids"])
        <= set(schedule["sets"]["50"]["negative_patch_ids"])
    )
    assert (
        set(schedule["sets"]["5"]["train_patch_ids"])
        <= set(schedule["sets"]["10"]["train_patch_ids"])
        <= set(schedule["sets"]["50"]["train_patch_ids"])
    )

    reversed_schedule = registered.build_registered_shot_schedule(
        task_name="building",
        train_ids=list(reversed(list(counts))),
        fold=0,
        seed=42,
        label_sha256="labels",
        split_sha256="split",
        pixel_count=lambda patch_id: counts[patch_id],
    )
    assert schedule == reversed_schedule

    with pytest.raises(ValueError, match="positive integers"):
        registered.build_registered_shot_schedule(
            task_name="building",
            train_ids=list(counts),
            fold=0,
            seed=42,
            label_sha256="labels",
            split_sha256="split",
            pixel_count=lambda patch_id: counts[patch_id],
            budgets=(True,),
        )


def test_registered_embedding_index_rejects_tampered_feature_file(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    feature = root / "haidian" / "patch_000001" / "202604_embedding_map.pt"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"original feature tensor")

    index = registered.build_embedding_file_index(root, "haidian", "202604")
    registered.verify_embedding_file_index(root, index, "haidian", "202604")

    feature.write_bytes(b"modified feature tensor")
    with pytest.raises(ValueError, match="hash mismatch"):
        registered.verify_embedding_file_index(root, index, "haidian", "202604")

    with pytest.raises(ValueError, match="patch IDs"):
        registered.verify_embedding_file_index(
            root,
            registered.build_embedding_file_index(root, "haidian", "202604"),
            "haidian",
            "202604",
            expected_patch_ids={"patch_000001", "patch_000002"},
        )


def test_registered_result_directory_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "result"
    registered.prepare_result_output(output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        registered.prepare_result_output(output)


def test_registered_embedding_index_requires_meta_seal(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    feature = root / "haidian" / "patch_000001" / "202604_embedding_map.pt"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"feature tensor")
    index_path = root / "embedding_file_index.json"
    index_path.write_text(
        json.dumps(registered.build_embedding_file_index(root, "haidian", "202604")),
        encoding="utf-8",
    )
    meta = {
        "embedding_file_index_sha256": registered.sha256_file(index_path),
        "embedding_file_index": {
            "path": "embedding_file_index.json",
            "region": "haidian",
            "month": "202604",
            "file_count": 1,
        },
    }
    registered.load_sealed_embedding_file_index(root, meta, "haidian", "202604")

    index_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match export metadata"):
        registered.load_sealed_embedding_file_index(root, meta, "haidian", "202604")


def test_registered_embedding_index_rejects_inconsistent_meta_description(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    feature = root / "haidian" / "patch_000001" / "202604_embedding_map.pt"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"feature tensor")
    index_path = root / "embedding_file_index.json"
    index_path.write_text(
        json.dumps(registered.build_embedding_file_index(root, "haidian", "202604")),
        encoding="utf-8",
    )
    meta = {
        "embedding_file_index_sha256": registered.sha256_file(index_path),
        "embedding_file_index": {
            "path": "embedding_file_index.json",
            "region": "haidian",
            "month": "202604",
            "file_count": 2,
        },
    }
    with pytest.raises(ValueError, match="metadata file_count"):
        registered.load_sealed_embedding_file_index(root, meta, "haidian", "202604")


def test_registered_embedding_index_sealing_writes_meta_binding(tmp_path: Path) -> None:
    root = tmp_path / "embeddings"
    feature = root / "haidian" / "patch_000001" / "202604_embedding_map.pt"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"feature tensor")
    (root / "meta.json").write_text("{}", encoding="utf-8")

    index = registered.seal_embedding_file_index(root, "haidian", "202604")
    stored_meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert stored_meta["embedding_file_index_sha256"] == registered.sha256_file(
        root / "embedding_file_index.json"
    )
    assert index["files"][0]["path"] == "haidian/patch_000001/202604_embedding_map.pt"


def test_registered_embedding_registry_requires_external_index_hash(tmp_path: Path) -> None:
    registry_path = tmp_path / "embedding_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exports": [
                    {
                        "checkpoint_sha256": "checkpoint",
                        "manifest_sha256": "manifest",
                        "embedding_file_index_sha256": "index",
                        "region": "haidian",
                        "month": "202604",
                        "patch_count": 320,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registered.verify_embedding_registry(
        registry_path,
        checkpoint_sha256="checkpoint",
        manifest_sha256="manifest",
        index_sha256="index",
        region="haidian",
        month="202604",
        patch_count=320,
    )

    with pytest.raises(ValueError, match="does not contain the sealed export"):
        registered.verify_embedding_registry(
            registry_path,
            checkpoint_sha256="checkpoint",
            manifest_sha256="manifest",
            index_sha256="tampered",
            region="haidian",
            month="202604",
            patch_count=320,
        )


def test_registered_external_registry_must_be_a_clean_head_tracked_file(tmp_path: Path) -> None:
    tracked = ROOT / "configs/eval/haidian_spatial_5fold_buffer1_seed42.json"
    registered.verify_git_head_file(tracked)
    with pytest.raises(ValueError, match="inside the repository"):
        registered.verify_git_head_file(tmp_path / "untracked.json")


def test_registered_result_registry_is_idempotent_and_rejects_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.jsonl"
    record = {"result_id": "same", "artifact_sha256": "artifact"}
    fsync_calls: list[int] = []

    class OsShim:
        @staticmethod
        def fsync(file_descriptor: int) -> None:
            fsync_calls.append(file_descriptor)

    monkeypatch.setattr(registered, "os", OsShim, raising=False)
    registered._append_registry(path, record)
    assert (path.parent / ".results.jsonl.lock").is_file()
    assert fsync_calls
    registered._append_registry(path, record)
    assert path.read_text(encoding="utf-8").count("\n") == 1

    with pytest.raises(ValueError, match="collision"):
        registered._append_registry(path, {"result_id": "same", "artifact_sha256": "other"})
