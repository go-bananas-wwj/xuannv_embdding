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
