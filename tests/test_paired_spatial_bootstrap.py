from __future__ import annotations

import pytest

from scripts.report import paired_spatial_bootstrap as bootstrap


def _confusion(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, int]:
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def test_confusion_metrics_are_calculated_from_aggregated_counts() -> None:
    metrics = bootstrap.metrics_from_confusion(_confusion(tp=8, fp=2, fn=2, tn=8))

    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["miou"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(0.8)


def test_paired_bootstrap_requires_identical_patch_support() -> None:
    with pytest.raises(ValueError, match="identical patch IDs"):
        bootstrap.bootstrap_group_difference(
            {"patch_a": _confusion(5, 1, 1)},
            {"patch_b": _confusion(6, 1, 1)},
            n_resamples=10,
            seed=1,
        )


@pytest.mark.parametrize("invalid", [True, -1, 0.5])
def test_paired_bootstrap_rejects_non_count_confusion_values(invalid: object) -> None:
    baseline = {"patch_a": {"tp": invalid, "fp": 0, "fn": 0, "tn": 1}}
    candidate = {"patch_a": _confusion(1, 0, 0, 1)}

    with pytest.raises(ValueError, match="non-negative integer tp"):
        bootstrap.bootstrap_group_difference(baseline, candidate, n_resamples=10, seed=1)


def test_paired_bootstrap_preserves_direction_for_uniform_improvement() -> None:
    baseline = {
        "patch_a": _confusion(4, 4, 4),
        "patch_b": _confusion(3, 3, 3),
    }
    candidate = {
        "patch_a": _confusion(8, 1, 1),
        "patch_b": _confusion(6, 1, 1),
    }

    summary = bootstrap.bootstrap_group_difference(baseline, candidate, n_resamples=200, seed=7)

    for metric in ("f1", "miou", "precision", "recall"):
        difference = summary["candidate_minus_baseline"][metric]
        assert difference["point_difference"] > 0
        assert difference["ci95_low"] > 0
        assert difference["probability_candidate_better"] == pytest.approx(1.0)


def test_hierarchical_bootstrap_rejects_mismatched_group_keys() -> None:
    baseline = {
        (0, 42): {"patch_a": _confusion(4, 4, 4)},
    }
    candidate = {
        (1, 42): {"patch_a": _confusion(8, 1, 1)},
    }

    with pytest.raises(ValueError, match="identical evaluation groups"):
        bootstrap.hierarchical_paired_bootstrap(baseline, candidate, n_resamples=10, seed=1)


def test_hierarchical_bootstrap_resamples_spatial_folds_with_nested_seeds() -> None:
    baseline = {
        (0, 42): {"patch_a": _confusion(4, 4, 4)},
        (0, 43): {"patch_a": _confusion(4, 4, 4)},
        (0, 44): {"patch_a": _confusion(4, 4, 4)},
        (1, 42): {"patch_a": _confusion(4, 4, 4)},
        (1, 43): {"patch_a": _confusion(4, 4, 4)},
        (1, 44): {"patch_a": _confusion(4, 4, 4)},
    }
    candidate = {group: {"patch_a": _confusion(8, 1, 1)} for group in baseline}

    summary = bootstrap.hierarchical_paired_bootstrap(baseline, candidate, n_resamples=100, seed=1)

    assert summary["n_spatial_folds"] == 2
    assert summary["probe_seeds_per_fold"] == [42, 43, 44]
    assert summary["candidate_minus_baseline"]["f1"]["ci95_low"] > 0


def test_hierarchical_bootstrap_rejects_inconsistent_seed_support() -> None:
    baseline = {
        (0, 42): {"patch_a": _confusion(4, 4, 4)},
        (0, 43): {"patch_a": _confusion(4, 4, 4)},
        (1, 42): {"patch_a": _confusion(4, 4, 4)},
    }
    candidate = {group: {"patch_a": _confusion(8, 1, 1)} for group in baseline}

    with pytest.raises(ValueError, match="same non-empty probe-seed set"):
        bootstrap.hierarchical_paired_bootstrap(baseline, candidate, n_resamples=10, seed=1)


def test_result_identities_are_sorted_and_fail_closed() -> None:
    records = [
        {
            "result_id": "result-b",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
        },
        {
            "result_id": "result-a",
            "artifact_sha256": "artifact-a",
            "registry_entry_sha256": "entry-a",
        },
    ]

    assert bootstrap.result_identities(records) == [
        {
            "result_id": "result-a",
            "artifact_sha256": "artifact-a",
            "registry_entry_sha256": "entry-a",
        },
        {
            "result_id": "result-b",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
        },
    ]

    with pytest.raises(ValueError, match="incomplete identity"):
        bootstrap.result_identities(
            [
                {
                    "result_id": "result-a",
                    "artifact_sha256": "",
                    "registry_entry_sha256": "entry-a",
                }
            ]
        )
