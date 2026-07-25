from __future__ import annotations

import json
from pathlib import Path

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


def test_input_snapshot_has_a_stable_identity_hash() -> None:
    baseline = [
        {"result_id": "base-a", "artifact_sha256": "artifact-a", "registry_entry_sha256": "entry-a"}
    ]
    candidate = [
        {
            "result_id": "candidate-a",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
        }
    ]

    snapshot = bootstrap.build_input_identity_snapshot(baseline, candidate)

    assert snapshot["schema_version"] == 1
    assert snapshot["baseline_result_count"] == 1
    assert snapshot["candidate_result_count"] == 1
    assert len(snapshot["sha256"]) == 64


def test_matrix_selection_excludes_unrequested_task_or_shot() -> None:
    records = [
        {"metric_provenance": {"task": "building", "shot": "5"}},
        {"metric_provenance": {"task": "building", "shot": "10"}},
        {"metric_provenance": {"task": "road", "shot": "5"}},
    ]

    selected = bootstrap.select_records_for_matrix(records, tasks=("building",), shots=("5",))

    assert selected == [records[0]]


def test_derived_bootstrap_admission_is_never_automatic() -> None:
    admission = bootstrap.derived_bootstrap_admission()

    assert admission == {
        "preliminary": True,
        "paper_eligible": False,
        "admission_status": "derived_statistic_pending_external_admission",
    }


def test_compare_rejects_duplicate_matrix_dimensions_before_loading_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        bootstrap.aggregate,
        "load_verified_records",
        lambda *_args, **_kwargs: pytest.fail(
            "duplicate dimensions must fail before record loading"
        ),
    )

    with pytest.raises(ValueError, match="unique"):
        bootstrap.compare_families(
            registry_path=tmp_path / "results.jsonl",
            baseline_family="full_40",
            candidate_family="full_150",
            tasks=("building", "building"),
            shots=("5",),
            allow_preliminary=True,
            n_resamples=1,
            seed=1,
        )


def _matrix_records(prefix: str, *, include_unrelated: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in ("building", "road", "water"):
        for shot in ("5", "10"):
            for fold in range(5):
                for seed in (42, 43, 44):
                    result_id = f"{prefix}-{task}-{shot}-{fold}-{seed}"
                    records.append(
                        {
                            "result_id": result_id,
                            "artifact_sha256": f"artifact-{result_id}",
                            "registry_entry_sha256": f"entry-{result_id}",
                            "metric_provenance": {
                                "task": task,
                                "shot": shot,
                                "fold": fold,
                                "shot_seed": seed,
                                "per_patch_confusion": {
                                    "patch-a": _confusion(4, 1, 1),
                                    "patch-b": _confusion(3, 1, 1),
                                },
                            },
                        }
                    )
    if include_unrelated:
        records.append(
            {
                "result_id": f"{prefix}-unrelated",
                "artifact_sha256": f"artifact-{prefix}-unrelated",
                "registry_entry_sha256": f"entry-{prefix}-unrelated",
                "metric_provenance": {
                    "task": "unrelated",
                    "shot": "50",
                    "fold": 0,
                    "shot_seed": 42,
                    "per_patch_confusion": {"patch-a": _confusion(1, 0, 0)},
                },
            }
        )
    return records


def test_compare_snapshot_excludes_unrelated_family_records_and_stays_preliminary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _matrix_records("baseline", include_unrelated=True)
    candidate = _matrix_records("candidate", include_unrelated=True)

    def load_records(_path: object, *, family: str, **_kwargs: object) -> list[dict[str, object]]:
        return baseline if family == "full_40" else candidate

    monkeypatch.setattr(bootstrap.aggregate, "load_verified_records", load_records)
    report = bootstrap.compare_families(
        registry_path=tmp_path / "results.jsonl",
        baseline_family="full_40",
        candidate_family="full_150",
        tasks=("building", "road", "water"),
        shots=("5", "10"),
        allow_preliminary=False,
        n_resamples=1,
        seed=1,
    )

    snapshot = report["input_identity_snapshot"]
    assert snapshot["baseline_result_count"] == 90
    assert snapshot["candidate_result_count"] == 90
    assert all("unrelated" not in item["result_id"] for item in snapshot["baseline_input_results"])
    assert all("unrelated" not in item["result_id"] for item in snapshot["candidate_input_results"])
    assert report["preliminary"] is True
    assert report["paper_eligible"] is False


def test_main_writes_a_sealed_selected_input_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _matrix_records("baseline", include_unrelated=True)
    candidate = _matrix_records("candidate", include_unrelated=True)

    def load_records(_path: object, *, family: str, **_kwargs: object) -> list[dict[str, object]]:
        return baseline if family == "full_40" else candidate

    output_path = tmp_path / "bootstrap.json"
    monkeypatch.setattr(bootstrap.aggregate, "load_verified_records", load_records)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paired_spatial_bootstrap.py",
            "--registry",
            str(tmp_path / "results.jsonl"),
            "--baseline-family",
            "full_40",
            "--candidate-family",
            "full_150",
            "--output",
            str(output_path),
            "--n-resamples",
            "1",
        ],
    )

    bootstrap.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    snapshot_path = Path(report["input_identity_snapshot"]["path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_path.name == "bootstrap_input_identity_snapshot.json"
    assert report["input_identity_snapshot"][
        "sha256"
    ] == bootstrap.aggregate.registered.sha256_file(snapshot_path)
    assert snapshot["baseline_result_count"] == 90
    assert snapshot["candidate_result_count"] == 90
    assert all("unrelated" not in item["result_id"] for item in snapshot["baseline_input_results"])
    assert all("unrelated" not in item["result_id"] for item in snapshot["candidate_input_results"])
    assert report["preliminary"] is True
    assert report["paper_eligible"] is False
