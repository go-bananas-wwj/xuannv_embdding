from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report import paired_spatial_bootstrap as bootstrap


def _confusion(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, int]:
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _complete_block(tp: int, fp: int, fn: int) -> dict[str, dict[str, int]]:
    return {f"patch_{index}": _confusion(tp, fp, fn) for index in range(4)}


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
        (0, 42): _complete_block(4, 4, 4),
    }
    candidate = {
        (1, 42): _complete_block(8, 1, 1),
    }

    with pytest.raises(ValueError, match="identical evaluation groups"):
        bootstrap._hierarchical_paired_resample(
            baseline,
            candidate,
            n_resamples=10,
            seed=1,
            clusters_by_fold={0: [("patch_0", "patch_1", "patch_2", "patch_3")]},
        )


def test_hierarchical_bootstrap_resamples_spatial_folds_with_nested_seeds() -> None:
    baseline = {
        (0, 42): _complete_block(4, 4, 4),
        (0, 43): _complete_block(4, 4, 4),
        (0, 44): _complete_block(4, 4, 4),
        (1, 42): _complete_block(4, 4, 4),
        (1, 43): _complete_block(4, 4, 4),
        (1, 44): _complete_block(4, 4, 4),
    }
    candidate = {group: _complete_block(8, 1, 1) for group in baseline}

    summary = bootstrap._hierarchical_paired_resample(
        baseline,
        candidate,
        n_resamples=100,
        seed=1,
        clusters_by_fold={
            0: [("patch_0", "patch_1", "patch_2", "patch_3")],
            1: [("patch_0", "patch_1", "patch_2", "patch_3")],
        },
    )

    assert summary["n_spatial_folds"] == 2
    assert summary["support_schedule_seeds_per_fold"] == [42, 43, 44]
    assert summary["candidate_minus_baseline"]["f1"]["ci95_low"] > 0


def test_hierarchical_bootstrap_exposes_non_degenerate_seed_uncertainty() -> None:
    patch_ids = tuple(f"patch_{index}" for index in range(8))
    clusters = {0: [patch_ids[:4], patch_ids[4:]], 1: [patch_ids[:4], patch_ids[4:]]}
    baseline = {
        (fold, probe_seed): {patch_id: _confusion(4, 4, 4) for patch_id in patch_ids}
        for fold in (0, 1)
        for probe_seed in (42, 43, 44)
    }
    candidate = {}
    seed_bonus = {42: 4, 43: 2, 44: 1}
    for fold in (0, 1):
        for probe_seed in (42, 43, 44):
            candidate[(fold, probe_seed)] = {
                patch_id: _confusion(
                    4
                    + seed_bonus[probe_seed]
                    + (2 if fold == 0 else 0)
                    + (1 if patch_id in patch_ids[:4] else 0),
                    1,
                    1,
                )
                for patch_id in patch_ids
            }

    summary = bootstrap._hierarchical_paired_resample(
        baseline, candidate, n_resamples=200, seed=7, clusters_by_fold=clusters
    )

    f1 = summary["candidate_minus_baseline"]["f1"]
    descriptive = summary["descriptive_candidate_minus_baseline_standard_deviation"]["f1"]
    assert summary["resampling_hierarchy"] == [
        "fold",
        "complete_2x2_geographic_cluster",
        "support_schedule_seed",
    ]
    assert descriptive["across_spatial_folds"] > 0.0
    assert descriptive["across_support_schedule_seeds"] > 0.0
    assert f1["ci95_low"] < f1["ci95_high"]


def test_internal_resampler_exposes_cluster_only_uncertainty() -> None:
    patch_ids = tuple(f"patch_{index}" for index in range(8))
    clusters = {0: [patch_ids[:4], patch_ids[4:]], 1: [patch_ids[:4], patch_ids[4:]]}
    baseline = {
        (fold, probe_seed): {patch_id: _confusion(4, 4, 4) for patch_id in patch_ids}
        for fold in (0, 1)
        for probe_seed in (42, 43, 44)
    }
    candidate = {
        (fold, probe_seed): {
            patch_id: _confusion(10 if patch_id in patch_ids[:4] else 5, 1, 1)
            for patch_id in patch_ids
        }
        for fold in (0, 1)
        for probe_seed in (42, 43, 44)
    }

    summary = bootstrap._hierarchical_paired_resample(
        baseline, candidate, n_resamples=200, seed=9, clusters_by_fold=clusters
    )

    f1 = summary["candidate_minus_baseline"]["f1"]
    descriptive = summary["descriptive_candidate_minus_baseline_standard_deviation"]["f1"]
    assert descriptive == {"across_spatial_folds": 0.0, "across_support_schedule_seeds": 0.0}
    assert f1["ci95_low"] < f1["ci95_high"]


def test_internal_resampler_exposes_fold_only_uncertainty() -> None:
    patch_ids = tuple(f"patch_{index}" for index in range(4))
    clusters = {fold: [patch_ids] for fold in range(5)}
    baseline = {
        (fold, probe_seed): {patch_id: _confusion(4, 4, 4) for patch_id in patch_ids}
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }
    candidate = {
        (fold, probe_seed): {patch_id: _confusion(5 + fold, 1, 1) for patch_id in patch_ids}
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }

    summary = bootstrap._hierarchical_paired_resample(
        baseline, candidate, n_resamples=200, seed=11, clusters_by_fold=clusters
    )

    f1 = summary["candidate_minus_baseline"]["f1"]
    descriptive = summary["descriptive_candidate_minus_baseline_standard_deviation"]["f1"]
    assert descriptive["across_spatial_folds"] > 0.0
    assert descriptive["across_support_schedule_seeds"] == 0.0
    assert f1["ci95_low"] < f1["ci95_high"]


def test_internal_resampler_exposes_seed_only_uncertainty() -> None:
    patch_ids = tuple(f"patch_{index}" for index in range(4))
    clusters = {fold: [patch_ids] for fold in range(5)}
    baseline = {
        (fold, probe_seed): {patch_id: _confusion(4, 4, 4) for patch_id in patch_ids}
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }
    candidate = {
        (fold, probe_seed): {
            patch_id: _confusion({42: 8, 43: 6, 44: 5}[probe_seed], 1, 1) for patch_id in patch_ids
        }
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }

    summary = bootstrap._hierarchical_paired_resample(
        baseline, candidate, n_resamples=200, seed=13, clusters_by_fold=clusters
    )

    f1 = summary["candidate_minus_baseline"]["f1"]
    descriptive = summary["descriptive_candidate_minus_baseline_standard_deviation"]["f1"]
    assert descriptive["across_spatial_folds"] == 0.0
    assert descriptive["across_support_schedule_seeds"] > 0.0
    assert f1["ci95_low"] < f1["ci95_high"]


def test_public_bootstrap_requires_registered_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap, "PROTOCOL_N_RESAMPLES", 1)
    baseline = {
        (fold, probe_seed): _complete_block(4, 4, 4)
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }
    candidate = {
        (fold, probe_seed): _complete_block(8, 1, 1)
        for fold in range(5)
        for probe_seed in (42, 43, 44)
    }
    bounds = {
        "patch_0": (0.0, 0.0, 1.0, 1.0),
        "patch_1": (1.0, 0.0, 2.0, 1.0),
        "patch_2": (0.0, 1.0, 1.0, 2.0),
        "patch_3": (1.0, 1.0, 2.0, 2.0),
    }

    summary = bootstrap.hierarchical_paired_bootstrap(
        baseline, candidate, n_resamples=1, seed=1, bounds_by_patch=bounds
    )

    assert summary["n_spatial_folds"] == 5
    with pytest.raises(ValueError, match="exactly"):
        bootstrap.hierarchical_paired_bootstrap(
            baseline, candidate, n_resamples=2, seed=1, bounds_by_patch=bounds
        )


def test_hierarchical_bootstrap_rejects_inconsistent_seed_support() -> None:
    baseline = {
        (0, 42): _complete_block(4, 4, 4),
        (0, 43): _complete_block(4, 4, 4),
        (1, 42): _complete_block(4, 4, 4),
    }
    candidate = {group: _complete_block(8, 1, 1) for group in baseline}

    with pytest.raises(ValueError, match="same non-empty probe-seed set"):
        bootstrap._hierarchical_paired_resample(
            baseline,
            candidate,
            n_resamples=10,
            seed=1,
            clusters_by_fold={
                0: [("patch_0", "patch_1", "patch_2", "patch_3")],
                1: [("patch_0", "patch_1", "patch_2", "patch_3")],
            },
        )


def test_geographic_clusters_group_neighboring_patches_into_two_by_two_blocks() -> None:
    bounds = {
        "patch_a": (0.0, 0.0, 1.0, 1.0),
        "patch_b": (1.0, 0.0, 2.0, 1.0),
        "patch_c": (0.0, 1.0, 1.0, 2.0),
        "patch_d": (1.0, 1.0, 2.0, 2.0),
        "patch_e": (2.0, 0.0, 3.0, 1.0),
    }

    with pytest.raises(ValueError, match="complete 2 x 2"):
        bootstrap.build_two_by_two_geographic_clusters(bounds, sorted(bounds))


def test_geographic_clusters_keep_a_global_lattice_anchor_for_fold_subsets() -> None:
    bounds = {
        "patch_a": (0.0, 0.0, 1.0, 1.0),
        "patch_b": (1.0, 0.0, 2.0, 1.0),
        "patch_c": (0.0, 1.0, 1.0, 2.0),
        "patch_d": (1.0, 1.0, 2.0, 2.0),
        "patch_e": (2.0, 0.0, 3.0, 1.0),
        "patch_f": (3.0, 0.0, 4.0, 1.0),
        "patch_g": (2.0, 1.0, 3.0, 2.0),
        "patch_h": (3.0, 1.0, 4.0, 2.0),
    }

    with pytest.raises(ValueError, match="complete 2 x 2"):
        bootstrap.build_two_by_two_geographic_clusters(
            bounds, ["patch_b", "patch_e", "patch_d", "patch_g"]
        )


def test_load_patch_bounds_reads_project_patch_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "patches.json"
    metadata.write_text(
        json.dumps(
            [
                {"patch_id": "patch_a", "bounds": [0.0, 0.0, 1.0, 1.0]},
                {"patch_id": "patch_b", "bounds": [1.0, 0.0, 2.0, 1.0]},
            ]
        ),
        encoding="utf-8",
    )

    assert bootstrap.load_patch_bounds(metadata) == {
        "patch_a": (0.0, 0.0, 1.0, 1.0),
        "patch_b": (1.0, 0.0, 2.0, 1.0),
    }


def test_geographic_clusters_reject_misaligned_or_overlapping_patch_bounds() -> None:
    with pytest.raises(ValueError, match="regular patch lattice"):
        bootstrap.build_two_by_two_geographic_clusters(
            {
                "patch_a": (0.0, 0.0, 1.0, 1.0),
                "patch_b": (1.25, 0.0, 2.25, 1.0),
            },
            ["patch_a", "patch_b"],
        )
    with pytest.raises(ValueError, match="must not overlap"):
        bootstrap.build_two_by_two_geographic_clusters(
            {
                "patch_a": (0.0, 0.0, 1.0, 1.0),
                "patch_b": (0.0, 0.0, 1.0, 1.0),
            },
            ["patch_a", "patch_b"],
        )


def test_hierarchical_bootstrap_rejects_clusters_that_do_not_partition_patch_support() -> None:
    baseline = {
        (0, 42): _complete_block(4, 4, 4),
        (0, 43): _complete_block(4, 4, 4),
        (0, 44): _complete_block(4, 4, 4),
    }
    candidate = {group: values for group, values in baseline.items()}

    with pytest.raises(ValueError, match="exactly four"):
        bootstrap._hierarchical_paired_resample(
            baseline,
            candidate,
            n_resamples=10,
            seed=1,
            clusters_by_fold={0: [("patch_0", "patch_1", "patch_2")]},
        )


def test_result_identities_are_sorted_and_fail_closed() -> None:
    records = [
        {
            "result_id": "result-b",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
            "metrics_sha256": "metrics-b",
        },
        {
            "result_id": "result-a",
            "artifact_sha256": "artifact-a",
            "registry_entry_sha256": "entry-a",
            "metrics_sha256": "metrics-a",
        },
    ]

    assert bootstrap.result_identities(records) == [
        {
            "result_id": "result-a",
            "artifact_sha256": "artifact-a",
            "registry_entry_sha256": "entry-a",
            "metrics_sha256": "metrics-a",
        },
        {
            "result_id": "result-b",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
            "metrics_sha256": "metrics-b",
        },
    ]

    with pytest.raises(ValueError, match="complete sealed identity"):
        bootstrap.result_identities(
            [
                {
                    "result_id": "result-a",
                    "artifact_sha256": "",
                    "registry_entry_sha256": "entry-a",
                    "metrics_sha256": "metrics-a",
                }
            ]
        )


def test_input_snapshot_has_a_stable_identity_hash() -> None:
    baseline = [
        {
            "result_id": "base-a",
            "artifact_sha256": "artifact-a",
            "registry_entry_sha256": "entry-a",
            "metrics_sha256": "metrics-a",
        }
    ]
    candidate = [
        {
            "result_id": "candidate-a",
            "artifact_sha256": "artifact-b",
            "registry_entry_sha256": "entry-b",
            "metrics_sha256": "metrics-b",
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
            n_resamples=bootstrap.PROTOCOL_N_RESAMPLES,
            seed=1,
            patch_metadata_path=tmp_path / "unused.json",
        )


def test_compare_rejects_non_protocol_resample_count_before_loading_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        bootstrap.aggregate,
        "load_verified_records",
        lambda *_args, **_kwargs: pytest.fail(
            "non-protocol resample count must fail before record loading"
        ),
    )

    with pytest.raises(ValueError, match="exactly"):
        bootstrap.compare_families(
            registry_path=tmp_path / "results.jsonl",
            baseline_family="full_40",
            candidate_family="full_150",
            tasks=("building",),
            shots=("5",),
            allow_preliminary=True,
            n_resamples=1,
            seed=1,
            patch_metadata_path=tmp_path / "unused.json",
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
                            "metrics_sha256": f"metrics-{result_id}",
                            "metric_provenance": {
                                "task": task,
                                "shot": shot,
                                "fold": fold,
                                "shot_seed": seed,
                                "spatial_split_sha256": "split-sha",
                                "manifest_sha256": "manifest-sha",
                                "shot_manifest_sha256": f"shot-{task}-{fold}-{seed}",
                                "shot_manifest": {
                                    "label_sha256": "label-sha",
                                    "split_sha256": "split-sha",
                                },
                                "per_patch_confusion": {
                                    "patch-a": _confusion(4, 1, 1),
                                    "patch-b": _confusion(3, 1, 1),
                                    "patch-c": _confusion(5, 1, 1),
                                    "patch-d": _confusion(2, 1, 1),
                                },
                            },
                        }
                    )
                    records[-1]["label_sha256"] = "label-sha"
    if include_unrelated:
        records.append(
            {
                "result_id": f"{prefix}-unrelated",
                "artifact_sha256": f"artifact-{prefix}-unrelated",
                "registry_entry_sha256": f"entry-{prefix}-unrelated",
                "metrics_sha256": f"metrics-{prefix}-unrelated",
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
    metadata_path = tmp_path / "patches.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"patch_id": "patch-a", "bounds": [0.0, 0.0, 1.0, 1.0]},
                {"patch_id": "patch-b", "bounds": [1.0, 0.0, 2.0, 1.0]},
                {"patch_id": "patch-c", "bounds": [0.0, 1.0, 1.0, 2.0]},
                {"patch_id": "patch-d", "bounds": [1.0, 1.0, 2.0, 2.0]},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap, "PROTOCOL_N_RESAMPLES", 1)
    report = bootstrap.compare_families(
        registry_path=tmp_path / "results.jsonl",
        baseline_family="full_40",
        candidate_family="full_150",
        tasks=("building", "road", "water"),
        shots=("5", "10"),
        allow_preliminary=True,
        n_resamples=1,
        seed=1,
        patch_metadata_path=metadata_path,
    )

    snapshot = report["input_identity_snapshot"]
    assert snapshot["baseline_result_count"] == 90
    assert snapshot["candidate_result_count"] == 90
    assert all("unrelated" not in item["result_id"] for item in snapshot["baseline_input_results"])
    assert all("unrelated" not in item["result_id"] for item in snapshot["candidate_input_results"])
    assert report["preliminary"] is True
    assert report["paper_eligible"] is False
    assert report["comparisons"]["building|5"]["geographic_cluster_membership_by_fold"]
    assert report["comparisons"]["building|5"][
        "descriptive_candidate_minus_baseline_standard_deviation"
    ]["f1"] == {"across_spatial_folds": 0.0, "across_support_schedule_seeds": 0.0}


@pytest.mark.parametrize(
    "mutation",
    (
        "label",
        "spatial_split",
        "manifest",
        "shot_manifest",
        "shot_manifest_label",
        "shot_manifest_split",
        "test_patch_support",
    ),
)
def test_compare_rejects_mismatched_pairing_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    baseline = _matrix_records("baseline", include_unrelated=False)
    candidate = _matrix_records("candidate", include_unrelated=False)
    payload = candidate[0]["metric_provenance"]
    assert isinstance(payload, dict)
    if mutation == "label":
        candidate[0]["label_sha256"] = "different-label-sha"
    elif mutation == "spatial_split":
        payload["spatial_split_sha256"] = "different-split-sha"
    elif mutation == "manifest":
        payload["manifest_sha256"] = "different-manifest-sha"
    elif mutation == "shot_manifest":
        payload["shot_manifest_sha256"] = "different-shot-sha"
    elif mutation == "shot_manifest_label":
        payload["shot_manifest"]["label_sha256"] = "different-label-sha"
    elif mutation == "shot_manifest_split":
        payload["shot_manifest"]["split_sha256"] = "different-split-sha"
    else:
        payload["per_patch_confusion"].pop("patch-d")
    monkeypatch.setattr(
        bootstrap.aggregate,
        "load_verified_records",
        lambda _path, *, family, **_kwargs: baseline if family == "full_40" else candidate,
    )
    metadata_path = tmp_path / "patches.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"patch_id": "patch-a", "bounds": [0.0, 0.0, 1.0, 1.0]},
                {"patch_id": "patch-b", "bounds": [1.0, 0.0, 2.0, 1.0]},
                {"patch_id": "patch-c", "bounds": [0.0, 1.0, 1.0, 2.0]},
                {"patch_id": "patch-d", "bounds": [1.0, 1.0, 2.0, 2.0]},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap, "PROTOCOL_N_RESAMPLES", 1)

    with pytest.raises(ValueError, match="identical|test patch IDs"):
        bootstrap.compare_families(
            registry_path=tmp_path / "results.jsonl",
            baseline_family="full_40",
            candidate_family="full_150",
            tasks=("building", "road", "water"),
            shots=("5", "10"),
            allow_preliminary=True,
            n_resamples=1,
            seed=1,
            patch_metadata_path=metadata_path,
        )


def test_main_writes_a_sealed_selected_input_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _matrix_records("baseline", include_unrelated=True)
    candidate = _matrix_records("candidate", include_unrelated=True)

    def load_records(_path: object, *, family: str, **_kwargs: object) -> list[dict[str, object]]:
        return baseline if family == "full_40" else candidate

    output_path = tmp_path / "bootstrap.json"
    metadata_path = tmp_path / "patches.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"patch_id": "patch-a", "bounds": [0.0, 0.0, 1.0, 1.0]},
                {"patch_id": "patch-b", "bounds": [1.0, 0.0, 2.0, 1.0]},
                {"patch_id": "patch-c", "bounds": [0.0, 1.0, 1.0, 2.0]},
                {"patch_id": "patch-d", "bounds": [1.0, 1.0, 2.0, 2.0]},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bootstrap.aggregate, "load_verified_records", load_records)
    monkeypatch.setattr(bootstrap, "PROTOCOL_N_RESAMPLES", 1)
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
            "--patch-metadata",
            str(metadata_path),
            "--n-resamples",
            "1",
            "--allow-preliminary",
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


def test_main_refuses_to_overwrite_a_bootstrap_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bootstrap output and its identity snapshot are write-once evidence artifacts."""
    output_path = tmp_path / "bootstrap.json"
    output_path.write_text('{"existing":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        bootstrap,
        "compare_families",
        lambda **_kwargs: {
            "input_identity_snapshot": {
                "sha256": "identity",
                "baseline_result_count": 90,
                "candidate_result_count": 90,
            }
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "paired_spatial_bootstrap.py",
            "--registry",
            str(tmp_path / "results.jsonl"),
            "--baseline-family",
            "aef_v5",
            "--candidate-family",
            "full_150",
            "--output",
            str(output_path),
            "--patch-metadata",
            str(tmp_path / "patches.json"),
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bootstrap.main()
