from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval import run_registered_paper_downstream as registered
from scripts.report import aggregate_registered_paper_results as aggregate


def _write_record(
    root: Path,
    registry_path: Path,
    *,
    fold: int,
    task: str,
    shot: str,
    seed: int,
    f1: float,
    paper_eligible: bool,
) -> None:
    output = root / f"full150_fold{fold}" / task / f"shot_{shot}" / f"seed_{seed}"
    output.mkdir(parents=True)
    config_path = (
        root
        / "configs"
        / "paper_registered_20260716"
        / f"paper_registered_full_150_fold{fold}_20260716.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"data:\n  paper_fold: {fold}\n", encoding="utf-8")
    predictions = output / "predictions_test.npz"
    predictions.write_bytes(b"predictions")
    (output / "final_probe.pt").write_bytes(b"probe")
    metric_payload = {
        "paper_eligible": paper_eligible,
        "admission_status": (
            "registered_paper_eligible"
            if paper_eligible
            else "registered_preliminary_pending_external_gates"
        ),
        "task": task,
        "fold": fold,
        "shot": shot,
        "shot_seed": seed,
        "f1_at_threshold": f1,
        "ap": f1 + 0.1,
        "auc_roc": f1 + 0.2,
        "miou": f1 / 2,
        "precision": f1,
        "recall": f1,
        "provenance": {
            "checkpoint_sha256": f"checkpoint-{fold}",
            "config_sha256": registered.sha256_file(config_path),
        },
        "spatial_split_sha256": "split",
        "manifest_sha256": "manifest",
        "embedding_export": {
            "embedding_file_index_sha256": f"embedding-{fold}",
            "config_path": str(config_path),
        },
        "embedding_registry": {"sha256": "exports"},
        "shot_manifest_sha256": f"shots-{fold}-{task}-{seed}",
        "git_commit": "commit",
        "probe": {"head": "conv3x3"},
    }
    (output / "metrics.json").write_text(json.dumps(metric_payload), encoding="utf-8")
    result_id = f"result-{fold}-{task}-{shot}-{seed}"
    artifact = registered.build_artifact_manifest(
        metric_payload=metric_payload,
        output=output,
        predictions=predictions,
        result_id=result_id,
        registry_path=registry_path,
        label_sha256="labels",
        patch_count=64,
    )
    artifact_path = output / "artifact_manifest.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    record = {
        "result_id": result_id,
        "artifact_sha256": registered.sha256_file(artifact_path),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    record["registry_entry_sha256"] = registered._canonical_sha256(record)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_registered_aggregation_rejects_preliminary_by_default(tmp_path: Path) -> None:
    registry_path = tmp_path / "results.jsonl"
    _write_record(
        tmp_path,
        registry_path,
        fold=0,
        task="building",
        shot="5",
        seed=42,
        f1=0.4,
        paper_eligible=False,
    )

    with pytest.raises(ValueError, match="paper_eligible"):
        aggregate.load_verified_records(registry_path, family="full_150", allow_preliminary=False)


def test_registered_aggregation_uses_seed_level_fold_means(tmp_path: Path) -> None:
    registry_path = tmp_path / "results.jsonl"
    for seed, base in ((42, 0.2), (43, 0.4), (44, 0.6)):
        for fold, increment in ((0, 0.0), (1, 0.1)):
            _write_record(
                tmp_path,
                registry_path,
                fold=fold,
                task="building",
                shot="5",
                seed=seed,
                f1=base + increment,
                paper_eligible=True,
            )

    records = aggregate.load_verified_records(
        registry_path, family="full_150", allow_preliminary=False
    )
    report = aggregate.summarize_records(
        records,
        tasks=("building",),
        shots=("5",),
        folds=(0, 1),
        seeds=(42, 43, 44),
    )
    metric = report["cells"]["building|5"]["metrics"]["f1_at_threshold"]
    assert metric["per_seed_fold_means"] == pytest.approx({"42": 0.25, "43": 0.45, "44": 0.65})
    assert metric["mean"] == pytest.approx(0.45)
    assert metric["std"] == pytest.approx(0.2)


def test_registered_aggregation_rejects_missing_expected_cell(tmp_path: Path) -> None:
    registry_path = tmp_path / "results.jsonl"
    _write_record(
        tmp_path,
        registry_path,
        fold=0,
        task="building",
        shot="5",
        seed=42,
        f1=0.4,
        paper_eligible=True,
    )
    records = aggregate.load_verified_records(
        registry_path, family="full_150", allow_preliminary=False
    )

    with pytest.raises(ValueError, match="Missing registered result"):
        aggregate.summarize_records(
            records,
            tasks=("building",),
            shots=("5",),
            folds=(0, 1),
            seeds=(42,),
        )


def test_registered_aggregation_rejects_duplicate_folds_or_seeds(tmp_path: Path) -> None:
    registry_path = tmp_path / "results.jsonl"
    _write_record(
        tmp_path,
        registry_path,
        fold=0,
        task="building",
        shot="5",
        seed=42,
        f1=0.4,
        paper_eligible=True,
    )
    records = aggregate.load_verified_records(
        registry_path, family="full_150", allow_preliminary=False
    )

    with pytest.raises(ValueError, match="unique"):
        aggregate.summarize_records(
            records,
            tasks=("building",),
            shots=("5",),
            folds=(0, 0),
            seeds=(42,),
        )


def test_primary_matrix_requires_all_registered_dimensions() -> None:
    assert aggregate.is_primary_matrix(
        tasks=("building", "road", "water"),
        shots=("5", "10"),
        folds=(0, 1, 2, 3, 4),
        seeds=(42, 43, 44),
    )
    assert not aggregate.is_primary_matrix(
        tasks=("building",),
        shots=("5",),
        folds=(0, 1, 2, 3, 4),
        seeds=(42, 43, 44),
    )


def test_paper_eligible_aggregation_requires_registry_anchor(tmp_path: Path) -> None:
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="registry-anchor"):
        aggregate.verify_result_registry_anchor(registry_path, None)


def test_external_archive_copy_must_match_the_anchored_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "results.jsonl"
    registry_path.write_text("registry", encoding="utf-8")
    anchor_path = tmp_path / "anchor.json"

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"tampered"

    monkeypatch.setattr(aggregate, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(registered, "verify_git_head_file", lambda _path: None)
    anchor_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_path": str(registry_path.resolve()),
                "registry_sha256": registered.sha256_file(registry_path),
                "external_archive_doi": "https://doi.org/10.5281/zenodo.1",
                "external_archive_registry_url": "https://zenodo.org/records/1/files/registry.jsonl",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="archive hash"):
        aggregate.verify_result_registry_anchor(registry_path, anchor_path)
