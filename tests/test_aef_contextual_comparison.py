from __future__ import annotations

import pytest

from scripts.report import aggregate_aef_contextual_comparison as contextual


def _record(protocol: str, *, cell: tuple[str, str, int, int], test_ids: str = "test") -> dict[str, object]:
    task, shot, fold, seed = cell
    return {
        "result_id": f"{protocol}-result",
        "metric_provenance": {
            "protocol_id": protocol,
            "task": task,
            "shot": shot,
            "fold": fold,
            "shot_seed": seed,
            "label_sha256": "labels",
            "spatial_split_sha256": "split",
            "manifest_sha256": "manifest",
            "shot_manifest_sha256": "schedule",
            "test_patch_ids_sha256": test_ids,
            "per_patch_confusion": {"patch_000000": {"tp": 1, "fp": 0, "fn": 0, "tn": 1}},
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
        payload["family"] = "full_150"
    return record


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
