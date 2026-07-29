from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/eval/audit_harbin_pu_query_transfer.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_pu_query_transfer_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture_protocol(tmp_path: Path, *, families: list[str]) -> tuple[Path, Path]:
    split_path = _write_json(
        tmp_path / "split.json",
        {"folds": [{"train": ["train"], "val": ["val"], "test": ["test"], "buffer": []}]},
    )
    matrix_path = _write_json(
        tmp_path / "matrix.json",
        {
            "families": [{"id": family} for family in families],
            "tasks": ["building"],
            "folds": [0],
            "seeds": [42],
            "spatial_split": {"path": str(split_path), "sha256": _sha256(split_path)},
        },
    )
    config_path = _write_json(
        tmp_path / "config.json",
        {
            "protocol_id": "test_protocol",
            "base_matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "polygon_counts": [1],
            "prototype_modes": ["single"],
            "query_modes": ["disabled"],
        },
    )
    root = tmp_path / "v3"
    lock_path = _write_json(
        root / "protocol_input_lock.json",
        {"protocol_id": "test_protocol", "expected_cells": len(families)},
    )
    for family in families:
        _write_json(
            root / "results" / family / "result.json",
            {
                "protocol_id": "test_protocol",
                "family": family,
                "task": "building",
                "fold": 0,
                "seed": 42,
                "polygon_count": 1,
                "prototype_mode": "single",
                "query_mode": "disabled",
                "support_schedule": {
                    "support_polygon_sha256": "shared",
                    "support_polygons": [{"patch_id": "train"}],
                },
                "test_patch_ids": ["test"],
                "protocol_input_lock": str(lock_path),
                "protocol_input_lock_sha256": _sha256(lock_path),
                "test_metrics": {"f1": 0.4, "ap": 0.5, "auc_roc": 0.6},
            },
        )
    return root, config_path


def test_audit_rejects_missing_paired_family_cell(tmp_path: Path) -> None:
    """A completed grid needs all three paired feature families for every cell."""
    module = _load_module()
    root, config_path = _fixture_protocol(tmp_path, families=["scratch", "frozen", "aef"])
    (root / "results" / "aef" / "result.json").unlink()

    with pytest.raises(ValueError, match="missing paired cell"):
        module.audit_results(root, config_path, tmp_path / "audit")


def test_audit_writes_group_summary_and_excludes_unselected_roots(tmp_path: Path) -> None:
    """The audit uses only its explicit root and writes a reproducible aggregate."""
    module = _load_module()
    root, config_path = _fixture_protocol(tmp_path, families=["scratch", "frozen", "aef"])

    summary = module.audit_results(root, config_path, tmp_path / "audit")

    assert summary["audit"]["result_cell_count"] == 3
    assert summary["audit"]["excluded_roots"] == []
    assert summary["summary_rows"][0]["f1_mean"] == pytest.approx(0.4)
    assert (tmp_path / "audit" / "summary.csv").is_file()
    assert (tmp_path / "audit" / "audit.json").is_file()
