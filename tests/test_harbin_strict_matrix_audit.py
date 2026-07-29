from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/audit_harbin_strict_conv3x3_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_strict_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_cells_and_summary_use_all_folds_and_seeds() -> None:
    module = _load_module()
    matrix = {
        "families": [{"id": "scratch"}, {"id": "aef"}],
        "tasks": ["building"],
        "folds": [0, 1],
        "shots": [5],
        "seeds": [42, 43, 44],
    }
    rows = [
        {
            "family": "scratch",
            "task": "building",
            "shot": 5,
            "f1_at_threshold": value,
            "ap": value + 0.1,
            "auc_roc": value + 0.2,
            "threshold": 0.5,
        }
        for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]

    cells = module.expected_cells(matrix)
    summary = module.aggregate_rows(rows)

    assert len(cells) == 12
    assert summary[0]["n"] == 6
    assert summary[0]["f1_mean"] == 0.35
    assert round(summary[0]["f1_std"], 6) == round(0.18708286933869708, 6)
