from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/run_harbin_strict_conv3x3_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_strict_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_declares_all_paired_strict_cells() -> None:
    module = _load_module()
    matrix_path = (
        Path(__file__).parents[1] / "configs/eval/harbin_strict_conv3x3_matrix_20260728.json"
    )
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    jobs = module.build_jobs(matrix)

    assert len(jobs) == 270
    assert {job["family"] for job in jobs} == {
        "p10c_harbin_scratch",
        "p10c_haidian_frozen_harbin",
        "aef_annual_2025",
    }
    assert {job["shot"] for job in jobs} == {5, 10}
    assert matrix["threshold_rule"]["selection_split"] == "spatial_validation_only"
