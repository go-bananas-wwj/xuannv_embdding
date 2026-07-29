from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/run_harbin_frozen_transfer_multihead.py"
CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/eval/harbin_frozen_transfer_multihead_20260729.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_frozen_transfer_multihead", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_primary_matrix_rejects_harbin_scratch_family(tmp_path: Path) -> None:
    """The primary table must reject scratch before reading any family export."""
    module = _load_module()
    config_with_scratch = tmp_path / "scratch.json"
    config_with_scratch.write_text(
        json.dumps({"primary_families": [{"id": "p10c_harbin_scratch"}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="scratch"):
        module.prepare_multihead_matrix(config_with_scratch, tmp_path / "output")


def test_primary_matrix_has_matched_readers(tmp_path: Path) -> None:
    """Frozen P10C and AEF use one identical ordered reader suite."""
    module = _load_module()

    prepared = module.prepare_multihead_matrix(CONFIG_PATH, tmp_path / "output")

    assert prepared.heads == (
        "linear",
        "wide_mlp",
        "deep_wide_mlp",
        "conv3x3",
        "unet",
        "deeplab_lite",
    )
    assert tuple(family["id"] for family in prepared.families) == (
        "p10c_haidian_frozen_harbin",
        "aef_annual_2025",
    )


def test_linear_reader_has_no_spatial_kernel() -> None:
    """The linear probe is exactly a 1×1 pixelwise readout."""
    module = _load_module()

    assert module.build_reader("linear", 64).kernel_size == (1, 1)


def test_result_is_published_only_after_metrics_and_predictions_exist(tmp_path: Path) -> None:
    """A finished cell appears only as a complete metrics/predictions bundle."""
    module = _load_module()
    output = tmp_path / "cell"
    metrics = {"f1": 0.5}
    predictions = {
        "validation": {"patch_ids": ["val"], "probabilities": np.zeros((1, 2, 2))},
        "test": {"patch_ids": ["test"], "probabilities": np.ones((1, 2, 2))},
    }

    module.write_cell_atomically(output, metrics, predictions)

    assert (output / "metrics.json").is_file()
    assert (output / "predictions_test.npz").is_file()
    assert (output / "predictions_validation.npz").is_file()


def test_reader_cell_accepts_the_task_field_emitted_by_matrix_jobs() -> None:
    """Worker expansion must pass every job field into the reader-cell interface."""
    module = _load_module()

    assert "task" in inspect.signature(module.run_reader_cell).parameters
