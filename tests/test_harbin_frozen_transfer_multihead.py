from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
