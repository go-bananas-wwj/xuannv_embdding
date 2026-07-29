from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts/eval/export_harbin_p10c_highres_masked.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("harbin_highres_masked_export", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = types.ModuleType(spec.name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mask_contract_zeros_optical_and_records_missing_sar() -> None:
    """The diagnostic must force optical off and explicitly retain SAR as unavailable."""
    module = _load_module()
    masks = {"highres_optical_haidian": torch.ones(1, 1, 4, 4)}

    masked, audit = module.mask_highres_availability(masks, module.DISABLED_HIGHRES_SOURCES)

    assert torch.count_nonzero(masked["highres_optical_haidian"]) == 0
    assert audit["highres_optical_haidian"] == {
        "present_in_batch": True,
        "available_before": True,
        "available_after": False,
        "mask_sum_before": 16.0,
        "mask_sum_after": 0.0,
    }
    assert audit["highres_sar_haidian"] == {
        "present_in_batch": False,
        "available_before": False,
        "available_after": False,
        "mask_sum_before": 0.0,
        "mask_sum_after": 0.0,
    }


def test_paper_export_helper_is_bound_to_repository_path() -> None:
    """The standalone exporter must not resolve the unrelated downstreams/scripts package."""
    module = _load_module()

    assert (
        Path(module.paper_export.__file__).resolve()
        == (REPO_ROOT / "scripts/eval/export_p10c_harbin_paper_embeddings.py").resolve()
    )


def test_availability_audit_rejects_any_nonzero_mask_after_export(tmp_path: Path) -> None:
    """A sealed masked export cannot accept an optical or SAR availability leak."""
    module = _load_module()
    audit_path = tmp_path / "highres_availability_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "patches": {
                    "harbin_patch_000001": {
                        "highres_optical_haidian": {"mask_sum_after": 1.0},
                        "highres_sar_haidian": {"mask_sum_after": 0.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="availability remains enabled"):
        module.verify_mask_audit(audit_path, ["harbin_patch_000001"])
