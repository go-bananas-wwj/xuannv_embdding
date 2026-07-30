from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts/eval/run_harbin_p10c_highres_mask_diagnostic.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("harbin_highres_mask_diagnostic", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCRIPT_PATH}")
    module = types.ModuleType(spec.name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_masked_provenance_requires_both_sources_disabled() -> None:
    """No P10C diagnostic cell may use a seal that leaves either high-res source available."""
    module = _load_module()
    provenance = {
        "embedding_verification": {
            "patch_count": 380,
            "shape": [64, 128, 128],
            "files": {f"patch_{index:03d}": "hash" for index in range(380)},
        },
        "highres_mask_verification": {
            "patch_count": 380,
            "disabled_highres_sources": ["highres_optical_haidian", "highres_sar_haidian"],
            "all_mask_sums_zero": False,
        },
        "unmasked_embedding_as_masked_result": False,
    }

    with pytest.raises(ValueError, match="all_mask_sums_zero"):
        module.require_masked_provenance(provenance)
