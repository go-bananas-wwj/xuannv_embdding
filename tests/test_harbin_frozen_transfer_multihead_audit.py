from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/eval/audit_harbin_frozen_transfer_multihead.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_frozen_transfer_multihead_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_rejects_scratch_family_contamination(tmp_path: Path) -> None:
    """A primary result root cannot contain a Harbin scratch family result at all."""
    module = _load_module()
    root = tmp_path / "primary"
    scratch_metrics = root / "results" / "p10c_harbin_scratch" / "metrics.json"
    scratch_metrics.parent.mkdir(parents=True)
    scratch_metrics.write_text(json.dumps({"family": "p10c_harbin_scratch"}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="scratch"):
        module.audit_multihead_results(root, config, tmp_path / "audit")
