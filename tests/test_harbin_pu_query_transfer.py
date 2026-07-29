from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/run_harbin_pu_query_transfer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_pu_query_transfer", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_rejects_non_380_coverage(tmp_path: Path) -> None:
    """A changed coverage universe must stop the paired protocol before any sampling."""
    module = _load_module()
    config = tmp_path / "bad_config.json"
    config.write_text(json.dumps({"coverage_patch_count": 379}), encoding="utf-8")

    with pytest.raises(ValueError, match="380"):
        module.prepare_harbin_pu_query(config, tmp_path / "output")


def test_polygon_schedule_is_reused_for_every_family(tmp_path: Path) -> None:
    """Family identity must not enter the prompt sampling key."""
    module = _load_module()
    schedule = {
        "building|0|42|5": {
            "support_patch_ids": ["harbin_patch_000001"],
            "polygon_indices": [3, 9, 12, 17, 21],
        }
    }
    prepared = module.PreparedProtocol(
        patch_ids=["harbin_patch_000001"],
        label_ids={"harbin_patch_000001": "patch_000001"},
        schedules=schedule,
    )

    assert prepared.schedule_for("building", 0, 42, 5) == schedule["building|0|42|5"]
    assert prepared.schedule_for("building", 0, 42, 5) == schedule["building|0|42|5"]


def test_label_resolution_uses_locked_source_patch_id() -> None:
    """A missing mapping cannot fall back to string rewriting or an unpaired label."""
    module = _load_module()
    mapping = {"harbin_patch_000001": "patch_000001"}

    assert module.resolve_label_id("harbin_patch_000001", mapping) == "patch_000001"
    with pytest.raises(KeyError, match="unlocked"):
        module.resolve_label_id("harbin_patch_000002", mapping)
