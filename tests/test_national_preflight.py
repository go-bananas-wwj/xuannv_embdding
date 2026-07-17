from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/national_preflight.py"
SPEC = importlib.util.spec_from_file_location("national_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_rejects_partial_artifact_and_missing_osm_snapshot(tmp_path: Path) -> None:
    policy = {
        "support_period": {
            "archive_months": MODULE._month_sequence("2025-04", "2026-04"),
            "training_window_months": 6,
            "training_windows": [["2025-04"] * 6 for _ in range(8)],
        },
        "quality_gate": {
            "loss_uses_per_pixel_valid_masks": True,
            "fail_closed_on_missing_mask": True,
            "allow_low_quality_fallback": False,
        },
        "osm": {
            "snapshot": "https://example.test/china-250101.osm.pbf",
            "required_snapshot_not_later_than": "2025-04-01",
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "broken.osm.pbf.partial").write_bytes(b"partial")

    report = MODULE.build_report(policy_path, data_root)
    assert not report["pixel_download_permitted"]
    assert not report["gates"]["osm_training_snapshot"]["passed"]
    assert not report["gates"]["no_incomplete_artifacts"]["passed"]
