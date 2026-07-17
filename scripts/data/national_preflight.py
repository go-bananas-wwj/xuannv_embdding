#!/usr/bin/env python3
"""Create a fail-closed, metadata-only readiness report for China V1 data prep.

This command intentionally does not contact STAC or download imagery.  It
freezes the policy hash, verifies its temporal contract, records the executing
source tree and identifies local artifacts that must not enter the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _month_sequence(start: str, end: str) -> list[str]:
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    values: list[str] = []
    while (year, month) <= (end_year, end_month):
        values.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def _validate_policy(policy: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    support = policy.get("support_period", {})
    months = support.get("archive_months")
    expected = _month_sequence("2025-04", "2026-04")
    if months != expected:
        failures.append("support_period.archive_months must be 2025-04 through 2026-04 inclusive")
    if support.get("training_window_months") != 6:
        failures.append("support_period.training_window_months must be 6")
    windows = support.get("training_windows", [])
    if len(windows) != 8 or any(len(window) != 6 for window in windows):
        failures.append("support_period.training_windows must contain eight six-month windows")

    quality = policy.get("quality_gate", {})
    if not quality.get("loss_uses_per_pixel_valid_masks"):
        failures.append("quality_gate.loss_uses_per_pixel_valid_masks must be true")
    if not quality.get("fail_closed_on_missing_mask"):
        failures.append("quality_gate.fail_closed_on_missing_mask must be true")
    if quality.get("allow_low_quality_fallback"):
        failures.append("quality_gate.allow_low_quality_fallback must be false")

    osm = policy.get("osm", {})
    if osm.get("snapshot") is not None:
        failures.append("osm.snapshot must remain null until a pre-period checksum is verified")
    if osm.get("required_snapshot_not_later_than") != "2025-04-01":
        failures.append("osm.required_snapshot_not_later_than must be 2025-04-01")
    return failures


def build_report(policy_path: Path, data_root: Path) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    partials = sorted(str(path) for path in data_root.rglob("*.partial")) if data_root.exists() else []
    usage = shutil.disk_usage(data_root if data_root.exists() else data_root.parent)
    policy_failures = _validate_policy(policy)
    snapshot = policy.get("osm", {}).get("snapshot")
    gates = {
        "policy_contract": {"passed": not policy_failures, "failures": policy_failures},
        "osm_training_snapshot": {
            "passed": isinstance(snapshot, str) and bool(snapshot),
            "reason": "A pre-period OSM URL and checksum must be registered before OSM weak supervision.",
        },
        "no_incomplete_artifacts": {
            "passed": not partials,
            "partials": partials,
            "reason": "Partial downloads are quarantined and are never valid source artifacts.",
        },
        "free_space": {
            "passed": usage.free >= 2 * 1024**4,
            "free_bytes": usage.free,
            "free_tib": round(usage.free / 1024**4, 3),
            "required_tib": 2.0,
        },
    }
    return {
        "schema_version": "china_v1_preflight_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": _sha256(policy_path),
        "data_root": str(data_root.resolve()),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"])),
        "python_executable": sys.executable,
        "python_path_head": sys.path[:3],
        "gates": gates,
        "pixel_download_permitted": all(gate["passed"] for gate in gates.values()),
        "note": "This report authorizes neither STAC imagery nor national model training; both require a frozen atlas and a separate 2,000-chip pilot approval.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.policy, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
