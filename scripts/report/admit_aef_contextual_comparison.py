#!/usr/bin/env python3
"""Seal the registered AEF/Xuannv contextual comparison before paper rendering.

The admission is deliberately a separate, write-once, Git-tracked artifact.
It keeps the annual-AEF contextual evidence boundary intact while binding every
input result identity and a deterministic recomputation of the bootstrap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.eval import run_registered_paper_downstream as registered
from scripts.eval.run_registered_paper_downstream import sha256_file
from scripts.report import aggregate_aef_contextual_comparison as contextual
from scripts.report import aggregate_registered_paper_results as aggregate
from scripts.report import compare_aef_contextual_protocols as comparison
from scripts.report import paired_spatial_bootstrap as bootstrap

CANONICAL_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "configs/eval/release_admissions/rse_aef_annual_2025_contextual_comparison_admission.json"
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    Path(contextual.__file__).resolve(),
    Path(comparison.__file__).resolve(),
    Path(bootstrap.__file__).resolve(),
    Path(aggregate.__file__).resolve(),
    Path(registered.__file__).resolve(),
)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return registered._canonical_sha256(dict(payload))


def build_contextual_input_identity_snapshot(
    aef_records: Mapping[contextual.CellKey, Mapping[str, object]],
    xuannv_records: Mapping[contextual.CellKey, Mapping[str, object]],
) -> dict[str, object]:
    """Enumerate the exact two ordered 90-cell inputs of one contextual comparison."""
    contextual.verify_contextual_pairing(aef_records, xuannv_records)
    baseline = sorted(
        (aggregate.sealed_result_identity(dict(record)) for record in aef_records.values()),
        key=lambda item: item["result_id"],
    )
    candidate = sorted(
        (aggregate.sealed_result_identity(dict(record)) for record in xuannv_records.values()),
        key=lambda item: item["result_id"],
    )
    snapshot: dict[str, object] = {
        "schema_version": 1,
        "baseline_input_results": baseline,
        "candidate_input_results": candidate,
        "baseline_result_count": len(baseline),
        "candidate_result_count": len(candidate),
        "baseline_input_results_sha256": _canonical_sha256({"results": baseline}),
        "candidate_input_results_sha256": _canonical_sha256({"results": candidate}),
    }
    snapshot["sha256"] = _canonical_sha256(snapshot)
    return snapshot


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing contextual comparison report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Contextual comparison report must be a JSON object")
    return report


def _verify_report_recomputation(
    report: Mapping[str, object],
    aef_records: Mapping[contextual.CellKey, Mapping[str, object]],
    xuannv_records: Mapping[contextual.CellKey, Mapping[str, object]],
    *,
    patch_metadata_path: Path,
) -> None:
    """Refuse promotion unless the published report exactly replays the registered bootstrap."""
    expected = comparison.compare_contextual_records(
        aef_records,
        xuannv_records,
        n_resamples=bootstrap.PROTOCOL_N_RESAMPLES,
        seed=bootstrap.PROTOCOL_RANDOM_SEED,
        patch_metadata_path=patch_metadata_path,
    )
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Contextual comparison report differs from sealed recomputation: {key}")


def build_contextual_comparison_admission(
    *,
    aef_registry_path: Path,
    xuannv_registry_path: Path,
    comparison_report_path: Path,
    patch_metadata_path: Path = bootstrap.PAPER_PATCH_METADATA,
) -> dict[str, Any]:
    """Build a paper-eligible contextual admission without changing source artifacts."""
    for source_path in SOURCE_FILES:
        registered.verify_git_head_file(source_path)
    aef = contextual.load_verified_contextual_records(
        aef_registry_path, expected_protocol=contextual.AEF_PROTOCOL
    )
    xuannv = contextual.load_verified_contextual_records(
        xuannv_registry_path, expected_protocol=contextual.XUANNV_PROTOCOL
    )
    snapshot = build_contextual_input_identity_snapshot(aef, xuannv)
    report = _load_report(comparison_report_path)
    _verify_report_recomputation(
        report, aef, xuannv, patch_metadata_path=patch_metadata_path
    )
    return {
        "schema_version": 1,
        "comparison_id": "annual_aef_2025_vs_monthly_xuannv_202604_contextual",
        "paper_eligible": False,
        "admission_status": "contextual_paired_comparison_pending_external_release_admissions",
        "evidence_scope": "osm_assisted_spatial_readout",
        "time_inequivalent_contextual": True,
        "temporal_statement": report["temporal_statement"],
        "prohibited_claims": report["prohibited_claims"],
        "aef_registry": {
            "path": str(aef_registry_path.resolve()),
            "sha256": sha256_file(aef_registry_path),
        },
        "xuannv_registry": {
            "path": str(xuannv_registry_path.resolve()),
            "sha256": sha256_file(xuannv_registry_path),
        },
        "comparison_report": {
            "path": str(comparison_report_path.resolve()),
            "sha256": sha256_file(comparison_report_path),
        },
        "input_identity_snapshot": snapshot,
        "source_files": [
            {"path": str(path.relative_to(Path(__file__).resolve().parents[2])), "sha256": sha256_file(path)}
            for path in SOURCE_FILES
        ],
    }


def write_contextual_comparison_admission(output_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Write the admission once at its committed canonical path."""
    if output_path.resolve() != CANONICAL_OUTPUT.resolve():
        raise ValueError(f"Contextual admission must use canonical path: {CANONICAL_OUTPUT}")
    payload = build_contextual_comparison_admission(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aef-registry", type=Path, required=True)
    parser.add_argument("--xuannv-registry", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-metadata", type=Path, default=bootstrap.PAPER_PATCH_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_contextual_comparison_admission(
        args.output,
        aef_registry_path=args.aef_registry,
        xuannv_registry_path=args.xuannv_registry,
        comparison_report_path=args.comparison_report,
        patch_metadata_path=args.patch_metadata,
    )
    print(json.dumps({"output": str(args.output.resolve()), "sha256": _canonical_sha256(payload)}))


if __name__ == "__main__":
    main()
