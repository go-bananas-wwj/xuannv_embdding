#!/usr/bin/env python3
"""Fail-closed paired bootstrap for the registered annual-AEF contextual comparison.

This consumer deliberately accepts only the annual-2025 AEF versus monthly
Xuannv V5 protocol pair.  It is separate from the within-V5 family comparator
because the two encoders have different source-time products.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.eval.run_registered_paper_downstream import sha256_file
from scripts.report import aggregate_aef_contextual_comparison as contextual
from scripts.report import paired_spatial_bootstrap as bootstrap


def _groups_for_cell(
    records: Mapping[contextual.CellKey, Mapping[str, object]], *, task: str, shot: str
) -> dict[bootstrap.GroupKey, bootstrap.PatchConfusion]:
    groups: dict[bootstrap.GroupKey, bootstrap.PatchConfusion] = {}
    for fold in range(5):
        for probe_seed in (42, 43, 44):
            payload = contextual._payload(records[(task, shot, fold, probe_seed)])
            confusion = payload.get("per_patch_confusion")
            if not isinstance(confusion, Mapping):
                raise ValueError("Contextual comparison requires per-patch test confusion provenance")
            groups[(fold, probe_seed)] = confusion
    return groups


def _point_estimates(
    aef_records: Mapping[contextual.CellKey, Mapping[str, object]],
    xuannv_records: Mapping[contextual.CellKey, Mapping[str, object]],
) -> dict[str, object]:
    """Report sealed AP/AUC point estimates beside confusion-reconstructable CIs."""
    baseline = contextual.summarize_contextual_records(aef_records)
    candidate = contextual.summarize_contextual_records(xuannv_records)
    output: dict[str, object] = {}
    for key, aef_summary in baseline.items():
        xuannv_summary = candidate[key]
        aef_metrics = aef_summary["metrics"]
        xuannv_metrics = xuannv_summary["metrics"]
        assert isinstance(aef_metrics, Mapping) and isinstance(xuannv_metrics, Mapping)
        output[key] = {
            "baseline": aef_summary,
            "candidate": xuannv_summary,
            "candidate_minus_baseline": {
                metric: float(xuannv_metrics[metric]["mean"])
                - float(aef_metrics[metric]["mean"])
                for metric in contextual.METRICS
            },
            "uncertainty_rule": "bootstrap_ci_for_confusion_metrics_only",
            "ap_auc_rule": "sealed_point_estimates_only_no_patch_confusion_ci",
        }
    return output


def compare_contextual_records(
    aef_records: Mapping[contextual.CellKey, Mapping[str, object]],
    xuannv_records: Mapping[contextual.CellKey, Mapping[str, object]],
    *,
    n_resamples: int = bootstrap.PROTOCOL_N_RESAMPLES,
    seed: int = bootstrap.PROTOCOL_RANDOM_SEED,
    patch_metadata_path: Path = bootstrap.PAPER_PATCH_METADATA,
) -> dict[str, Any]:
    """Compute registered contextual intervals after exact source and readout pairing.

    The result remains contextual and preliminary until a separate immutable
    comparison-admission record seals both input registries and this report.
    """
    if n_resamples != bootstrap.PROTOCOL_N_RESAMPLES:
        raise ValueError(
            f"Contextual comparison requires exactly {bootstrap.PROTOCOL_N_RESAMPLES} resamples"
        )
    if seed != bootstrap.PROTOCOL_RANDOM_SEED:
        raise ValueError(
            f"Contextual comparison requires registered random seed {bootstrap.PROTOCOL_RANDOM_SEED}"
        )
    contextual.verify_contextual_pairing(aef_records, xuannv_records)
    bootstrap.validate_paper_patch_metadata(patch_metadata_path)
    bounds_by_patch = bootstrap.load_patch_bounds(patch_metadata_path)
    comparisons: dict[str, object] = {}
    for task in ("building", "road", "water"):
        for shot in ("5", "10"):
            comparisons[f"{task}|{shot}"] = bootstrap.hierarchical_paired_bootstrap(
                _groups_for_cell(aef_records, task=task, shot=shot),
                _groups_for_cell(xuannv_records, task=task, shot=shot),
                n_resamples=n_resamples,
                seed=seed,
                bounds_by_patch=bounds_by_patch,
            )
    return {
        "schema_version": 1,
        "comparison_id": "annual_aef_2025_vs_monthly_xuannv_202604_contextual",
        "baseline_protocol_id": contextual.AEF_PROTOCOL,
        "candidate_protocol_id": contextual.XUANNV_PROTOCOL,
        "comparison_direction": "xuannv_minus_aef",
        "time_inequivalent_contextual": True,
        "evidence_scope": "osm_assisted_spatial_readout",
        "baseline_output_identifier": "annual_2025",
        "candidate_month": "202604",
        "temporal_statement": (
            "Annual AEF 2025 versus monthly Xuannv 2026-04 under an aligned spatial "
            "downstream protocol."
        ),
        "prohibited_claims": [
            "matched_temporal_information",
            "matched_inputs",
            "labelled_patch_efficiency_against_information_matched_baseline",
            "independent_transfer",
            "geographically_unseen_pretraining",
            "contemporaneous_ground_truth",
            "monthly_change_evidence",
            "global_superiority",
        ],
        "bootstrap": {
            "n_resamples": n_resamples,
            "random_seed": seed,
            "hierarchy": ["fold", "complete_2x2_geographic_cluster", "support_schedule_seed"],
            "patch_metadata_path": str(patch_metadata_path.resolve()),
            "patch_metadata_sha256": sha256_file(patch_metadata_path),
        },
        "comparison_matrix_sha256": contextual.contextual_matrix_sha256(),
        "point_estimates": _point_estimates(aef_records, xuannv_records),
        "comparisons": comparisons,
        "admission_status": "contextual_comparison_pending_admission",
        "paper_eligible": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aef-registry", type=Path, required=True)
    parser.add_argument("--xuannv-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-metadata", type=Path, default=bootstrap.PAPER_PATCH_METADATA)
    parser.add_argument("--n-resamples", type=int, default=bootstrap.PROTOCOL_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=bootstrap.PROTOCOL_RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Contextual comparison exists; refusing to overwrite: {args.output}")
    aef = contextual.load_verified_contextual_records(
        args.aef_registry, expected_protocol=contextual.AEF_PROTOCOL
    )
    xuannv = contextual.load_verified_contextual_records(
        args.xuannv_registry, expected_protocol=contextual.XUANNV_PROTOCOL
    )
    result = compare_contextual_records(
        aef,
        xuannv,
        n_resamples=args.n_resamples,
        seed=args.seed,
        patch_metadata_path=args.patch_metadata,
    )
    result["aef_registry"] = {
        "path": str(args.aef_registry.resolve()),
        "sha256": sha256_file(args.aef_registry),
    }
    result["xuannv_registry"] = {
        "path": str(args.xuannv_registry.resolve()),
        "sha256": sha256_file(args.xuannv_registry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
