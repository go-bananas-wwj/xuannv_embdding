#!/usr/bin/env python3
"""Fail-closed aggregation for registered downstream-probe results.

The default mode accepts only records explicitly admitted to the paper.  The
``--allow-preliminary`` escape hatch is exclusively for diagnostic coverage
reports and marks every generated artifact as preliminary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable
from urllib.request import urlopen

import yaml

from scripts.eval import run_registered_paper_downstream as registered

METRICS = ("f1_at_threshold", "ap", "auc_roc", "miou", "precision", "recall")
PRIMARY_FOLDS = (0, 1, 2, 3, 4)
PRIMARY_SEEDS = (42, 43, 44)
PRIMARY_TASKS = ("building", "road", "water")
PRIMARY_SHOTS = ("5", "10")
ADMISSION_STATUSES = {
    False: {"registered_preliminary_pending_external_gates"},
    True: {"registered_paper_eligible"},
}
REPORT_KINDS = ("diagnostic", "weak_supervision", "independent_transfer")


def _metric_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("metric_provenance")
    if not isinstance(payload, dict):
        raise ValueError("Registry record lacks metric_provenance")
    return payload


def _family_from_record(record: dict[str, Any]) -> str:
    payload = _metric_payload(record)
    export = payload.get("embedding_export")
    if not isinstance(export, dict) or not isinstance(export.get("config_path"), str):
        raise ValueError("Metric provenance lacks the encoder config path")
    config_path = Path(export["config_path"]).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing attested encoder config: {config_path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Metric provenance lacks encoder provenance")
    if provenance.get("config_sha256") != registered.sha256_file(config_path):
        raise ValueError("Encoder config hash differs from metric provenance")
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict) or not isinstance(config_data.get("data"), dict):
        raise ValueError("Attested encoder config lacks data.paper_fold")
    if int(config_data["data"].get("paper_fold", -1)) != int(payload["fold"]):
        raise ValueError("Attested encoder config fold differs from downstream-result fold")
    if payload.get("protocol_id") == "v5_osm_assisted":
        registered.validate_v5_encoder_config_provenance(
            config_path, str(provenance["config_sha256"])
        )
    family = registered.registered_family_from_experiment(
        Path(export["config_path"]).stem, int(payload["fold"])
    )
    if payload.get("protocol_id") == "v5_osm_assisted" and payload.get("family") not in (
        None,
        family,
    ):
        raise ValueError("V5 result family differs from the attested encoder config")
    return family


def _record_evidence(record: dict[str, Any]) -> dict[str, str]:
    """Read and validate the evidence descriptor sealed into a result payload."""
    payload = _metric_payload(record)
    if not any(field in payload for field in registered.result_evidence("v4_diagnostic")):
        payload = {**payload, **registered.result_evidence("v4_diagnostic")}
    return registered.validate_result_evidence(payload)


def validate_report_evidence(records: Iterable[dict[str, Any]], report_kind: str) -> dict[str, str]:
    """Allow one protocol only in report types that its evidence class permits."""
    if report_kind not in REPORT_KINDS:
        raise ValueError(f"Unknown report kind: {report_kind}")
    evidence = [_record_evidence(record) for record in records]
    if not evidence:
        raise ValueError("Cannot classify an empty result set")
    protocol_ids = {item["protocol_id"] for item in evidence}
    if len(protocol_ids) != 1:
        raise ValueError("Aggregation requires exactly one protocol_id")
    expected = registered.result_evidence(next(iter(protocol_ids)))
    if any(item != expected for item in evidence):
        raise ValueError("Aggregation found inconsistent evidence for one protocol_id")
    allowed = registered.PROTOCOL_DESCRIPTORS[expected["protocol_id"]]["allowed_report_kinds"]
    if report_kind not in allowed:
        human_kind = report_kind.replace("_", "-")
        raise ValueError(
            f"Protocol {expected['protocol_id']} cannot support an {human_kind} report"
        )
    return expected


def validate_registered_v5_matrix_records(records: Iterable[dict[str, Any]], family: str) -> None:
    """Reject v5 result records that do not map to one committed probe cell."""
    for record in records:
        payload = _metric_payload(record)
        if payload.get("protocol_id") != "v5_osm_assisted":
            continue
        if payload.get("family") != family:
            raise ValueError("V5 result family does not match the requested aggregation family")
        probe = payload.get("probe")
        if not isinstance(probe, dict):
            raise ValueError("V5 result lacks probe provenance")
        registered.validate_registered_matrix_cell(
            "v5_osm_assisted",
            {
                "family": family,
                "head": probe.get("head"),
                "task": payload.get("task"),
                "shot": payload.get("shot"),
                "fold": payload.get("fold"),
                "seed": payload.get("shot_seed"),
            },
        )


def load_verified_records(
    registry_path: Path, *, family: str, allow_preliminary: bool
) -> list[dict[str, Any]]:
    """Load one family after cryptographically verifying each artifact sidecar."""
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing result registry: {registry_path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if _family_from_record(record) != family:
            continue
        result_id = record.get("result_id")
        if not isinstance(result_id, str) or result_id in seen_ids:
            raise ValueError(f"Duplicate or invalid result_id for family {family}: {result_id}")
        seen_ids.add(result_id)
        metrics_path = Path(record["metrics_path"])
        artifact_path = metrics_path.parent / "artifact_manifest.json"
        registered.verify_artifact_registry_binding(artifact_path, registry_path)
        payload = _metric_payload(record)
        if json.loads(metrics_path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Metric provenance mismatch: {metrics_path}")
        if record.get("paper_eligible") is not payload.get("paper_eligible"):
            raise ValueError(f"Registry and metric eligibility differ: {result_id}")
        if record.get("admission_status") != payload.get("admission_status"):
            raise ValueError(f"Registry and metric admission status differ: {result_id}")
        eligible = record.get("paper_eligible")
        if (
            not isinstance(eligible, bool)
            or record.get("admission_status") not in ADMISSION_STATUSES[eligible]
        ):
            raise ValueError(f"Invalid eligibility/admission-status pair: {result_id}")
        if not allow_preliminary and record.get("paper_eligible") is not True:
            raise ValueError(
                f"Result {result_id} is not paper_eligible; "
                "use --allow-preliminary only for diagnostics"
            )
        records.append(record)
    if not records:
        raise ValueError(f"No registered records found for family: {family}")
    return records


def _cell_key(task: str, shot: str) -> str:
    return f"{task}|{shot}"


def is_primary_matrix(
    *,
    tasks: tuple[str, ...],
    shots: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> bool:
    """Return true only for the preregistered task/shot/fold/seed matrix."""
    return (
        tasks == PRIMARY_TASKS
        and shots == PRIMARY_SHOTS
        and folds == PRIMARY_FOLDS
        and seeds == PRIMARY_SEEDS
    )


def verify_result_registry_anchor(registry_path: Path, anchor_path: Path | None) -> dict[str, Any]:
    """Require a Git-anchored declaration before emitting paper-eligible output."""
    if anchor_path is None:
        raise ValueError("Paper-eligible aggregation requires --registry-anchor")
    registered.verify_git_head_file(anchor_path)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(anchor, dict) or anchor.get("schema_version") != 1:
        raise ValueError("Invalid result-registry anchor schema")
    if anchor.get("registry_path") != str(registry_path.resolve()):
        raise ValueError("Registry anchor path does not match the requested registry")
    if anchor.get("registry_sha256") != registered.sha256_file(registry_path):
        raise ValueError("Registry anchor hash does not match the requested registry")
    archive_doi = anchor.get("external_archive_doi")
    archive_url = anchor.get("external_archive_registry_url")
    doi_match = (
        re.fullmatch(r"https://doi\.org/10\.5281/zenodo\.(\d+)", archive_doi)
        if isinstance(archive_doi, str)
        else None
    )
    if doi_match is None:
        raise ValueError("Registry anchor requires a versioned Zenodo DOI")
    record_id = doi_match.group(1)
    if not isinstance(archive_url, str) or not re.match(
        rf"https://zenodo\.org/records/{record_id}/files/[^?#]+(?:\?[^#]*)?$", archive_url
    ):
        raise ValueError("Registry archive URL is not a file in the DOI's Zenodo record")
    try:
        with urlopen(archive_url, timeout=30) as response:  # noqa: S310 - DOI-bound archive URL.
            archived_registry = response.read()
    except OSError as exc:
        raise ValueError("Could not fetch the external registry archive") from exc
    if sha256(archived_registry).hexdigest() != anchor["registry_sha256"]:
        raise ValueError("External registry archive hash does not match the anchor")
    return {
        "path": str(anchor_path.resolve()),
        "sha256": registered.sha256_file(anchor_path),
        **anchor,
    }


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    tasks: tuple[str, ...],
    shots: tuple[str, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Compute mean/std across seed-level means after requiring a full matrix."""
    record_list = list(records)
    protocol_ids = {
        str(_metric_payload(record).get("protocol_id", "v4_diagnostic")) for record in record_list
    }
    if len(protocol_ids) != 1:
        raise ValueError("Aggregation requires exactly one protocol_id")
    if len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds):
        raise ValueError("Fold and seed lists must contain unique values")
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in record_list:
        payload = _metric_payload(record)
        key = (
            str(payload["task"]),
            str(payload["shot"]),
            int(payload["fold"]),
            int(payload["shot_seed"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate registered result for cell: {key}")
        indexed[key] = payload

    cells: dict[str, Any] = {}
    for task in tasks:
        for shot in shots:
            per_seed_payloads: dict[int, list[dict[str, Any]]] = {}
            for seed in seeds:
                rows: list[dict[str, Any]] = []
                for fold in folds:
                    key = (task, shot, fold, seed)
                    if key not in indexed:
                        raise ValueError(f"Missing registered result for cell: {key}")
                    rows.append(indexed[key])
                per_seed_payloads[seed] = rows
            metric_summary: dict[str, Any] = {}
            for metric in METRICS:
                per_seed = {
                    str(seed): mean(float(row[metric]) for row in rows)
                    for seed, rows in per_seed_payloads.items()
                }
                values = list(per_seed.values())
                pooled = [float(row[metric]) for rows in per_seed_payloads.values() for row in rows]
                metric_summary[metric] = {
                    "per_seed_fold_means": per_seed,
                    "mean": mean(values),
                    "std": stdev(values) if len(values) > 1 else 0.0,
                    "n_seeds": len(values),
                    "pooled_fold_mean": mean(pooled),
                    "pooled_fold_std": stdev(pooled) if len(pooled) > 1 else 0.0,
                    "n_fold_seed_runs": len(pooled),
                }
            cells[_cell_key(task, shot)] = {"task": task, "shot": shot, "metrics": metric_summary}
    return {"cells": cells, "folds": list(folds), "seeds": list(seeds)}


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "shot",
                "metric",
                "mean",
                "std",
                "n_seeds",
                "pooled_fold_mean",
                "n_fold_seed_runs",
                "family",
                "preliminary",
                "primary_matrix",
                "report_kind",
                "protocol_id",
                "evidence_class",
                "label_independence_status",
                "registry_sha256",
                "selected_results_sha256",
            ],
        )
        writer.writeheader()
        for cell in report["summary"]["cells"].values():
            for metric, values in cell["metrics"].items():
                writer.writerow(
                    {
                        "task": cell["task"],
                        "shot": cell["shot"],
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                        "n_seeds": values["n_seeds"],
                        "pooled_fold_mean": values["pooled_fold_mean"],
                        "n_fold_seed_runs": values["n_fold_seed_runs"],
                        "family": report["family"],
                        "preliminary": report["preliminary"],
                        "primary_matrix": report["primary_matrix"],
                        "report_kind": report["report_kind"],
                        "protocol_id": report["protocol_id"],
                        "evidence_class": report["evidence_class"],
                        "label_independence_status": report["label_independence_status"],
                        "registry_sha256": report["registry_sha256"],
                        "selected_results_sha256": report["selected_results_sha256"],
                    }
                )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    status = (
        "Preliminary diagnostic only"
        if report["preliminary"]
        else "Paper-eligible registered result"
    )
    lines = [
        "# Registered downstream aggregation",
        "",
        f"**Status:** {status}.",
        f"**Family:** `{report['family']}`; **primary matrix:** `{report['primary_matrix']}`.",
        f"**report_kind:** `{report['report_kind']}`; **protocol_id:** `{report['protocol_id']}`.",
        f"**evidence_class:** `{report['evidence_class']}`; "
        f"**label_independence_status:** `{report['label_independence_status']}`.",
        f"**Registry SHA-256:** `{report['registry_sha256']}`.",
        f"**Selected-result identity:** `{report['selected_results_sha256']}`.",
        "",
        "Values are mean +/- sample standard deviation across per-seed fold means.",
        "",
        "| Task | Shot | F1 | AP | AUC-ROC | mIoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in report["summary"]["cells"].values():
        values = cell["metrics"]
        formatted = [
            f"{values[metric]['mean']:.4f} +/- {values[metric]['std']:.4f}" for metric in METRICS
        ]
        lines.append(f"| {cell['task']} | {cell['shot']} | " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=("building", "road", "water"))
    parser.add_argument("--shots", nargs="+", default=("5", "10"))
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--allow-preliminary", action="store_true")
    parser.add_argument("--protocol", choices=tuple(registered.PROTOCOL_DESCRIPTORS), default=None)
    parser.add_argument("--report-kind", choices=REPORT_KINDS, default="diagnostic")
    parser.add_argument(
        "--registry-anchor",
        type=Path,
        default=None,
        help="Git-tracked immutable anchor required for paper-eligible aggregation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks, shots, folds, seeds = (
        tuple(args.tasks),
        tuple(args.shots),
        tuple(args.folds),
        tuple(args.seeds),
    )
    for name, values in (("tasks", tasks), ("shots", shots), ("folds", folds), ("seeds", seeds)):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique values")
    primary_matrix = is_primary_matrix(tasks=tasks, shots=shots, folds=folds, seeds=seeds)
    if not args.allow_preliminary and not primary_matrix:
        raise ValueError(
            "Paper-eligible aggregation requires folds 0..4 and seeds 42,43,44 exactly"
        )
    anchor = (
        None
        if args.allow_preliminary
        else verify_result_registry_anchor(args.registry, args.registry_anchor)
    )
    records = load_verified_records(
        args.registry, family=args.family, allow_preliminary=args.allow_preliminary
    )
    evidence = validate_report_evidence(records, args.report_kind)
    if evidence["protocol_id"] == "v5_osm_assisted" and args.protocol != "v5_osm_assisted":
        raise ValueError("V5 aggregation requires --protocol v5_osm_assisted")
    if args.protocol is not None and evidence["protocol_id"] != args.protocol:
        raise ValueError("Aggregation --protocol does not match the result records")
    if args.protocol == "v5_osm_assisted":
        validate_registered_v5_matrix_records(records, args.family)
    report = {
        "schema_version": 1,
        "family": args.family,
        "registry": str(args.registry.resolve()),
        "preliminary": bool(args.allow_preliminary),
        "primary_matrix": primary_matrix,
        "report_kind": args.report_kind,
        **evidence,
        "record_count": len(records),
        "registry_sha256": registered.sha256_file(args.registry),
        "registry_anchor": anchor,
        "selected_results": [
            {
                "result_id": record["result_id"],
                "artifact_sha256": record["artifact_sha256"],
                "registry_entry_sha256": record["registry_entry_sha256"],
            }
            for record in records
        ],
        "summary": summarize_records(
            records,
            tasks=tasks,
            shots=shots,
            folds=folds,
            seeds=seeds,
        ),
    }
    report["selected_results_sha256"] = registered._canonical_sha256(
        {"selected_results": report["selected_results"]}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output.with_suffix(".csv"), report)
    _write_markdown(args.output.with_suffix(".md"), report)


if __name__ == "__main__":
    main()
