#!/usr/bin/env python3
"""Fail-closed audit and aggregate for the Harbin frozen-transfer multihead matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.eval import run_harbin_frozen_transfer_multihead as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = (("f1", "f1_at_threshold"), ("ap", "ap"), ("auc", "auc_roc"))
FAMILY_LABELS = {
    "p10c_haidian_frozen_harbin": "Frozen Haidian P10C",
    "aef_annual_2025": "AEF annual 2025",
}
TASK_LABELS = {"building": "Building", "road": "Road", "water": "Water"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def expected_cells(config: dict[str, Any]) -> set[tuple[str, str, int, int, int, str]]:
    return {
        (family["id"], task, fold, shot, seed, head)
        for family in config["primary_families"]
        for task in config["tasks"]
        for fold in config["folds"]
        for shot in config["shots"]
        for seed in config["seeds"]
        for head in config["heads"]
    }


def reject_scratch_contamination(results_root: Path) -> None:
    for path in (results_root / "results").glob("**/metrics.json"):
        if "scratch" in path.relative_to(results_root / "results").parts:
            raise ValueError(f"scratch family contamination in primary root: {path}")
        try:
            payload = read_json(path)
        except ValueError:
            continue
        if "scratch" in str(payload.get("family", "")):
            raise ValueError(f"scratch family contamination in primary root: {path}")


def read_prediction_archive(path: Path, expected_ids: list[str]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        expected_fields = {"patch_ids", "probability_maps", "target_maps"}
        if set(archive.files) != expected_fields:
            raise ValueError(f"prediction archive fields differ from atomic contract: {path}")
        patch_ids = [str(value) for value in archive["patch_ids"].tolist()]
        probabilities = archive["probability_maps"]
        targets = archive["target_maps"]
    shape = (len(expected_ids), 128, 128)
    if patch_ids != expected_ids or probabilities.shape != shape or targets.shape != shape:
        raise ValueError(f"prediction archive differs from locked split: {path}")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(f"prediction probabilities are invalid: {path}")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError(f"prediction targets are not binary: {path}")


def validate_result(
    metrics_path: Path,
    *,
    results_root: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    lock_path: Path,
) -> dict[str, Any]:
    payload = read_json(metrics_path)
    parts = metrics_path.relative_to(results_root / "results").parts
    if len(parts) != 7 or parts[-1] != "metrics.json":
        raise ValueError(f"unexpected primary result path: {metrics_path}")
    family, task, fold_part, shot_part, seed_part, head = parts[:6]
    if not (
        fold_part.startswith("fold")
        and shot_part.startswith("shot")
        and seed_part.startswith("seed")
    ):
        raise ValueError(f"unexpected primary result coordinates: {metrics_path}")
    fold, shot, seed = int(fold_part[4:]), int(shot_part[4:]), int(seed_part[4:])
    cell = (family, task, fold, shot, seed, head)
    if cell not in expected_cells(config):
        raise ValueError(f"result is outside the primary matrix: {cell}")
    if family not in FAMILY_LABELS or "scratch" in family:
        raise ValueError(f"scratch or unknown family in primary result: {metrics_path}")
    for field, expected in (
        ("family", family),
        ("task", task),
        ("fold", fold),
        ("shot", shot),
        ("seed", seed),
        ("head", head),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"result metadata/path mismatch for {field}: {metrics_path}")
    if payload.get("protocol_id") != config["protocol_id"]:
        raise ValueError(f"result protocol differs from primary contract: {metrics_path}")
    if Path(str(payload.get("matrix_input_lock", ""))).resolve() != lock_path.resolve():
        raise ValueError(f"result uses a different matrix lock: {metrics_path}")
    if payload.get("matrix_input_lock_sha256") != sha256_file(lock_path):
        raise ValueError(f"result matrix lock hash differs: {metrics_path}")
    reader = payload.get("reader")
    if not isinstance(reader, dict) or reader != {"head": head, **config["reader"]}:
        raise ValueError(f"reader architecture or hyperparameters differ: {metrics_path}")
    schedule_path = results_root / "frozen_shot_schedules" / f"{task}_fold{fold}_seed{seed}.json"
    if Path(str(payload.get("shot_schedule", ""))).resolve() != schedule_path.resolve():
        raise ValueError(f"result uses a different frozen schedule: {metrics_path}")
    if payload.get("shot_schedule_sha256") != sha256_file(schedule_path):
        raise ValueError(f"result schedule hash differs: {metrics_path}")
    for _, metric_name in METRICS:
        value = float(payload.get(metric_name, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid metric {metric_name}: {metrics_path}")
    for filename in ("predictions_validation.npz", "predictions_test.npz", "final_reader.pt"):
        artifact = metrics_path.parent / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"non-atomic result bundle: {metrics_path.parent}")
    read_prediction_archive(
        metrics_path.parent / "predictions_validation.npz", sorted(split["folds"][fold]["val"])
    )
    read_prediction_archive(
        metrics_path.parent / "predictions_test.npz", sorted(split["folds"][fold]["test"])
    )
    row = {"family": family, "task": task, "fold": fold, "shot": shot, "seed": seed, "head": head}
    row.update({short: float(payload[metric]) for short, metric in METRICS})
    row["metrics_path"] = str(metrics_path.resolve())
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["task"], int(row["shot"]), row["head"])].append(row)
    summary: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        result = dict(zip(("family", "task", "shot", "head"), key))
        result["n"] = len(group)
        for metric, _ in METRICS:
            values = [float(item[metric]) for item in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_multihead_results(
    results_root: Path, config_path: Path, audit_root: Path
) -> dict[str, Any]:
    """Validate every paired cell, then write per-cell and mean±std aggregates."""
    results_root = results_root.resolve()
    reject_scratch_contamination(results_root)
    config = read_json(config_path)
    if tuple(family["id"] for family in config.get("primary_families", [])) != tuple(FAMILY_LABELS):
        raise ValueError("primary configuration family contract is invalid")
    lock_path = results_root / "matrix_input_lock.json"
    lock = read_json(lock_path)
    if (
        lock.get("protocol_id") != config["protocol_id"]
        or lock.get("config_sha256") != runner.canonical_sha256(config)
        or lock.get("primary_family_ids") != list(FAMILY_LABELS)
    ):
        raise ValueError("primary matrix input lock differs from configuration")
    matrix = read_json(Path(lock["base_matrix_path"]))
    split_path = runner.strict.repo_path(matrix["spatial_split"]["path"])
    if sha256_file(split_path) != matrix["spatial_split"]["sha256"]:
        raise ValueError("locked spatial split differs from base matrix")
    split = read_json(split_path)
    paths = sorted((results_root / "results").glob("**/metrics.json"))
    rows = [
        validate_result(
            path, results_root=results_root, config=config, split=split, lock_path=lock_path
        )
        for path in paths
    ]
    observed = {
        (row["family"], row["task"], row["fold"], row["shot"], row["seed"], row["head"])
        for row in rows
    }
    expected = expected_cells(config)
    if expected - observed:
        raise ValueError(f"missing paired cell(s): {sorted(expected - observed)[:3]}")
    if observed - expected or len(rows) != len(observed):
        raise ValueError("unexpected or duplicate primary result cell")
    if len(rows) != int(lock["job_count"]):
        raise ValueError("completed cell count differs from input lock")
    summary = aggregate_rows(rows)
    audit = {
        "protocol_id": config["protocol_id"],
        "primary_families": list(FAMILY_LABELS),
        "completed_cells": len(rows),
        "expected_cells": len(expected),
        "aggregate_rows": len(summary),
        "n_per_aggregate": sorted({row["n"] for row in summary}),
        "scratch_excluded": True,
        "atomic_bundles_verified": len(rows),
        "validation_and_test_archives_verified": len(rows),
    }
    audit_root.mkdir(parents=True, exist_ok=True)
    write_csv(audit_root / "harbin_frozen_transfer_multihead_cells.csv", rows)
    write_csv(audit_root / "harbin_frozen_transfer_multihead_summary.csv", summary)
    (audit_root / "harbin_frozen_transfer_multihead_audit.json").write_text(
        json.dumps({"audit": audit, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    return {"audit": audit, "summary": summary}


def plot_f1(summary: list[dict[str, Any]], output: Path) -> None:
    lookup = {(row["family"], row["task"], row["shot"], row["head"]): row for row in summary}
    heads = list(runner.MATCHED_HEADS)
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharey=True)
    x = np.arange(len(heads))
    for task_index, task in enumerate(("building", "road", "water")):
        for shot_index, shot in enumerate((5, 10)):
            axis = axes[task_index, shot_index]
            for offset, family in zip((-0.18, 0.18), FAMILY_LABELS, strict=True):
                values = [lookup[(family, task, shot, head)]["f1_mean"] for head in heads]
                errors = [lookup[(family, task, shot, head)]["f1_std"] for head in heads]
                axis.bar(
                    x + offset, values, 0.36, yerr=errors, capsize=3, label=FAMILY_LABELS[family]
                )
            axis.set_title(f"{TASK_LABELS[task]} · {shot}-shot")
            axis.set_xticks(x, heads, rotation=28, ha="right", fontsize=8)
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            if shot_index == 0:
                axis.set_ylabel("Test F1 (mean ± sample std; n=15)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Harbin frozen-transfer multihead evaluation", fontweight="bold")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_markdown(
    summary: list[dict[str, Any]], audit: dict[str, Any], figure_relative: str
) -> str:
    completed = f"{audit['completed_cells']}/{audit['expected_cells']}"
    bundles = audit["atomic_bundles_verified"]
    lines = [
        "# Harbin frozen-transfer multihead evaluation",
        "",
        "## Audit status",
        "",
        f"- **Passed:** {completed} paired cells, with {bundles} complete atomic bundles.",
        (
            "- The primary table contains only frozen Haidian P10C and AEF annual-2025; "
            "Harbin scratch is rejected and excluded."
        ),
        (
            "- Thresholds are selected from the locked validation split; prediction archives are "
            "checked against locked validation/test patch IDs."
        ),
        "",
        f"![Per-head test F1]({figure_relative})",
        "",
        "## Per-head test metrics",
        "",
        (
            "Each value is mean ± sample standard deviation over the locked 5 spatial folds × 3 "
            "seeds (n=15)."
        ),
        "",
        "| Family | Task | Shot | Head | F1 | AP | AUC | n |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        family = FAMILY_LABELS[row["family"]]
        task = TASK_LABELS[row["task"]]
        lines.append(
            f"| {family} | {task} | {row['shot']} | "
            f"{row['head']} | {row['f1_mean']:.3f} ± {row['f1_std']:.3f} | "
            f"{row['ap_mean']:.3f} ± {row['ap_std']:.3f} | "
            f"{row['auc_mean']:.3f} ± {row['auc_std']:.3f} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Modality-masking diagnostic status",
            "",
            (
                "No high-resolution-masked result is included in this primary table. The separate "
                "P10C-only diagnostic is valid only after a newly exported and sealed embedding "
                "set verifies that both high-resolution optical and high-resolution SAR "
                "availability are disabled. Unmasked embeddings are not treated as masked evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_multihead_results(args.results_root, args.config, args.audit_root)
    write_csv(args.summary_csv, payload["summary"])
    figure = args.assets / "harbin_frozen_transfer_multihead_f1.png"
    plot_f1(payload["summary"], figure)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            payload["summary"],
            payload["audit"],
            f"../../production/assets/{args.assets.name}/{figure.name}",
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
