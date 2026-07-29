#!/usr/bin/env python3
"""Audit and summarize only one sealed Harbin PU+Query result archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = (("f1", "f1"), ("ap", "ap"), ("auc", "auc_roc"))
TASK_LABELS = {"building": "Building", "road": "Road", "water": "Water"}
FAMILY_LABELS = {
    "p10c_harbin_scratch": "Harbin scratch P10C",
    "p10c_haidian_frozen_harbin": "Frozen Haidian P10C",
    "aef_annual_2025": "AEF annual 2025",
}
QUERY_STYLES = {"disabled": ("#355C7D", "Base"), "adaptive": ("#C06C84", "Query")}
PROTOTYPE_STYLES = {"single": "-", "max": "--"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def cell_key(row: dict[str, Any]) -> tuple[str, str, int, int, int, str, str]:
    return (
        str(row["family"]),
        str(row["task"]),
        int(row["fold"]),
        int(row["seed"]),
        int(row["polygon_count"]),
        str(row["prototype_mode"]),
        str(row["query_mode"]),
    )


def expected_keys(
    config: dict[str, Any], matrix: dict[str, Any]
) -> set[tuple[str, str, int, int, int, str, str]]:
    return {
        (family["id"], task, fold, seed, polygon_count, prototype_mode, query_mode)
        for family in matrix["families"]
        for task in matrix["tasks"]
        for fold in matrix["folds"]
        for seed in matrix["seeds"]
        for polygon_count in config["polygon_counts"]
        for prototype_mode in config["prototype_modes"]
        for query_mode in config["query_modes"]
    }


def verify_schedule_lock(result_root: Path, lock: dict[str, Any]) -> int:
    locked = lock.get("polygon_schedule_sha256")
    if locked is None:
        return 0
    if not isinstance(locked, dict):
        raise ValueError("protocol lock polygon_schedule_sha256 is invalid")
    observed = {
        str(path.relative_to(result_root)): sha256_file(path)
        for path in sorted((result_root / "frozen_polygon_schedules").glob("*.json"))
    }
    if observed != locked:
        raise ValueError("frozen polygon schedule lock differs")
    return len(observed)


def load_schedule_sets(result_root: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    schedules: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for path in sorted((result_root / "frozen_polygon_schedules").glob("*.json")):
        payload = read_json(path)
        for count, schedule in payload.get("sets", {}).items():
            key = (str(payload["task"]), int(payload["fold"]), int(payload["seed"]), int(count))
            if key in schedules:
                raise ValueError(f"duplicate frozen schedule: {key}")
            schedules[key] = schedule
    return schedules


def validate_split_isolation(
    row: dict[str, Any],
    split: dict[str, Any],
    schedules: dict[tuple[str, int, int, int], dict[str, Any]],
) -> None:
    fold = int(row["fold"])
    fold_payload = split["folds"][fold]
    train, validation, test, buffer = (
        set(fold_payload["train"]),
        set(fold_payload["val"]),
        set(fold_payload["test"]),
        set(fold_payload["buffer"]),
    )
    if train & validation or train & test or validation & test:
        raise ValueError(f"fold {fold} train/validation/test partitions overlap")
    if (train | validation | test) & buffer:
        raise ValueError(f"fold {fold} buffer overlaps scored partitions")
    support = row["support_schedule"]
    support_ids = {item["patch_id"] for item in support["support_polygons"]}
    if not support_ids <= train:
        raise ValueError(f"support polygons escape locked training split for fold {fold}")
    if set(row["test_patch_ids"]) != test:
        raise ValueError(f"test patch IDs differ from locked test split for fold {fold}")
    schedule_key = (str(row["task"]), fold, int(row["seed"]), int(row["polygon_count"]))
    expected_schedule = schedules.get(schedule_key)
    if expected_schedule is not None and canonical_sha256(support) != canonical_sha256(
        expected_schedule
    ):
        raise ValueError(f"result support schedule differs from frozen schedule: {schedule_key}")
    minimum_background = support.get("eligibility_audit", {}).get(
        "minimum_reliable_background_pixels"
    )
    if minimum_background is not None and any(
        int(item.get("reliable_background_pixels", 0)) < int(minimum_background)
        for item in support["support_polygons"]
    ):
        raise ValueError(
            f"support schedule violates reliable background eligibility: {schedule_key}"
        )


def validate_result_rows(
    result_root: Path,
    config: dict[str, Any],
    matrix: dict[str, Any],
    lock: dict[str, Any],
) -> list[dict[str, Any]]:
    split_ref = matrix["spatial_split"]
    split_path = resolve_path(split_ref["path"])
    if sha256_file(split_path) != split_ref["sha256"]:
        raise ValueError("spatial split lock differs from the base matrix")
    split = read_json(split_path)
    schedules = load_schedule_sets(result_root)
    expected = expected_keys(config, matrix)
    paths = sorted((result_root / "results").glob("**/result.json"))
    rows: list[dict[str, Any]] = []
    observed: dict[tuple[str, str, int, int, int, str, str], Path] = {}
    lock_path = result_root / "protocol_input_lock.json"
    lock_sha256 = sha256_file(lock_path)
    for path in paths:
        row = read_json(path)
        key = cell_key(row)
        if key in observed:
            raise ValueError(f"duplicate cell result: {key}")
        observed[key] = path
        if row.get("protocol_id") != config["protocol_id"]:
            raise ValueError(f"protocol ID differs: {path}")
        if Path(row.get("protocol_input_lock", "")).resolve() != lock_path.resolve():
            raise ValueError(f"cell uses a different protocol lock: {path}")
        if row.get("protocol_input_lock_sha256") != lock_sha256:
            raise ValueError(f"cell protocol lock hash differs: {path}")
        metrics = row.get("test_metrics")
        if not isinstance(metrics, dict) or any(
            not np.isfinite(float(metrics[name])) or not 0.0 <= float(metrics[name]) <= 1.0
            for _, name in METRICS
            if name in metrics
        ):
            raise ValueError(f"invalid test metrics: {path}")
        if any(name not in metrics for _, name in METRICS):
            raise ValueError(f"missing F1/AP/AUC metric: {path}")
        validate_split_isolation(row, split, schedules)
        rows.append({**row, "result_path": str(path)})
    missing = expected - set(observed)
    unexpected = set(observed) - expected
    if missing:
        raise ValueError(f"missing paired cell(s): {sorted(missing)[:3]}")
    if unexpected:
        raise ValueError(f"unexpected result cell(s): {sorted(unexpected)[:3]}")
    if len(rows) != int(lock["expected_cells"]):
        raise ValueError("result count differs from protocol lock")
    return rows


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["family"],
                row["task"],
                int(row["polygon_count"]),
                row["prototype_mode"],
                row["query_mode"],
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        row: dict[str, Any] = dict(
            zip(("family", "task", "polygon_count", "prototype_mode", "query_mode"), key)
        )
        row["n"] = len(group)
        for output_name, metric_name in METRICS:
            values = np.asarray([item["test_metrics"][metric_name] for item in group], dtype=float)
            row[f"{output_name}_mean"] = float(values.mean())
            row[f"{output_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def query_margins(
    rows: Iterable[dict[str, Any]], query_modes: Iterable[str]
) -> list[dict[str, Any]]:
    if set(query_modes) != {"disabled", "adaptive"}:
        return []
    paired: dict[tuple[str, str, int, int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[
            (
                row["family"],
                row["task"],
                int(row["fold"]),
                int(row["seed"]),
                int(row["polygon_count"]),
                row["prototype_mode"],
            )
        ][row["query_mode"]] = row
    grouped: dict[tuple[str, str, int, str], list[dict[str, float]]] = defaultdict(list)
    for key, modes in paired.items():
        if set(modes) != {"disabled", "adaptive"}:
            raise ValueError(f"missing Query pair: {key}")
        grouped[(key[0], key[1], key[4], key[5])].append(
            {
                output_name: float(modes["adaptive"]["test_metrics"][metric_name])
                - float(modes["disabled"]["test_metrics"][metric_name])
                for output_name, metric_name in METRICS
            }
        )
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        row: dict[str, Any] = dict(zip(("family", "task", "polygon_count", "prototype_mode"), key))
        row["n"] = len(values)
        for metric, _ in METRICS:
            delta = np.asarray([item[metric] for item in values], dtype=float)
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_delta_std"] = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_results(
    result_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    excluded_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Validate a sealed root, then write only its cell and grouped summaries."""
    result_root = result_root.resolve()
    config = read_json(config_path)
    lock_path = result_root / "protocol_input_lock.json"
    lock = read_json(lock_path)
    if lock.get("protocol_id") != config.get("protocol_id"):
        raise ValueError("protocol lock ID differs from config")
    if lock.get("config_sha256") not in (None, canonical_sha256(config)):
        raise ValueError("protocol lock config hash differs")
    matrix_ref = config["base_matrix"]
    matrix_path = resolve_path(matrix_ref["path"])
    if sha256_file(matrix_path) != matrix_ref["sha256"]:
        raise ValueError("base matrix lock differs from config")
    matrix = read_json(matrix_path)
    expected = expected_keys(config, matrix)
    if int(lock["expected_cells"]) != len(expected):
        raise ValueError("protocol lock expected cell count differs from protocol grid")
    schedule_count = verify_schedule_lock(result_root, lock)
    rows = validate_result_rows(result_root, config, matrix, lock)
    summary_rows = aggregate_rows(rows)
    delta_rows = query_margins(rows, config["query_modes"])
    excluded = [str(Path(path).resolve()) for path in excluded_roots]
    audit = {
        "results_root": str(result_root),
        "protocol_id": config["protocol_id"],
        "expected_cell_count": len(expected),
        "result_cell_count": len(rows),
        "frozen_schedule_count": schedule_count,
        "split_isolation_verified_cells": len(rows),
        "shared_three_family_schedule_groups": len(
            {(row["task"], row["fold"], row["seed"], row["polygon_count"]) for row in rows}
        ),
        "excluded_roots": excluded,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cells.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    if delta_rows:
        write_csv(output_dir / "query_margins.csv", delta_rows)
    payload = {"audit": audit, "summary_rows": summary_rows, "query_margin_rows": delta_rows}
    (output_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def plot_f1(summary_rows: list[dict[str, Any]], matrix: dict[str, Any], output: Path) -> None:
    lookup = {
        (
            row["family"],
            row["task"],
            row["polygon_count"],
            row["prototype_mode"],
            row["query_mode"],
        ): row
        for row in summary_rows
    }
    families = [family["id"] for family in matrix["families"]]
    tasks = list(matrix["tasks"])
    counts = sorted({int(row["polygon_count"]) for row in summary_rows})
    figure, axes = plt.subplots(
        len(tasks), len(families), figsize=(14, 9), sharex=True, sharey=True
    )
    for row_index, task in enumerate(tasks):
        for column_index, family in enumerate(families):
            axis = axes[row_index, column_index]
            for prototype, style in PROTOTYPE_STYLES.items():
                for query, (color, label) in QUERY_STYLES.items():
                    values = [
                        lookup[(family, task, count, prototype, query)]["f1_mean"]
                        for count in counts
                    ]
                    axis.plot(
                        counts,
                        values,
                        color=color,
                        linestyle=style,
                        marker="o",
                        label=f"{label}, {prototype}",
                    )
            if row_index == 0:
                axis.set_title(FAMILY_LABELS.get(family, family), fontsize=10, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{TASK_LABELS.get(task, task)}\nmean test F1")
            if row_index == len(tasks) - 1:
                axis.set_xlabel("Support polygons")
            axis.set_xticks(counts)
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        "Harbin strict PU+Query: test F1 across locked polygon budgets", fontweight="bold"
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_markdown(payload: dict[str, Any], matrix: dict[str, Any], asset_name: str) -> str:
    summary = payload["summary_rows"]
    margins = payload["query_margin_rows"]
    audit = payload["audit"]
    reference = [
        row for row in summary if row["polygon_count"] == 3 and row["prototype_mode"] == "max"
    ]
    audit_count = f"{audit['result_cell_count']}/{audit['expected_cell_count']}"
    schedule_count = audit["frozen_schedule_count"]
    isolation_count = audit["split_isolation_verified_cells"]
    figure_path = f"../../production/assets/{asset_name}/harbin_pu_query_transfer_f1.png"
    lines = [
        "# Harbin PU+Query transfer: strict paired audit",
        "",
        "## Audit result",
        "",
        f"- Audited exactly **{audit_count}** locked cells from one v3 result root.",
        (
            f"- Verified all {schedule_count} frozen schedules, every cell protocol lock, and "
            f"train/validation/test isolation for all {isolation_count} cells."
        ),
        (
            "- The prior v1/v2 roots and their logs are explicitly excluded; no parent directory "
            "scan or legacy result file is used."
        ),
        (
            "- Every aggregate is mean ± sample standard deviation across the locked 5 spatial "
            "folds × 3 seeds (n=15)."
        ),
        "",
        f"![Mean F1 by budget, prototype mode, and Query arm]({figure_path})",
        "",
        "## Reference readout: 3 polygons, max prototype",
        "",
        "| Family | Task | Query | F1 | AP | AUC | n |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        reference, key=lambda item: (item["task"], item["family"], item["query_mode"])
    ):
        family = FAMILY_LABELS.get(row["family"], row["family"])
        task = TASK_LABELS.get(row["task"], row["task"])
        lines.append(
            f"| {family} | {task} | "
            f"{row['query_mode']} | {row['f1_mean']:.3f} ± {row['f1_std']:.3f} | "
            f"{row['ap_mean']:.3f} ± {row['ap_std']:.3f} | "
            f"{row['auc_mean']:.3f} ± {row['auc_std']:.3f} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Marginal effect of Query",
            "",
            (
                "The following paired differences are `adaptive − disabled`, averaged over each "
                "locked fold/seed pair. Positive values favor Query; they do not compare different "
                "schedules or thresholds."
            ),
            "",
            "| Family | Task | Polygons | Prototype | ΔF1 | ΔAP | ΔAUC | n |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in margins:
        family = FAMILY_LABELS.get(row["family"], row["family"])
        task = TASK_LABELS.get(row["task"], row["task"])
        lines.append(
            f"| {family} | {task} | {row['polygon_count']} | {row['prototype_mode']} | "
            f"{row['f1_delta_mean']:+.3f} ± {row['f1_delta_std']:.3f} | "
            f"{row['ap_delta_mean']:+.3f} ± {row['ap_delta_std']:.3f} | "
            f"{row['auc_delta_mean']:+.3f} ± {row['auc_delta_std']:.3f} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Scope and interpretation boundary",
            "",
            (
                "This is a polygon-prompt PU+Query readout, not the patch-shot Conv3x3 benchmark. "
                "The three families share the same 380-patch universe, spatial folds, and frozen "
                "polygon schedules, seeds, and validation-only threshold. It therefore supports "
                "paired "
                "readout comparisons only; it must not be merged with the separate Conv3x3 table."
            ),
            "",
            (
                "The complete 144-row family/task/budget/prototype/Query aggregate is provided in "
                "the companion CSV; the detailed cell and audit records remain under the sealed v3 "
                "data root."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--excluded-root", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = audit_results(
        args.result_root,
        args.config,
        args.output_dir,
        excluded_roots=args.excluded_root,
    )
    matrix = read_json(resolve_path(read_json(args.config)["base_matrix"]["path"]))
    write_csv(args.summary_csv, payload["summary_rows"])
    plot_f1(payload["summary_rows"], matrix, args.assets / "harbin_pu_query_transfer_f1.png")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(payload, matrix, args.assets.name), encoding="utf-8")
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
