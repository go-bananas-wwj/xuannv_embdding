#!/usr/bin/env python3
"""Fail-closed audit and paper aggregation for the sealed Harbin Conv3x3 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("f1_at_threshold", "ap", "auc_roc", "threshold")
FAMILY_LABELS = {
    "p10c_harbin_scratch": "Harbin scratch P10C",
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


def expected_cells(matrix: dict[str, Any]) -> set[tuple[str, str, int, int, int]]:
    return {
        (family["id"], task, fold, shot, seed)
        for family in matrix["families"]
        for task in matrix["tasks"]
        for fold in matrix["folds"]
        for shot in matrix["shots"]
        for seed in matrix["seeds"]
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["task"], int(row["shot"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (family, task, shot), group in sorted(grouped.items()):
        summary: dict[str, Any] = {"family": family, "task": task, "shot": shot, "n": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            short = "f1" if metric == "f1_at_threshold" else metric
            summary[f"{short}_mean"] = statistics.mean(values)
            summary[f"{short}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def read_archive(
    path: Path,
    expected_patch_ids: list[str],
    target_hashes: dict[str, str] | None,
) -> None:
    """Verify a lossless archive is finite, binary, and binds the frozen test IDs."""
    with np.load(path, allow_pickle=True) as archive:
        required = {"patch_ids", "probabilities", "targets", "valid_masks"}
        if set(archive.files) != required:
            raise ValueError(f"prediction archive has unexpected fields: {path}")
        patch_ids = [str(value) for value in archive["patch_ids"].tolist()]
        probabilities = archive["probabilities"]
        targets = archive["targets"]
        valid_masks = archive["valid_masks"]
    if patch_ids != expected_patch_ids:
        raise ValueError(f"prediction archive test IDs differ from frozen split: {path}")
    expected_shape = (len(expected_patch_ids), 128, 128)
    if probabilities.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError(f"prediction archive has unexpected map shape: {path}")
    if (
        valid_masks.shape != expected_shape
        or valid_masks.dtype != np.bool_
        or not valid_masks.all()
    ):
        raise ValueError(f"prediction archive valid mask is incomplete: {path}")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(f"prediction archive probabilities are invalid: {path}")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError(f"prediction archive targets are not binary: {path}")
    for index, patch_id in enumerate(patch_ids):
        if target_hashes is None:
            continue
        target = np.ascontiguousarray(targets[index].astype(np.uint8, copy=False))
        header = f"target-support-v1|{target.dtype.str}|128,128|".encode("utf-8")
        actual = hashlib.sha256(header + target.tobytes()).hexdigest()
        if actual != target_hashes.get(patch_id):
            raise ValueError(f"prediction archive target hash differs for {patch_id}: {path}")


def validate_result(
    metrics_path: Path,
    *,
    root: Path,
    matrix: dict[str, Any],
    lock: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    relative = metrics_path.relative_to(root / "results")
    parts = relative.parts
    if len(parts) != 6 or parts[-1] != "metrics.json":
        raise ValueError(f"unexpected result path: {metrics_path}")
    family, task, fold_part, shot_part, seed_part = parts[:5]
    if (
        not fold_part.startswith("fold")
        or not shot_part.startswith("shot")
        or not seed_part.startswith("seed")
    ):
        raise ValueError(f"unexpected result coordinates: {metrics_path}")
    fold, shot, seed = int(fold_part[4:]), int(shot_part[4:]), int(seed_part[4:])
    cell = (family, task, fold, shot, seed)
    if cell not in expected_cells(matrix):
        raise ValueError(f"result is outside the locked matrix: {cell}")
    for field, expected in (
        ("family", family),
        ("task", task),
        ("fold", fold),
        ("shot", shot),
        ("seed", seed),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"result metadata/path mismatch for {field}: {metrics_path}")
    if payload.get("protocol_id") != matrix["protocol_id"]:
        raise ValueError(f"result protocol differs from locked matrix: {metrics_path}")
    if payload.get("matrix_input_lock_sha256") != sha256_file(root / "matrix_input_lock.json"):
        raise ValueError(f"result does not bind the matrix input lock: {metrics_path}")
    expected_schedule = root / "frozen_shot_schedules" / f"{task}_fold{fold}_seed{seed}.json"
    if Path(str(payload.get("shot_schedule", ""))).resolve() != expected_schedule.resolve():
        raise ValueError(f"result uses a different shot schedule: {metrics_path}")
    if payload.get("shot_schedule_sha256") != sha256_file(expected_schedule):
        raise ValueError(f"result shot schedule hash differs: {metrics_path}")
    for metric in METRICS:
        value = float(payload.get(metric, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {metric} in {metrics_path}")
    if payload.get("probe", {}).get("head") != matrix["probe"]["head"]:
        raise ValueError(f"result uses a different downstream head: {metrics_path}")
    for key in ("epochs", "batch_size", "lr", "weight_decay"):
        if payload.get("probe", {}).get(key) != matrix["probe"][key]:
            raise ValueError(f"result probe {key} differs from locked matrix: {metrics_path}")
    expected_test_ids = sorted(split["folds"][fold]["test"])
    confusion = payload.get("per_patch_confusion")
    target_hashes = payload.get("per_patch_target_support_sha256")
    if not isinstance(confusion, dict) or not isinstance(target_hashes, dict):
        raise ValueError(f"result lacks per-patch audit evidence: {metrics_path}")
    if sorted(confusion) != expected_test_ids or sorted(target_hashes) != expected_test_ids:
        raise ValueError(f"result test support differs from frozen split: {metrics_path}")
    totals = {
        key: sum(int(value[key]) for value in confusion.values())
        for key in ("tp", "fp", "fn", "tn")
    }
    if any(totals[key] != int(payload[key]) for key in totals):
        raise ValueError(f"per-patch confusion does not reconstruct metrics totals: {metrics_path}")
    for filename in ("predictions_test.npz", "predictions_validation.npz", "final_probe.pt"):
        path = metrics_path.parent / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing result artifact: {path}")
    read_archive(metrics_path.parent / "predictions_test.npz", expected_test_ids, target_hashes)
    expected_val_ids = sorted(split["folds"][fold]["val"])
    read_archive(metrics_path.parent / "predictions_validation.npz", expected_val_ids, None)
    row = {"family": family, "task": task, "fold": fold, "shot": shot, "seed": seed}
    row.update({metric: float(payload[metric]) for metric in METRICS})
    row["miou"] = float(payload["miou"])
    row["precision"] = float(payload["precision"])
    row["recall"] = float(payload["recall"])
    row["train_seconds"] = float(payload["train_seconds"])
    row["metrics_path"] = str(metrics_path.resolve())
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_f1(summary: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lookup = {(row["family"], row["task"], row["shot"]): row for row in summary}
    families = list(FAMILY_LABELS)
    colors = ["#0072B2", "#CC79A7", "#D55E00"]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.2), sharey=True)
    for task_index, task in enumerate(("building", "road", "water")):
        for shot_index, shot in enumerate((5, 10)):
            axis = axes[task_index, shot_index]
            values = [lookup[(family, task, shot)]["f1_mean"] for family in families]
            errors = [lookup[(family, task, shot)]["f1_std"] for family in families]
            bars = axis.bar(
                range(3), values, yerr=errors, capsize=4, color=colors, edgecolor="#303030"
            )
            axis.set_title(f"{TASK_LABELS[task]} · {shot}-shot")
            axis.set_xticks(range(3), ["Scratch\nP10C", "Frozen\nHaidian P10C", "AEF\n2025"])
            axis.set_ylim(0, 1.0)
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
            axis.set_axisbelow(True)
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            if shot_index == 0:
                axis.set_ylabel("Test F1 (mean ± sample std; n=15)")
    fig.suptitle("Harbin strict spatial 5-fold Conv3x3 evaluation", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(
    summary: list[dict[str, Any]], audit: dict[str, Any], figure_relative: str
) -> str:
    lines = [
        "# 哈尔滨严格 Conv3x3 三方矩阵结果",
        "",
        "## 技术摘要",
        "",
        "本报告仅汇总锁定的 380-patch Harbin 协议：三种 embedding family、building/road/water、",
        "5 个空间 fold、3 个 seed 和 5/10-shot 的 Conv3x3 readout。每个 family/task/shot 的",
        "均值与样本标准差均基于 15 个独立的 fold×seed 单元；",
        "阈值仅从同 fold 的 validation split 选择。",
        "",
        f"审计状态：**通过**（{audit['completed_cells']}/{audit['expected_cells']} cells；",
        "无缺失、重复或协议偏离）。",
        "",
        "## F1 结果",
        "",
        f"![Harbin strict Conv3x3 F1]({figure_relative})",
        "",
        "图中误差线为跨 5 folds × 3 seeds 的样本标准差；图用于比较 representation，",
        "不能说明跨年份输入之间的因果效应。",
        "",
        "| Task | Shot | Harbin scratch P10C | Frozen Haidian P10C | AEF annual 2025 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lookup = {(row["family"], row["task"], row["shot"]): row for row in summary}
    for task in ("building", "road", "water"):
        for shot in (5, 10):
            cells = [lookup[(family, task, shot)] for family in FAMILY_LABELS]
            rendered = [f"{cell['f1_mean']:.3f} ± {cell['f1_std']:.3f}" for cell in cells]
            lines.append(f"| {TASK_LABELS[task]} | {shot} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## 完整指标",
            "",
            "| Family | Task | Shot | F1 | AP | AUC-ROC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            "| "
            f"{FAMILY_LABELS[row['family']]} | {TASK_LABELS[row['task']]} | {row['shot']} | "
            f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f} | "
            f"{row['ap_mean']:.3f} ± {row['ap_std']:.3f} | "
            f"{row['auc_roc_mean']:.3f} ± {row['auc_roc_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 协议与审计边界",
            "",
            "- 380 个 coverage-locked patch，所有 family 均绑定同一 matrix input lock",
            "  和 45 份冻结 shot schedule。",
            "- 每个结果均重新核验：路径/metadata cell、锁定协议、Conv3x3 超参数、阈值范围、",
            "  40 个 test patch、per-patch confusion 与总量、预测概率/标签 archive、",
            "  validation archive、probe state。",
            "- 报告 F1、AP、AUC 的全量均值±标准差在数据盘的 aggregate CSV；该比较仍应表述为",
            "  严格 paired spatial readout，AEF annual 2025 与 2026-04 Xuannv embedding",
            "  存在时间上下文差异。",
            "",
            "## 后续",
            "",
            "建议在论文中报告此表和完整 CSV，并在结果叙述中将 Harbin scratch 与",
            "冻结 Haidian P10C 的差异解释为迁移/区域训练条件下的 readout 差异，",
            "而非仅凭该矩阵做因果或时间等价声明。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--figure-relative", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    lock_path = args.result_root / "matrix_input_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("matrix_sha256") != sha256_file(args.matrix):
        raise ValueError("matrix input lock does not bind the requested matrix")
    if lock.get("job_count") != len(expected_cells(matrix)):
        raise ValueError("matrix input lock has an unexpected job count")
    split_path = REPO_ROOT / matrix["spatial_split"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    paths = sorted((args.result_root / "results").glob("*/*/fold*/shot*/seed*/metrics.json"))
    rows = [
        validate_result(path, root=args.result_root, matrix=matrix, lock=lock, split=split)
        for path in paths
    ]
    cells = [(row["family"], row["task"], row["fold"], row["shot"], row["seed"]) for row in rows]
    expected = expected_cells(matrix)
    if len(cells) != len(set(cells)):
        raise ValueError("duplicate matrix cells found")
    if set(cells) != expected:
        raise ValueError(
            "matrix cells differ: "
            f"missing={len(expected - set(cells))} extra={len(set(cells) - expected)}"
        )
    summary = aggregate_rows(rows)
    if any(row["n"] != 15 for row in summary) or len(summary) != 18:
        raise ValueError(
            "aggregate does not contain exactly 15 fold×seed runs per family/task/shot"
        )
    audit = {
        "protocol_id": matrix["protocol_id"],
        "matrix_path": str(args.matrix.resolve()),
        "matrix_sha256": sha256_file(args.matrix),
        "matrix_input_lock": str(lock_path.resolve()),
        "matrix_input_lock_sha256": sha256_file(lock_path),
        "expected_cells": len(expected),
        "completed_cells": len(rows),
        "summary_rows": len(summary),
        "status": "passed",
        "checks": {
            "unique_complete_cells": True,
            "metric_ranges": True,
            "locked_input_bindings": True,
            "frozen_shot_schedules": True,
            "prediction_archives": True,
            "per_patch_confusion": True,
        },
    }
    args.audit_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_root / "harbin_strict_conv3x3_cell_metrics.csv", rows)
    write_csv(args.audit_root / "harbin_strict_conv3x3_aggregate.csv", summary)
    (args.audit_root / "harbin_strict_conv3x3_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    plot_f1(summary, args.figure)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        markdown_report(summary, audit, args.figure_relative), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
