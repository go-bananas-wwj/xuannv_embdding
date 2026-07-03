#!/usr/bin/env python3
"""Visualize two pixel-probe benchmark outputs side by side."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.report.visualize_osm_downstream_outputs import (
    binary_metrics,
    embedding_pca,
    load_embedding,
    load_highres,
    load_prediction,
    load_threshold,
    read_mask,
    red_binary_mask,
    red_probability_map,
    show_panel,
)


TASK_INFO = {
    "haidian_building_osm": {"region": "haidian", "label_task": "building_osm"},
    "haidian_road_osm": {"region": "haidian", "label_task": "road_osm"},
    "haidian_water_osm": {"region": "haidian", "label_task": "osm_water"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-embedding-root", type=Path, required=True)
    parser.add_argument("--candidate-name", default="P7A ours")
    parser.add_argument("--candidate-month", default="202604")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-embedding-root", type=Path, required=True)
    parser.add_argument("--baseline-name", default="AEF official")
    parser.add_argument("--baseline-month", default="202512")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_INFO))
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed"),
    )
    return parser.parse_args()


def normalized_probability(prob: np.ndarray) -> np.ndarray:
    prob = np.nan_to_num(prob.astype(np.float32), nan=0.0)
    lo = float(np.nanmin(prob))
    hi = float(np.nanmax(prob))
    if hi > lo:
        prob = (prob - lo) / (hi - lo)
    else:
        prob = np.zeros_like(prob, dtype=np.float32)
    return red_probability_map(prob)


def resolve_prediction(root: Path, task: str, patch_id: str) -> Path | None:
    matches = sorted((root / task).glob(f"fold_*/predictions/{patch_id}_prob.tif"))
    return matches[0] if matches else None


def collect_existing_patch_ids(root: Path, task: str) -> set[str]:
    out: set[str] = set()
    pattern = re.compile(r"patch_\d{6}")
    for path in (root / task).glob("fold_*/predictions/patch_*_prob.tif"):
        match = pattern.search(path.name)
        if match:
            out.add(match.group(0))
    return out


def select_patch_ids(args: argparse.Namespace, task: str, label_dir: Path) -> list[str]:
    candidate_ids = collect_existing_patch_ids(args.candidate_root, task)
    baseline_ids = collect_existing_patch_ids(args.baseline_root, task)
    common_ids = sorted(candidate_ids & baseline_ids)
    scored: list[dict[str, Any]] = []
    for patch_id in common_ids:
        mask_path = label_dir / f"{patch_id}.tif"
        if not mask_path.exists():
            continue
        gt = read_mask(mask_path) > 0
        gt_pos = int(gt.sum())
        if gt_pos <= 0:
            continue
        pred_path = resolve_prediction(args.candidate_root, task, patch_id)
        if pred_path is None:
            continue
        prob = load_prediction(pred_path)
        threshold = load_threshold(pred_path)
        metrics = binary_metrics(prob >= threshold, gt)
        scored.append(
            {
                "patch_id": patch_id,
                "gt_pos": gt_pos,
                "f1": float(metrics["f1"]),
                "fp": int(metrics["fp"]),
            }
        )
    selected: list[dict[str, Any]] = []

    def add(items: list[dict[str, Any]]) -> None:
        seen = {item["patch_id"] for item in selected}
        for item in items:
            if item["patch_id"] not in seen:
                selected.append(item)
                seen.add(item["patch_id"])
            if len(selected) >= args.samples_per_task:
                return

    add(sorted(scored, key=lambda item: item["gt_pos"], reverse=True))
    add(sorted(scored, key=lambda item: item["f1"]))
    add(sorted(scored, key=lambda item: item["fp"], reverse=True))
    return [item["patch_id"] for item in selected[: args.samples_per_task]]


def load_row(
    benchmark_root: Path,
    embedding_root: Path,
    task: str,
    region: str,
    patch_id: str,
    month: str,
    gt: np.ndarray,
) -> dict[str, Any]:
    pred_path = resolve_prediction(benchmark_root, task, patch_id)
    if pred_path is None:
        raise FileNotFoundError(f"Missing prediction for {task}/{patch_id}: {benchmark_root}")
    prob = load_prediction(pred_path)
    threshold = load_threshold(pred_path)
    pred = (prob >= threshold).astype(np.float32)
    metrics = binary_metrics(pred > 0, gt > 0)
    return {
        "pca": embedding_pca(load_embedding(embedding_root, region, patch_id, month)),
        "prob": prob,
        "threshold": threshold,
        "prediction": pred,
        "metrics": metrics,
        "prediction_path": str(pred_path),
    }


def make_figure(
    args: argparse.Namespace,
    task: str,
    patch_id: str,
    info: dict[str, str],
) -> dict[str, Any]:
    region = info["region"]
    label_task = info["label_task"]
    label_path = (
        args.processed_root / region / "labels" / label_task / "masks" / f"{patch_id}.tif"
    )
    gt = (read_mask(label_path) > 0).astype(np.float32)
    highres = load_highres(args.processed_root, region, patch_id, args.candidate_month)
    candidate = load_row(
        args.candidate_root,
        args.candidate_embedding_root,
        task,
        region,
        patch_id,
        args.candidate_month,
        gt,
    )
    baseline = load_row(
        args.baseline_root,
        args.baseline_embedding_root,
        task,
        region,
        patch_id,
        args.baseline_month,
        gt,
    )
    fig, axes = plt.subplots(2, 5, figsize=(18.5, 7.2))
    for row_idx, (name, row) in enumerate(
        [(args.candidate_name, candidate), (args.baseline_name, baseline)]
    ):
        panels = [
            (highres, f"{name}\nHigh-res {args.candidate_month}"),
            (row["pca"], "Embedding PCA"),
            (normalized_probability(row["prob"]), "Prediction Prob\nper-image norm"),
            (
                red_binary_mask(row["prediction"]),
                f"Prediction Mask\nthr={row['threshold']:.3f}, F1={row['metrics']['f1']:.3f}",
            ),
            (red_binary_mask(gt), "GT"),
        ]
        for col_idx, (image, title) in enumerate(panels):
            show_panel(axes[row_idx, col_idx], image, title)
    fig.suptitle(f"{task} | {patch_id} | top={args.candidate_name}, bottom={args.baseline_name}")
    fig.tight_layout()
    out_dir = args.output_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{patch_id}_pixel_probe_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "task": task,
        "patch_id": patch_id,
        "figure": str(out_path),
        "candidate": {
            "threshold": candidate["threshold"],
            "metrics": candidate["metrics"],
            "prediction": candidate["prediction_path"],
        },
        "baseline": {
            "threshold": baseline["threshold"],
            "metrics": baseline["metrics"],
            "prediction": baseline["prediction_path"],
        },
        "gt": str(label_path),
    }


def write_index(records: list[dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# Pixel Probe Comparison Visualizations",
        "",
        "Top row is the candidate embedding; bottom row is the baseline embedding.",
        "Prediction probability is normalized per image for visual inspection.",
        "",
    ]
    for record in records:
        fig = Path(record["figure"])
        lines.extend(
            [
                f"## {record['task']} / {record['patch_id']}",
                "",
                f"![{fig.name}]({fig.relative_to(output_root).as_posix()})",
                "",
            ]
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        if task not in TASK_INFO:
            raise KeyError(f"Unknown task: {task}")
        info = TASK_INFO[task]
        label_dir = (
            args.processed_root
            / info["region"]
            / "labels"
            / info["label_task"]
            / "masks"
        )
        for patch_id in select_patch_ids(args, task, label_dir):
            records.append(make_figure(args, task, patch_id, info))
    write_index(records, args.output_root)
    print(f"saved {len(records)} figures to {args.output_root}")


if __name__ == "__main__":
    main()
