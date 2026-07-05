#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from scripts.eval.train_aef_downstream_probe import PixelProbe
from scripts.report.visualize_osm_downstream_outputs import (
    binary_metrics,
    embedding_pca,
    load_embedding,
    load_highres,
    red_binary_mask,
    red_probability_map,
    read_mask,
    show_panel,
    stretch,
)


TASKS = {
    "building": {"label_task": "building_osm", "title": "Building"},
    "road": {"label_task": "road_osm", "title": "Road"},
    "water": {"label_task": "osm_water", "title": "Water"},
    "construction": {"label_task": "construction", "title": "Construction"},
    "green": {"label_task": "osm_green", "title": "Green"},
    "residential": {"label_task": "osm_residential", "title": "Residential"},
    "commercial": {"label_task": "osm_commercial", "title": "Commercial"},
    "industrial": {"label_task": "osm_industrial", "title": "Industrial"},
    "agriculture": {"label_task": "osm_agriculture", "title": "Agriculture"},
    "rail": {"label_task": "osm_rail", "title": "Rail"},
    "playground": {"label_task": "osm_playground", "title": "Playground"},
    "path_walk": {"label_task": "osm_path_walk", "title": "Path Walk"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize fair capability-suite pixel probe outputs without retraining."
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--xuannv-embedding-root", type=Path, required=True)
    parser.add_argument(
        "--aef-embedding-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("/data/xuannv_embedding/processed"))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--shot", default="full")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--samples-per-task", type=int, default=6)
    parser.add_argument("--xuannv-month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_split(label_root: Path, fold: int) -> dict[str, Any]:
    split_path = label_root / "split_5fold.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return split["folds"][fold]


def mask_positive_count(label_root: Path, patch_id: str) -> int:
    mask_path = label_root / "masks" / f"{patch_id}.tif"
    if not mask_path.exists():
        return 0
    return int((read_mask(mask_path) > 0).sum())


def select_patch_ids(label_root: Path, fold: int, samples: int) -> list[str]:
    fold_info = load_split(label_root, fold)
    patch_ids = list(fold_info["test"])
    ranked = sorted(
        patch_ids,
        key=lambda patch_id: mask_positive_count(label_root, patch_id),
        reverse=True,
    )
    positives = [patch_id for patch_id in ranked if mask_positive_count(label_root, patch_id) > 0]
    selected = positives[:samples]
    if len(selected) < samples:
        for patch_id in ranked:
            if patch_id not in selected:
                selected.append(patch_id)
            if len(selected) >= samples:
                break
    return selected


def checkpoint_root(
    suite_root: Path,
    model_name: str,
    task: str,
    head: str,
    shot: str,
    fold: int,
) -> Path:
    return suite_root / model_name / task / head / f"shot_{shot}" / f"fold_{fold}" / f"fold_{fold}"


def load_probe(job_root: Path, head: str, device: torch.device) -> tuple[PixelProbe, float, dict[str, Any]]:
    metrics_path = job_root / "metrics.json"
    ckpt_path = job_root / "checkpoints" / "best.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model = PixelProbe(embed_dim=64, head=head, hidden_dim=128).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    threshold = float(metrics.get("val_threshold", metrics.get("threshold", 0.5)))
    return model, threshold, metrics


def predict_probability(
    model: PixelProbe,
    embedding_root: Path,
    region: str,
    patch_id: str,
    month: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    emb = load_embedding(embedding_root, region, patch_id, month)
    if emb is None:
        raise FileNotFoundError(f"Missing embedding for {region}/{patch_id}/{month}: {embedding_root}")
    channels, height, width = emb.shape
    x = emb.float().permute(1, 2, 0).reshape(-1, channels).contiguous()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 262144):
            parts.append(torch.sigmoid(model(x[start : start + 262144].to(device))).detach().cpu())
    prob = torch.cat(parts, dim=0).reshape(height, width).numpy().astype(np.float32)
    pca = embedding_pca(emb)
    if pca is None:
        pca = np.zeros((height, width, 3), dtype=np.float32)
    return prob, pca


def normalized_prob_rgb(prob: np.ndarray) -> np.ndarray:
    return red_probability_map(stretch(prob, lower=0.0, upper=100.0))


def make_figure(
    args: argparse.Namespace,
    task: str,
    patch_id: str,
    xuannv_model: PixelProbe,
    xuannv_threshold: float,
    xuannv_metrics: dict[str, Any],
    aef_model: PixelProbe,
    aef_threshold: float,
    aef_metrics: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    info = TASKS[task]
    region = "haidian"
    label_root = args.processed_root / region / "labels" / info["label_task"]
    gt_path = label_root / "masks" / f"{patch_id}.tif"
    gt = (read_mask(gt_path) > 0).astype(np.float32)
    highres = load_highres(args.processed_root, region, patch_id, args.xuannv_month)

    xuannv_prob, xuannv_pca = predict_probability(
        xuannv_model,
        args.xuannv_embedding_root,
        region,
        patch_id,
        args.xuannv_month,
        device,
    )
    aef_prob, aef_pca = predict_probability(
        aef_model,
        args.aef_embedding_root,
        region,
        patch_id,
        args.aef_month,
        device,
    )
    rows = [
        ("Xuannv Haidian v1", xuannv_pca, xuannv_prob, xuannv_threshold, xuannv_metrics),
        ("AEF annual 2025", aef_pca, aef_prob, aef_threshold, aef_metrics),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18.8, 7.2))
    record: dict[str, Any] = {
        "task": task,
        "patch_id": patch_id,
        "gt_path": str(gt_path),
        "rows": {},
    }
    for row_idx, (name, pca, prob, threshold, metrics) in enumerate(rows):
        pred = (prob >= threshold).astype(np.float32)
        patch_metrics = binary_metrics(pred > 0, gt > 0)
        panels = [
            (highres, f"{name}\nHigh-res {args.xuannv_month}"),
            (pca, f"Embedding PCA\n{args.xuannv_month if name.startswith('Xuannv') else args.aef_month}"),
            (normalized_prob_rgb(prob), "Prediction prob\nper-image norm"),
            (
                red_binary_mask(pred),
                f"Pred mask\nthr={threshold:.3f}, F1={patch_metrics['f1']:.3f}",
            ),
            (red_binary_mask(gt), "GT"),
        ]
        for col_idx, (image, title) in enumerate(panels):
            show_panel(axes[row_idx, col_idx], image, title)
        record["rows"][name] = {
            "threshold": threshold,
            "suite_f1": metrics.get("f1_at_threshold"),
            "suite_ap": metrics.get("ap"),
            "patch_metrics": patch_metrics,
        }
    fig.suptitle(
        f"{info['title']} | {args.head} shot={args.shot} fold={args.fold} | {patch_id}",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir = args.output_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{args.head}_shot_{args.shot}_{patch_id}_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record["figure"] = str(out_path)
    return record


def write_index(records: list[dict[str, Any]], output_root: Path) -> None:
    lines = [
        "# Haidian Capability Suite Visualizations",
        "",
        "Top row is Xuannv Haidian v1. Bottom row is AEF annual 2025.",
        "Each row shows high-res image, embedding PCA, normalized probability, thresholded prediction, and GT.",
        "",
    ]
    for record in records:
        fig = Path(record["figure"])
        xuannv = record["rows"]["Xuannv Haidian v1"]["patch_metrics"]
        aef = record["rows"]["AEF annual 2025"]["patch_metrics"]
        lines.extend(
            [
                f"## {record['task']} / {record['patch_id']}",
                "",
                f"- Xuannv patch F1: `{xuannv['f1']:.4f}`, IoU: `{xuannv['iou']:.4f}`",
                f"- AEF patch F1: `{aef['f1']:.4f}`, IoU: `{aef['iou']:.4f}`",
                "",
                f"![{fig.name}]({fig.relative_to(output_root).as_posix()})",
                "",
            ]
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)

    records: list[dict[str, Any]] = []
    for task in args.tasks:
        if task not in TASKS:
            raise KeyError(f"Unknown task {task}. Available: {sorted(TASKS)}")
        label_root = args.processed_root / "haidian" / "labels" / TASKS[task]["label_task"]
        patch_ids = select_patch_ids(label_root, args.fold, args.samples_per_task)
        xuannv_root = checkpoint_root(
            args.suite_root,
            "xuannv_haidian_v1",
            task,
            args.head,
            args.shot,
            args.fold,
        )
        aef_root = checkpoint_root(
            args.suite_root,
            "aef_annual_2025",
            task,
            args.head,
            args.shot,
            args.fold,
        )
        xuannv_model, xuannv_threshold, xuannv_metrics = load_probe(xuannv_root, args.head, device)
        aef_model, aef_threshold, aef_metrics = load_probe(aef_root, args.head, device)
        for patch_id in patch_ids:
            records.append(
                make_figure(
                    args=args,
                    task=task,
                    patch_id=patch_id,
                    xuannv_model=xuannv_model,
                    xuannv_threshold=xuannv_threshold,
                    xuannv_metrics=xuannv_metrics,
                    aef_model=aef_model,
                    aef_threshold=aef_threshold,
                    aef_metrics=aef_metrics,
                    device=device,
                )
            )
    write_index(records, args.output_root)
    print(f"saved {len(records)} visualizations to {args.output_root}")


if __name__ == "__main__":
    main()
