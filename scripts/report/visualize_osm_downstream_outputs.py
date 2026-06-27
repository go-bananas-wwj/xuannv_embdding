#!/usr/bin/env python3
"""Visualize OSM downstream predictions, GT masks, and P1B embeddings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from sklearn.decomposition import PCA

DEFAULT_RUN_ROOT = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/"
    "p1b_osm_quick_20260627"
)
DEFAULT_EMBEDDING_ROOT = Path(
    "/data/xuannv_embedding/embeddings/v2_202512_202605/"
    "20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_"
    "v2_p1_sparse_sampler_hardneg_20260627_090500"
)
DEFAULT_PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")

TASKS = {
    "haidian_building_osm": {"region": "haidian", "label_task": "building_osm"},
    "haidian_road_osm": {"region": "haidian", "label_task": "road_osm"},
    "harbin_building_osm": {"region": "harbin", "label_task": "building_osm"},
    "harbin_road_osm": {"region": "harbin", "label_task": "road_osm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--month", default="202605")
    parser.add_argument("--samples-per-task", type=int, default=4)
    return parser.parse_args()


def stretch(image: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0)
    if image.ndim == 2:
        lo, hi = np.percentile(image, [lower, upper])
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    out = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        band = image[..., channel]
        lo, hi = np.percentile(band, [lower, upper])
        if hi > lo:
            out[..., channel] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def choose_month_file(files: list[Path], month: str) -> Path | None:
    files = [path for path in files if not path.name.endswith("_mask.tif")]
    if not files:
        return None
    month_re = re.compile(rf"_{month}(?:\d{{2}})?_")
    candidates = [path for path in files if month_re.search(path.name)]
    if candidates:
        return sorted(candidates)[-1]
    return sorted(files)[-1]


def load_highres(processed_root: Path, region: str, patch_id: str, month: str) -> np.ndarray | None:
    root = processed_root / region / "patches" / "highres_optical"
    path = choose_month_file(sorted(root.glob(f"*{patch_id}.tif")), month)
    if path is None:
        return None
    with rasterio.open(path) as src:
        arr = src.read()
    return stretch(np.transpose(arr[:3], (1, 2, 0)))


def load_embedding(embedding_root: Path, region: str, patch_id: str, month: str) -> torch.Tensor | None:
    candidates = [
        embedding_root / region / patch_id / f"{month}_embedding_map.pt",
        embedding_root / region / f"{region}_{patch_id}" / f"{month}_embedding_map.pt",
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
    return None


def embedding_pca(embedding: torch.Tensor | None) -> np.ndarray | None:
    if embedding is None:
        return None
    arr = embedding.float().numpy()
    channels, height, width = arr.shape
    flat = arr.reshape(channels, -1).T
    flat = flat - flat.mean(axis=0, keepdims=True)
    if float(np.nanstd(flat)) < 1e-8:
        return np.zeros((height, width, 3), dtype=np.float32)
    rgb = PCA(n_components=3).fit_transform(flat).reshape(height, width, 3)
    return stretch(rgb)


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def overlay_mask(image: np.ndarray | None, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray | None:
    if image is None:
        return None
    out = image.copy()
    h, w = out.shape[:2]
    y_idx = np.linspace(0, mask.shape[0] - 1, h).astype(int)
    x_idx = np.linspace(0, mask.shape[1] - 1, w).astype(int)
    up = mask[np.ix_(y_idx, x_idx)] > 0
    color_arr = np.array(color, dtype=np.float32)
    out[up] = out[up] * 0.45 + color_arr * 0.55
    return out


def load_prediction(pred_path: Path) -> np.ndarray:
    with rasterio.open(pred_path) as src:
        return src.read(1)


def positive_pixels(label_root: Path, patch_id: str) -> int:
    mask_path = label_root / "masks" / f"{patch_id}.tif"
    if not mask_path.exists():
        return 0
    return int((read_mask(mask_path) > 0).sum())


def select_predictions(run_root: Path, task: str, label_root: Path, samples_per_task: int) -> list[Path]:
    pred_paths = sorted((run_root / task).glob("fold_*/predictions/*_prob.tif"))
    scored: list[tuple[int, float, Path]] = []
    for pred_path in pred_paths:
        patch_id = pred_path.stem.removesuffix("_prob")
        pred = load_prediction(pred_path)
        scored.append((positive_pixels(label_root, patch_id), float(np.nanmax(pred)), pred_path))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:samples_per_task]]


def show_panel(ax: plt.Axes, image: np.ndarray | None, title: str, cmap: str | None = None) -> None:
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
    elif cmap is not None:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    elif image.ndim == 2:
        ax.imshow(image, cmap="magma")
    else:
        ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_visual(
    task: str,
    pred_path: Path,
    region: str,
    label_task: str,
    embedding_root: Path,
    processed_root: Path,
    output_dir: Path,
    month: str,
) -> dict[str, Any]:
    patch_id = pred_path.stem.removesuffix("_prob")
    label_root = processed_root / region / "labels" / label_task
    gt_path = label_root / "masks" / f"{patch_id}.tif"
    highres = load_highres(processed_root, region, patch_id, month)
    emb_pca = embedding_pca(load_embedding(embedding_root, region, patch_id, month))
    pred = stretch(load_prediction(pred_path), lower=0.0, upper=100.0)
    gt = (read_mask(gt_path) > 0).astype(np.float32) if gt_path.exists() else None
    pred_binary = (load_prediction(pred_path) >= 0.5).astype(np.float32)
    gt_overlay = overlay_mask(highres, gt if gt is not None else np.zeros((128, 128)), (1.0, 0.05, 0.05))
    pred_overlay = overlay_mask(highres, pred_binary, (0.05, 0.85, 1.0))

    panels = [
        (highres, f"High-res {month}", None),
        (emb_pca, f"P1B Emb PCA {month}", None),
        (pred, "Prediction Prob", "viridis"),
        (gt, "OSM GT", "Reds"),
        (pred_overlay, "Pred Overlay", None),
        (gt_overlay, "GT Overlay", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
    for ax, (image, title, cmap) in zip(axes, panels, strict=True):
        show_panel(ax, image, title, cmap)
    fig.suptitle(f"{task} | {patch_id}", fontsize=11)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task}_{patch_id}_osm_contact.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gt_positive = int(gt.sum()) if gt is not None else 0
    return {
        "task": task,
        "region": region,
        "label_task": label_task,
        "patch_id": patch_id,
        "figure": str(out_path),
        "prediction": str(pred_path),
        "gt_mask": str(gt_path) if gt_path.exists() else None,
        "gt_positive_pixels": gt_positive,
        "missing": {
            "highres": highres is None,
            "embedding": emb_pca is None,
            "gt_mask": gt is None,
        },
    }


def write_index(records: list[dict[str, Any]], output_root: Path) -> None:
    lines = ["# P1B OSM Downstream Contact Sheets", ""]
    for record in records:
        fig = Path(record["figure"])
        lines.extend(
            [
                f"## {record['task']} / {record['patch_id']}",
                "",
                f"- region: `{record['region']}`",
                f"- label_task: `{record['label_task']}`",
                f"- gt_positive_pixels: `{record['gt_positive_pixels']}`",
                f"- prediction: `{record['prediction']}`",
                f"- gt_mask: `{record['gt_mask']}`",
                f"- missing: `{record['missing']}`",
                "",
                f"![{fig.name}]({fig.relative_to(output_root).as_posix()})",
                "",
            ]
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.run_root / "osm_contact_sheets"
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        info = TASKS[task]
        label_root = args.processed_root / info["region"] / "labels" / info["label_task"]
        for pred_path in select_predictions(args.run_root, task, label_root, args.samples_per_task):
            records.append(
                make_visual(
                    task=task,
                    pred_path=pred_path,
                    region=info["region"],
                    label_task=info["label_task"],
                    embedding_root=args.embedding_root,
                    processed_root=args.processed_root,
                    output_dir=output_root / task,
                    month=args.month,
                )
            )
    write_index(records, output_root)
    print(f"saved {len(records)} contact sheets to {output_root}")


if __name__ == "__main__":
    main()
