#!/usr/bin/env python3
"""Create side-by-side downstream output visualizations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from sklearn.decomposition import PCA

DEFAULT_TASKS = (
    "construction",
    "haidian_building_osm",
    "haidian_road_osm",
    "haidian_water_osm",
    "building_change",
    "farm_change",
    "rubbish",
    "water",
    "construction_joint",
)

TASK_REGION = {
    "construction": "haidian",
    "haidian_building_osm": "haidian",
    "haidian_road_osm": "haidian",
    "haidian_water_osm": "haidian",
    "building_change": "harbin",
    "farm_change": "harbin",
    "rubbish": "harbin",
    "water": "harbin",
    "construction_joint": "construction_joint",
}

TASK_LABEL_DIR = {
    "water": "osm_water",
    "haidian_building_osm": "building_osm",
    "haidian_road_osm": "road_osm",
    "haidian_water_osm": "osm_water",
}


class PcaView(NamedTuple):
    model: PCA
    lo: np.ndarray
    hi: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--months", nargs=2, default=["202512", "202605"])
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-pca-mode",
        choices=["global", "patch"],
        default="global",
        help="global uses one PCA projection for all selected samples; patch fits PCA per patch.",
    )
    parser.add_argument(
        "--pca-sample-pixels",
        type=int,
        default=200000,
        help="Maximum embedding pixels sampled to fit global PCA.",
    )
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


def resolve_region_patch(task: str, patch_id: str) -> tuple[str, str]:
    for region in ("haidian", "harbin"):
        prefix = f"{region}_"
        if patch_id.startswith(prefix):
            return region, patch_id[len(prefix):]
    return TASK_REGION.get(task, "harbin"), patch_id


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


def load_prediction(pred_path: Path) -> np.ndarray:
    with rasterio.open(pred_path) as src:
        return src.read(1)


def load_task_thresholds(benchmark_root: Path, task: str) -> dict[int, float]:
    summary_path = benchmark_root / task / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    thresholds: dict[int, float] = {}
    for row in summary:
        if "fold" not in row:
            continue
        thresholds[int(row["fold"])] = float(row.get("val_threshold", row.get("threshold", 0.5)))
    return thresholds


def fold_from_prediction(pred_path: Path) -> int | None:
    match = re.match(r"fold_(\d+)$", pred_path.parent.parent.name)
    if match is None:
        return None
    return int(match.group(1))


def threshold_for_prediction(pred_path: Path, thresholds: dict[int, float]) -> tuple[float, str]:
    fold = fold_from_prediction(pred_path)
    if fold is not None and fold in thresholds:
        return thresholds[fold], f"fold_{fold}"
    if thresholds:
        values = list(thresholds.values())
        return float(np.median(values)), "median_fallback"
    return 0.5, "default_0.5"


def load_gt_mask(
    processed_root: Path,
    task: str,
    patch_id: str,
    preferred_month: str,
) -> tuple[np.ndarray | None, str | None, int]:
    if task == "construction_joint":
        mask_dir = processed_root / "construction_joint_v2" / "masks"
        candidate_ids = [patch_id]
    else:
        region, source_patch = resolve_region_patch(task, patch_id)
        label_dir = TASK_LABEL_DIR.get(task, task)
        mask_dir = processed_root / region / "labels" / label_dir / "masks"
        candidate_ids = [source_patch, patch_id]

    candidates: list[Path] = []
    for candidate_id in dict.fromkeys(candidate_ids):
        candidates.extend(mask_dir.glob(f"{candidate_id}.tif"))
        candidates.extend(mask_dir.glob(f"{candidate_id}_*.tif"))
    candidates = sorted(set(candidates))
    if not candidates:
        return None, None, 0

    preferred = [path for path in candidates if preferred_month in path.stem]
    search_order = preferred + [path for path in candidates if path not in preferred]

    best_mask: np.ndarray | None = None
    best_path: Path | None = None
    best_positive = -1
    for path in search_order:
        with rasterio.open(path) as src:
            mask = (src.read(1) > 0).astype(np.float32)
        positives = int(mask.sum())
        if path in preferred and positives > 0:
            return mask, str(path), positives
        if positives > best_positive:
            best_mask = mask
            best_path = path
            best_positive = positives

    if best_mask is None or best_path is None:
        return None, None, 0
    return best_mask, str(best_path), max(best_positive, 0)


def load_embedding(
    embedding_root: Path,
    task: str,
    patch_id: str,
    month: str,
) -> torch.Tensor | None:
    region = TASK_REGION.get(task, "harbin")
    candidates = [
        embedding_root / region / patch_id / f"{month}_embedding_map.pt",
        embedding_root / task / patch_id / f"{month}_embedding_map.pt",
    ]
    source_region, source_patch = resolve_region_patch(task, patch_id)
    candidates.extend(
        [
            embedding_root
            / region
            / f"{source_region}_{source_patch}"
            / f"{month}_embedding_map.pt",
            embedding_root / source_region / patch_id / f"{month}_embedding_map.pt",
            embedding_root / source_region / source_patch / f"{month}_embedding_map.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
    return None


def fit_global_embedding_pca(
    samples: list[tuple[str, Path]],
    embedding_root: Path,
    months: list[str],
    max_pixels: int,
    seed: int,
) -> PcaView | None:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    total_pixels = 0
    for task, pred_path in samples:
        patch_id = pred_path.stem.removesuffix("_prob")
        for month in months:
            embedding = load_embedding(embedding_root, task, patch_id, month)
            if embedding is None:
                continue
            arr = embedding.float().numpy()
            channels, height, width = arr.shape
            flat = arr.reshape(channels, height * width).T
            if flat.shape[0] == 0:
                continue
            remaining = max_pixels - total_pixels
            if remaining <= 0:
                break
            take = min(flat.shape[0], remaining)
            if take < flat.shape[0]:
                idx = rng.choice(flat.shape[0], size=take, replace=False)
                flat = flat[idx]
            chunks.append(flat.astype(np.float32, copy=False))
            total_pixels += flat.shape[0]
        if total_pixels >= max_pixels:
            break
    if not chunks:
        return None
    fit_data = np.concatenate(chunks, axis=0)
    if float(np.nanstd(fit_data)) < 1e-8:
        return None
    pca = PCA(n_components=3)
    coords = pca.fit_transform(fit_data)
    lo = np.percentile(coords, 2.0, axis=0).astype(np.float32)
    hi = np.percentile(coords, 98.0, axis=0).astype(np.float32)
    hi = np.where(hi <= lo, lo + 1.0, hi).astype(np.float32)
    return PcaView(pca, lo, hi)


def stretch_with_bounds(image: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        out[..., channel] = np.clip(
            (image[..., channel] - lo[channel]) / (hi[channel] - lo[channel]),
            0.0,
            1.0,
        )
    return out


def embedding_pca(
    embedding: torch.Tensor | None,
    pca_view: PcaView | None = None,
) -> np.ndarray | None:
    if embedding is None:
        return None
    arr = embedding.float().numpy()
    channels, height, width = arr.shape
    flat = arr.reshape(channels, -1).T
    if float(np.nanstd(flat)) < 1e-8:
        return np.zeros((height, width, 3), dtype=np.float32)
    if pca_view is None:
        flat = flat - flat.mean(axis=0, keepdims=True)
        pca_model = PCA(n_components=3)
        rgb = pca_model.fit_transform(flat).reshape(height, width, 3)
        return stretch(rgb)
    else:
        rgb = pca_view.model.transform(flat).reshape(height, width, 3)
        return stretch_with_bounds(rgb, pca_view.lo, pca_view.hi)


def embedding_delta(
    before: torch.Tensor | None,
    after: torch.Tensor | None,
    pca_view: PcaView | None = None,
) -> np.ndarray | None:
    if before is None or after is None:
        return None
    if pca_view is None:
        return embedding_pca(after.float() - before.float())
    delta = after.float() - before.float()
    return embedding_pca(delta, None)


def positive_pixels_for_task(processed_root: Path, task: str, patch_id: str) -> int:
    if task == "construction_joint":
        mask_dir = processed_root / "construction_joint_v2" / "masks"
    else:
        region, source_patch = resolve_region_patch(task, patch_id)
        label_dir = TASK_LABEL_DIR.get(task, task)
        mask_dir = processed_root / region / "labels" / label_dir / "masks"
        patch_id = source_patch
    candidates = list(mask_dir.glob(f"{patch_id}.tif")) + list(mask_dir.glob(f"{patch_id}_*.tif"))
    positives = 0
    for path in candidates:
        with rasterio.open(path) as src:
            positives = max(positives, int((src.read(1) > 0).sum()))
    return positives


def select_predictions(
    benchmark_root: Path,
    processed_root: Path,
    task: str,
    samples_per_task: int,
) -> list[Path]:
    pred_paths = sorted((benchmark_root / task).glob("fold_*/predictions/*_prob.tif"))
    scored: list[tuple[int, float, Path]] = []
    for pred_path in pred_paths:
        patch_id = pred_path.stem.removesuffix("_prob")
        positives = positive_pixels_for_task(processed_root, task, patch_id)
        pred = load_prediction(pred_path)
        scored.append((positives, float(np.nanmax(pred)), pred_path))
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
    embedding_root: Path,
    processed_root: Path,
    output_dir: Path,
    months: list[str],
    threshold: float,
    threshold_source: str,
    pca_view: PcaView | None,
) -> dict[str, Any]:
    patch_id = pred_path.stem.removesuffix("_prob")
    source_region, source_patch = resolve_region_patch(task, patch_id)
    before_month, after_month = months

    before_hr = load_highres(processed_root, source_region, source_patch, before_month)
    after_hr = load_highres(processed_root, source_region, source_patch, after_month)
    before_emb = load_embedding(embedding_root, task, patch_id, before_month)
    after_emb = load_embedding(embedding_root, task, patch_id, after_month)
    before_pca = embedding_pca(before_emb, pca_view)
    after_pca = embedding_pca(after_emb, pca_view)
    delta = embedding_delta(before_emb, after_emb)
    pred_prob = load_prediction(pred_path)
    pred = stretch(pred_prob, lower=0.0, upper=100.0)
    pred_binary = (pred_prob >= threshold).astype(np.float32)
    gt_mask, gt_path, gt_positive_pixels = load_gt_mask(
        processed_root,
        task,
        patch_id,
        after_month,
    )

    panels = [
        (before_hr, f"High-res {before_month}", None),
        (after_hr, f"High-res {after_month}", None),
        (before_pca, f"Embedding PCA {before_month}", None),
        (after_pca, f"Embedding PCA {after_month}", None),
        (delta, "PDA / Delta Emb PCA", None),
        (pred, "Prediction Prob", "Reds"),
        (pred_binary, f"Pred >= {threshold:.3f} ({threshold_source})", "Reds"),
        (gt_mask, "GT / True Label", "Reds"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3))
    for ax, (image, title, cmap) in zip(axes, panels, strict=True):
        show_panel(ax, image, title, cmap)
    fig.suptitle(f"{task} | {patch_id}", fontsize=11)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task}_{patch_id}_output_strip.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "task": task,
        "patch_id": patch_id,
        "source_region": source_region,
        "source_patch_id": source_patch,
        "prediction": str(pred_path),
        "figure": str(out_path),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "fold": fold_from_prediction(pred_path),
        "gt_mask": gt_path,
        "gt_positive_pixels": gt_positive_pixels,
        "missing": {
            "before_highres": before_hr is None,
            "after_highres": after_hr is None,
            "before_embedding": before_emb is None,
            "after_embedding": after_emb is None,
            "gt_mask": gt_mask is None,
        },
    }


def write_index(records: list[dict[str, Any]], output_root: Path) -> None:
    lines = ["# Downstream Output Visualizations", ""]
    for record in records:
        fig = Path(record["figure"])
        lines.extend(
            [
                f"## {record['task']} / {record['patch_id']}",
                "",
                f"- source_region: `{record['source_region']}`",
                f"- source_patch_id: `{record['source_patch_id']}`",
                f"- prediction: `{record['prediction']}`",
                f"- gt_mask: `{record['gt_mask']}`",
                f"- gt_positive_pixels: `{record['gt_positive_pixels']}`",
                f"- missing: `{record['missing']}`",
                "",
                f"![{fig.name}]({fig.relative_to(output_root)})",
                "",
            ]
        )
    (output_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    selected: list[tuple[str, Path]] = []
    for task in args.tasks:
        for pred_path in select_predictions(
            args.benchmark_root,
            args.processed_root,
            task,
            args.samples_per_task,
        ):
            selected.append((task, pred_path))

    pca_view = None
    if args.embedding_pca_mode == "global":
        pca_view = fit_global_embedding_pca(
            selected,
            args.embedding_root,
            [str(month) for month in args.months],
            args.pca_sample_pixels,
            args.seed,
        )

    thresholds_by_task = {
        task: load_task_thresholds(args.benchmark_root, task) for task in args.tasks
    }
    for task, pred_path in selected:
        task_output = args.output_root / task
        threshold, threshold_source = threshold_for_prediction(
            pred_path,
            thresholds_by_task.get(task, {}),
        )
        records.append(
            make_visual(
                task,
                pred_path,
                args.embedding_root,
                args.processed_root,
                task_output,
                [str(month) for month in args.months],
                threshold,
                threshold_source,
                pca_view,
            )
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_index(records, args.output_root)
    print(f"saved {len(records)} visualizations to {args.output_root}")


if __name__ == "__main__":
    main()
