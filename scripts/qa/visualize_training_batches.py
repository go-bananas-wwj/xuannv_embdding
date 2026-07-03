#!/usr/bin/env python3
"""Visualize monthly training batches after manifest filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset


BACKGROUND = "#ffffff"
CLASS_RULES = [
    (1, "residential", "osm_residential", "#f9844a"),
    (2, "commercial", "osm_commercial", "#ff70a6"),
    (3, "industrial", "osm_industrial", "#6d6875"),
    (4, "agriculture", "osm_agriculture", "#90be6d"),
    (5, "green", "osm_green", "#43aa8b"),
    (6, "playground", "osm_playground", "#577590"),
    (7, "construction", "osm_construction", "#9d4edd"),
    (8, "water", "osm_water", "#4aa3df"),
    (9, "rail", "osm_rail", "#7a4fa3"),
    (10, "path/walk", "osm_path_walk", "#f9c74f"),
    (11, "minor road", "osm_minor_road", "#f7a072"),
    (12, "major road", "osm_major_road", "#f4d35e"),
    (13, "building", "osm_building", "#d62828"),
]


def _month_int(month: str) -> int:
    year, mon = month.split("-", 1)
    return int(year) * 100 + int(mon)


def _stretch(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= mask.astype(bool)
    if not finite.any():
        return np.ones((*arr.shape[-2:], 3), dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out


def _to_rgb(chw: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    arr = chw.detach().cpu().float().numpy()
    mask_np = None if mask is None else mask.detach().cpu().float().numpy() > 0.0
    if arr.shape[0] >= 3:
        rgb = np.moveaxis(arr[:3], 0, -1)
        rgb = _stretch(rgb, mask_np[..., None] if mask_np is not None else None)
    elif arr.shape[0] == 2:
        channels = np.stack([arr[0], arr[1], 0.5 * (arr[0] + arr[1])], axis=-1)
        rgb = _stretch(channels, mask_np[..., None] if mask_np is not None else None)
    else:
        gray = _stretch(arr[0], mask_np)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mask_np is not None:
        rgb = rgb * (0.35 + 0.65 * mask_np[..., None])
    return rgb


def _blank(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    return np.ones((*shape, 3), dtype=np.float32)


def _osm_multiclass(labels: dict[str, torch.Tensor], sample_idx: int) -> np.ndarray:
    label = np.zeros((128, 128), dtype=np.uint8)
    for class_id, _name, task, _color in CLASS_RULES:
        candidates = (task, f"haidian_{task}")
        tensor = next((labels[key] for key in candidates if key in labels), None)
        if tensor is None:
            continue
        mask = tensor[sample_idx].detach().cpu().numpy() > 0.5
        label[mask] = class_id
    return label


def _osm_cmap() -> ListedColormap:
    max_id = max(rule[0] for rule in CLASS_RULES)
    colors = [BACKGROUND] * (max_id + 1)
    for class_id, _name, _task, color in CLASS_RULES:
        colors[class_id] = color
    return ListedColormap(colors)


def _source_month_image(
    batch: dict,
    source: str,
    sample_idx: int,
    month_idx: int,
    month: int,
) -> np.ndarray:
    frames = batch["source_frames"][source]
    masks = batch["source_masks"][source]
    pixel_masks = batch["source_pixel_masks"][source]
    if source.startswith("highres"):
        timestamps = batch["source_timestamps"][source][sample_idx]
        matches = (timestamps == month).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            _, _, height, width = frames.shape[-4:]
            return _blank((height, width))
        frame_idx = int(matches[0].item())
        if masks[sample_idx, frame_idx].item() <= 0:
            _, _, height, width = frames.shape[-4:]
            return _blank((height, width))
        return _to_rgb(frames[sample_idx, frame_idx], pixel_masks[sample_idx, frame_idx])

    if masks[sample_idx, month_idx].item() <= 0:
        _, height, width = frames.shape[-3:]
        return _blank((height, width))
    return _to_rgb(frames[sample_idx, month_idx], pixel_masks[sample_idx, month_idx])


def build_dataset(config_path: Path) -> tuple[Config, MonthlyEmbeddingDataset]:
    cfg = Config.from_yaml(config_path)
    data_cfg = cfg.data
    if data_cfg.statistics_dir is not None:
        statistics_dir = data_cfg.statistics_dir
    else:
        statistics_dir = data_cfg.root.parent / "statistics" / data_cfg.region
    ref_year, ref_month = (2025, 1)
    if data_cfg.months:
        year_str, month_str = data_cfg.months[0].split("-", 1)
        ref_year, ref_month = int(year_str), int(month_str)
    dataset = MonthlyEmbeddingDataset(
        manifest_path=data_cfg.manifest_path,
        statistics_dir=statistics_dir,
        sources=data_cfg.sources,
        patch_size=data_cfg.patch_size,
        max_patches=data_cfg.max_patches,
        num_months=data_cfg.num_months,
        ref_year=ref_year,
        ref_month=ref_month,
        statistics_dirs_by_region=data_cfg.statistics_dirs_by_region,
        supervised_label_roots=data_cfg.supervised_label_roots,
    )
    return cfg, dataset


def visualize_batch(
    batch: dict,
    months: list[str],
    sources: list[str],
    batch_idx: int,
    output_dir: Path,
) -> Path:
    month_ints = [_month_int(month) for month in months]
    patch_ids = list(batch["patch_ids"])
    rows_per_sample = len(sources) + 1
    nrows = len(patch_ids) * rows_per_sample
    ncols = len(months)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.35 * ncols, 1.8 * nrows),
        squeeze=False,
    )
    osm_cmap = _osm_cmap()

    for sample_idx, patch_id in enumerate(patch_ids):
        row_base = sample_idx * rows_per_sample
        for source_idx, source in enumerate(sources):
            row = row_base + source_idx
            for month_idx, month in enumerate(month_ints):
                ax = axes[row][month_idx]
                img = _source_month_image(batch, source, sample_idx, month_idx, month)
                ax.imshow(img)
                if row == 0:
                    ax.set_title(months[month_idx], fontsize=10)
                if month_idx == 0:
                    ax.set_ylabel(f"{patch_id}\n{source}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

        osm_label = _osm_multiclass(batch.get("supervised_labels", {}), sample_idx)
        row = row_base + len(sources)
        for month_idx, _month in enumerate(month_ints):
            ax = axes[row][month_idx]
            ax.imshow(osm_label, cmap=osm_cmap, vmin=0, vmax=len(CLASS_RULES), interpolation="nearest")
            if month_idx == 0:
                ax.set_ylabel(f"{patch_id}\nOSM weak", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    legend = "\n".join(f"{class_id}: {name}" for class_id, name, _task, _color in CLASS_RULES)
    fig.text(0.995, 0.5, legend, va="center", ha="right", fontsize=8, family="monospace")
    fig.suptitle(f"Filtered training batch {batch_idx}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.90, 0.975])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"filtered_training_batch_{batch_idx:02d}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/v2_p6a_haidian_202512_202605_pixelmask_full_20260703.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/xuannv_embedding/qa/pretrain_p6a_20260703/filtered_training_batches"),
    )
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["s2", "landsat", "s1", "highres_optical_haidian", "highres_sar_haidian"],
    )
    args = parser.parse_args()

    cfg, dataset = build_dataset(args.config)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
    )

    count = 0
    for batch_idx, batch in enumerate(loader):
        out_path = visualize_batch(
            batch,
            months=cfg.data.months,
            sources=[source for source in args.sources if source in cfg.data.sources],
            batch_idx=batch_idx,
            output_dir=args.output_dir,
        )
        print(out_path)
        count += 1
        if count >= args.num_batches:
            break


if __name__ == "__main__":
    main()
