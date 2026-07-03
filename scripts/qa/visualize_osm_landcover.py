#!/usr/bin/env python3
"""Visualize merged OSM land-cover labels against high-resolution imagery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap


def _stretch_rgb(arr: np.ndarray) -> np.ndarray:
    img = np.moveaxis(arr[:3].astype(np.float32), 0, -1)
    finite = np.isfinite(img)
    if not finite.any():
        return np.ones((*img.shape[:2], 3), dtype=np.float32)
    lo, hi = np.nanpercentile(img[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _resize_nearest(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if label.shape == shape:
        return label
    tensor = torch.from_numpy(label.astype(np.int64))[None, None].float()
    resized = F.interpolate(tensor, size=shape, mode="nearest")[0, 0]
    return resized.numpy().astype(label.dtype)


def _read_reference(processed_root: Path, patch_id: str, date: str) -> np.ndarray:
    path = (
        processed_root
        / "patches"
        / "highres_optical"
        / f"highres_optical_{date}_{patch_id}.tif"
    )
    with rasterio.open(path) as src:
        return _stretch_rgb(src.read())


def _read_label(processed_root: Path, patch_id: str) -> np.ndarray:
    mask_dir = processed_root / "labels" / "osm_landcover" / "masks"
    candidates = sorted(mask_dir.glob(f"*{patch_id}.tif"))
    path = candidates[-1] if candidates else mask_dir / f"{patch_id}.tif"
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def _load_metadata(processed_root: Path) -> tuple[list[str], ListedColormap]:
    path = processed_root / "labels" / "osm_landcover" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    names_by_id = {int(k): v for k, v in metadata["class_names"].items()}
    colors_by_id = {int(k): v for k, v in metadata["class_colors"].items()}
    max_id = max(names_by_id)
    names = ["background"] * (max_id + 1)
    colors = ["#ffffff"] * (max_id + 1)
    for class_id, name in names_by_id.items():
        names[class_id] = name
    for class_id, color in colors_by_id.items():
        colors[class_id] = color
    return names, ListedColormap(colors)


def visualize_patch(
    processed_root: Path,
    patch_id: str,
    output_dir: Path,
    reference_date: str,
) -> Path:
    reference = _read_reference(processed_root, patch_id, reference_date)
    label_lowres = _read_label(processed_root, patch_id)
    label = _resize_nearest(label_lowres, reference.shape[:2])
    names, cmap = _load_metadata(processed_root)

    label_rgb = cmap(label)[..., :3]
    overlay = reference.copy()
    mask = label > 0
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * label_rgb[mask]

    counts = np.bincount(label.reshape(-1), minlength=len(names))
    present = [(idx, names[idx], counts[idx]) for idx in range(len(names)) if counts[idx] > 0]
    legend = "\n".join(f"{idx}: {name} ({count})" for idx, name, count in present)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.65]},
    )
    axes[0].imshow(reference)
    axes[0].set_title(f"High-res optical {reference_date}")
    axes[1].imshow(label, cmap=cmap, vmin=0, vmax=len(names) - 1, interpolation="nearest")
    axes[1].set_title("OSM landcover")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[3].axis("off")
    axes[3].text(0.0, 1.0, legend, va="top", fontsize=9, family="monospace")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Merged OSM land-cover | {patch_id}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patch_id}_osm_landcover.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian"),
    )
    parser.add_argument("--patch-ids", nargs="+", required=True)
    parser.add_argument("--reference-date", default="20260501")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/xuannv_embedding/qa/pretrain_p6a_20260703/osm_landcover"),
    )
    args = parser.parse_args()
    for patch_id in args.patch_ids:
        print(visualize_patch(args.processed_root, patch_id, args.output_dir, args.reference_date))


if __name__ == "__main__":
    main()
