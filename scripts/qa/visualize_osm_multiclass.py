from __future__ import annotations

"""Visualize OSM weak semantic masks as one multiclass land-object map."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

CLASS_RULES = [
    (1, "water", "osm_water", "#4aa3df"),
    (2, "rail", "osm_rail", "#7a4fa3"),
    (3, "major road", "osm_major_road", "#f4d35e"),
    (4, "minor road", "osm_minor_road", "#f7a072"),
    (5, "path/walk", "osm_path_walk", "#f9c74f"),
    (6, "building", "osm_building", "#d62828"),
    (7, "construction", "osm_construction", "#9d4edd"),
    (8, "industrial", "osm_industrial", "#6d6875"),
    (9, "commercial", "osm_commercial", "#ff70a6"),
    (10, "residential", "osm_residential", "#f9844a"),
    (11, "green", "osm_green", "#43aa8b"),
    (12, "agriculture", "osm_agriculture", "#90be6d"),
    (13, "playground", "osm_playground", "#577590"),
    (14, "activity poi", "osm_activity_poi", "#f8961e"),
]
BACKGROUND = "#ffffff"


def _read_mask(label_root: Path, task: str, patch_id: str, shape: tuple[int, int]) -> np.ndarray:
    path = label_root / task / "masks" / f"{patch_id}.tif"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(path) as src:
        return src.read(1) > 0


def build_multiclass(label_root: Path, patch_id: str, shape: tuple[int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    for class_id, _name, task, _color in CLASS_RULES:
        mask = _read_mask(label_root, task, patch_id, shape)
        label[mask] = class_id
    return label


def _stretch_rgb(arr: np.ndarray) -> np.ndarray:
    img = np.moveaxis(arr[:3].astype(np.float32), 0, -1)
    finite = np.isfinite(img)
    if not finite.any():
        return np.ones((*img.shape[:2], 3), dtype=np.float32)
    lo, hi = np.nanpercentile(img[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _read_reference_rgb(processed_root: Path, patch_id: str, date: str) -> np.ndarray:
    path = (
        processed_root
        / "patches"
        / "highres_optical"
        / f"highres_optical_{date}_{patch_id}.tif"
    )
    with rasterio.open(path) as src:
        arr = src.read()
    return _stretch_rgb(arr)


def visualize_patch(
    processed_root: Path,
    patch_id: str,
    output_dir: Path,
    reference_date: str,
) -> Path:
    label_root = processed_root / "labels"
    reference = _read_reference_rgb(processed_root, patch_id, reference_date)
    label = build_multiclass(label_root, patch_id, reference.shape[:2])

    colors = [BACKGROUND] + [color for *_rest, color in CLASS_RULES]
    names = ["background"] + [name for _id, name, _task, _color in CLASS_RULES]
    cmap = ListedColormap(colors)

    overlay = reference.copy()
    label_rgb = cmap(label)[..., :3]
    mask = label > 0
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * label_rgb[mask]

    counts = np.bincount(label.reshape(-1), minlength=len(names))
    present = [(idx, names[idx], counts[idx]) for idx in range(len(names)) if counts[idx] > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reference)
    axes[0].set_title(f"High-res optical {reference_date}")
    axes[1].imshow(label, cmap=cmap, vmin=0, vmax=len(names) - 1, interpolation="nearest")
    axes[1].set_title("OSM multiclass")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_text = "\n".join(
        f"{idx}: {name} ({count})" for idx, name, count in present[:16]
    )
    fig.text(0.02, 0.02, legend_text, fontsize=9, va="bottom", family="monospace")
    fig.suptitle(f"OSM weak semantic map | {patch_id}", fontsize=15)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{patch_id}_osm_multiclass.png"
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
        default=Path("/data/xuannv_embedding/qa/pretrain_p6a_20260703/osm_multiclass"),
    )
    args = parser.parse_args()
    for patch_id in args.patch_ids:
        print(visualize_patch(args.processed_root, patch_id, args.output_dir, args.reference_date))


if __name__ == "__main__":
    main()
