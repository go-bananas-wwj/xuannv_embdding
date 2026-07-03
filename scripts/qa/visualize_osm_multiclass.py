from __future__ import annotations

"""Visualize OSM weak semantic masks as one multiclass land-object map."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

# Draw broad land-use areas first and fine object-like labels later. Later rules
# override earlier ones when OSM masks overlap.
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
OPTIONAL_NOISY_RULES = [
    (14, "activity poi", "osm_activity_poi", "#f8961e"),
]
BACKGROUND = "#ffffff"


def _read_mask(label_root: Path, task: str, patch_id: str, shape: tuple[int, int]) -> np.ndarray:
    path = label_root / task / "masks" / f"{patch_id}.tif"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(path) as src:
        return src.read(1) > 0


def build_multiclass(
    label_root: Path,
    patch_id: str,
    shape: tuple[int, int],
    class_rules: list[tuple[int, str, str, str]],
) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    for class_id, _name, task, _color in class_rules:
        mask = _read_mask(label_root, task, patch_id, shape)
        label[mask] = class_id
    return label


def _resize_label_nearest(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if label.shape == shape:
        return label
    tensor = torch.from_numpy(label.astype(np.int64))[None, None].float()
    resized = F.interpolate(tensor, size=shape, mode="nearest")[0, 0]
    return resized.numpy().astype(label.dtype)


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
    include_noisy: bool,
) -> Path:
    class_rules = CLASS_RULES + OPTIONAL_NOISY_RULES if include_noisy else CLASS_RULES
    label_root = processed_root / "labels"
    reference = _read_reference_rgb(processed_root, patch_id, reference_date)
    label_lowres = build_multiclass(label_root, patch_id, (128, 128), class_rules)
    label = _resize_label_nearest(label_lowres, reference.shape[:2])

    max_id = max(class_id for class_id, *_rest in class_rules)
    colors = [BACKGROUND] * (max_id + 1)
    names = ["background"] * (max_id + 1)
    for class_id, name, _task, color in class_rules:
        colors[class_id] = color
        names[class_id] = name
    cmap = ListedColormap(colors)

    overlay = reference.copy()
    label_rgb = cmap(label)[..., :3]
    mask = label > 0
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * label_rgb[mask]

    counts = np.bincount(label.reshape(-1), minlength=len(names))
    present = [(idx, names[idx], counts[idx]) for idx in range(len(names)) if counts[idx] > 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1, 0.65]})
    axes[0].imshow(reference)
    axes[0].set_title(f"High-res optical {reference_date}")
    axes[1].imshow(label, cmap=cmap, vmin=0, vmax=max_id, interpolation="nearest")
    axes[1].set_title("OSM multiclass")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[3].axis("off")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_text = "\n".join(
        f"{idx}: {name} ({count})" for idx, name, count in present[:16]
    )
    axes[3].text(0.0, 1.0, legend_text, fontsize=9, va="top", family="monospace")
    fig.suptitle(f"OSM weak semantic map | {patch_id}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

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
    parser.add_argument(
        "--include-noisy",
        action="store_true",
        help="Also draw noisy OSM POI buffers such as osm_activity_poi.",
    )
    args = parser.parse_args()
    for patch_id in args.patch_ids:
        print(
            visualize_patch(
                args.processed_root,
                patch_id,
                args.output_dir,
                args.reference_date,
                args.include_noisy,
            )
        )


if __name__ == "__main__":
    main()
