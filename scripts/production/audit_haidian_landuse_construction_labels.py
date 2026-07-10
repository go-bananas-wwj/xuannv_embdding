#!/usr/bin/env python3
"""Audit Haidian production land-use and construction labels."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw


PATCH_RE = re.compile(r"(patch_\d{6})")


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian"),
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/embeddings/production/"
            "20260709_haidian_embedding_v1_p10c_epoch800_production_epoch_80_"
            "haidian_embedding_v1_p10c_epoch800"
        ),
    )
    parser.add_argument("--month", default="202604")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/experiments/production/"
            "haidian_v1_landuse_construction_20260710/label_audit"
        ),
    )
    parser.add_argument("--sample-patches", nargs="*", default=None)
    parser.add_argument("--random-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def patch_id_from_path(path: Path) -> str:
    match = PATCH_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse patch id from {path}")
    return match.group(1)


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    image = np.moveaxis(arr[:3].astype(np.float32), 0, -1)
    image = np.nan_to_num(image, nan=0.0)
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(image[..., c], [2, 98])
        if hi > lo:
            out[..., c] = np.clip((image[..., c] - lo) / (hi - lo), 0.0, 1.0)
    return out


def load_highres(processed_root: Path, patch_id: str, month: str) -> np.ndarray | None:
    root = processed_root / "patches" / "highres_optical"
    files = sorted(root.glob(f"*{patch_id}.tif"))
    month_files = [p for p in files if f"_{month}" in p.name or f"_{month[:4]}{month[4:]}" in p.name]
    path = (month_files or files)[-1] if files else None
    if path is None:
        return None
    with rasterio.open(path) as src:
        return stretch_rgb(src.read())


def load_landcover_metadata(label_root: Path) -> tuple[list[str], ListedColormap, list[str]]:
    meta_path = label_root / "osm_landcover" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names_by_id = {int(k): v for k, v in meta["class_names"].items()}
    colors_by_id = {int(k): v for k, v in meta["class_colors"].items()}
    max_id = max(names_by_id)
    names = ["unknown"] * (max_id + 1)
    colors = ["#ffffff"] * (max_id + 1)
    for class_id, name in names_by_id.items():
        names[class_id] = name
    for class_id, color in colors_by_id.items():
        colors[class_id] = color
    return names, ListedColormap(colors), colors


def load_layout(mask_dir: Path) -> tuple[list[PatchLayout], int, int, int, int]:
    mask_paths = sorted(mask_dir.glob("*.tif"))
    records: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    for path in mask_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
        patch_id = patch_id_from_path(path)
        records.append((patch_id, path, float(bounds.left), float(bounds.top), width, height))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(patch_id, row_of[round(top, 3)], col_of[round(left, 3)], path)
        for patch_id, path, left, top, _w, _h in records
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def label_to_rgb(label: np.ndarray, colors: list[str]) -> np.ndarray:
    rgb = np.ones((*label.shape, 3), dtype=np.uint8) * 255
    for class_id, color in enumerate(colors):
        if class_id >= len(colors):
            continue
        color = color.lstrip("#")
        if len(color) != 6:
            continue
        rgb[label == class_id] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


def red_binary(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = (235, 20, 25)
    return rgb


def save_geo_canvas(
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    output_path: Path,
    title: str,
    renderer: Any,
) -> None:
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 238
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1)
        tile = renderer(mask)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(canvas)
    header = 72
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), title, fill=(0, 0, 0))
    out.paste(image, (0, header))
    out.save(output_path)


def summarize_mask_dir(mask_dir: Path, names: list[str] | None = None) -> dict[str, Any]:
    value_pixels: dict[int, int] = {}
    positive_patches: dict[int, int] = {}
    total_pixels = 0
    files = sorted(mask_dir.glob("*.tif"))
    for path in files:
        with rasterio.open(path) as src:
            arr = src.read(1)
        values, counts = np.unique(arr, return_counts=True)
        total_pixels += int(arr.size)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            value = int(value)
            value_pixels[value] = value_pixels.get(value, 0) + int(count)
            if count > 0:
                positive_patches[value] = positive_patches.get(value, 0) + 1
    classes = []
    for value in sorted(value_pixels):
        classes.append(
            {
                "class_id": value,
                "name": names[value] if names is not None and value < len(names) else str(value),
                "pixels": value_pixels[value],
                "pixel_ratio": value_pixels[value] / max(total_pixels, 1),
                "positive_patches": positive_patches.get(value, 0),
                "positive_patch_ratio": positive_patches.get(value, 0) / max(len(files), 1),
            }
        )
    return {"files": len(files), "total_pixels": total_pixels, "classes": classes}


def sample_patch_ids(
    construction_mask_dir: Path,
    landcover_mask_dir: Path,
    requested: list[str] | None,
    random_samples: int,
    seed: int,
) -> list[str]:
    if requested:
        return requested
    scores: list[tuple[int, str]] = []
    for path in sorted(construction_mask_dir.glob("*.tif")):
        with rasterio.open(path) as src:
            arr = src.read(1)
        if int((arr > 0).sum()) > 0:
            scores.append((int((arr > 0).sum()), patch_id_from_path(path)))
    scores.sort(reverse=True)
    selected = [patch_id for _score, patch_id in scores[:4]]
    land_ids = [patch_id_from_path(p) for p in sorted(landcover_mask_dir.glob("*.tif"))]
    rng = np.random.default_rng(seed)
    selected.extend(rng.choice(land_ids, size=min(random_samples, len(land_ids)), replace=False).tolist())
    return sorted(dict.fromkeys(selected))


def save_patch_sheet(
    processed_root: Path,
    embedding_root: Path,
    patch_id: str,
    month: str,
    construction_dir: Path,
    landcover_dir: Path,
    out_dir: Path,
    names: list[str],
    colors: list[str],
) -> Path:
    highres = load_highres(processed_root, patch_id, month)
    emb_path = embedding_root / "haidian" / patch_id / f"{month}_embedding_map.pt"
    pca_rgb = None
    if emb_path.exists():
        import torch
        from sklearn.decomposition import PCA

        emb = torch.load(emb_path, map_location="cpu", weights_only=True).float().numpy()
        flat = emb.reshape(emb.shape[0], -1).T
        rgb = PCA(n_components=3, random_state=42).fit_transform(flat).reshape(emb.shape[1], emb.shape[2], 3)
        lo = np.percentile(rgb.reshape(-1, 3), 2, axis=0)
        hi = np.percentile(rgb.reshape(-1, 3), 98, axis=0)
        pca_rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    land_path = next(iter(sorted(landcover_dir.glob(f"*{patch_id}.tif"))), None)
    cons_path = next(iter(sorted(construction_dir.glob(f"*{patch_id}.tif"))), None)
    land = None
    cons = None
    if land_path is not None:
        with rasterio.open(land_path) as src:
            land = src.read(1)
    if cons_path is not None:
        with rasterio.open(cons_path) as src:
            cons = src.read(1)

    panels: list[tuple[str, np.ndarray | None, str]] = [
        (f"High-res {month}", highres, "rgb"),
        ("Embedding PCA", pca_rgb, "rgb"),
        ("OSM land-use", land, "land"),
        ("Construction GT", cons, "binary"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.5))
    cmap = ListedColormap(colors)
    for ax, (title, image, kind) in zip(axes, panels, strict=True):
        if image is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        elif kind == "land":
            ax.imshow(image, cmap=cmap, vmin=0, vmax=len(colors) - 1, interpolation="nearest")
        elif kind == "binary":
            ax.imshow(image, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    legend = "\n".join(f"{i}: {name}" for i, name in enumerate(names))
    fig.text(0.83, 0.1, legend, fontsize=8, family="monospace")
    fig.suptitle(patch_id, fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.82, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patch_id}_audit_sheet.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    label_root = args.processed_root / "labels"
    construction_dir = label_root / "construction" / "masks"
    landcover_dir = label_root / "osm_landcover" / "masks"
    names, _cmap, colors = load_landcover_metadata(label_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    land_layouts, rows, cols, tile_h, tile_w = load_layout(landcover_dir)
    save_geo_canvas(
        land_layouts,
        rows,
        cols,
        tile_h,
        tile_w,
        args.output_root / "landuse_gt_320patch_geo.png",
        "Haidian OSM land-use labels, 320 patches",
        lambda mask: label_to_rgb(mask, colors),
    )

    cons_layouts, cons_rows, cons_cols, cons_h, cons_w = load_layout(construction_dir)
    save_geo_canvas(
        cons_layouts,
        cons_rows,
        cons_cols,
        cons_h,
        cons_w,
        args.output_root / "construction_gt_labeled_patches_geo.png",
        "Haidian manual construction-site labels, labeled patches only",
        red_binary,
    )

    summary = {
        "construction": summarize_mask_dir(construction_dir),
        "osm_landcover": summarize_mask_dir(landcover_dir, names=names),
        "figures": {
            "landuse_gt_320patch_geo": str(args.output_root / "landuse_gt_320patch_geo.png"),
            "construction_gt_labeled_patches_geo": str(
                args.output_root / "construction_gt_labeled_patches_geo.png"
            ),
        },
    }
    patch_ids = sample_patch_ids(
        construction_dir,
        landcover_dir,
        args.sample_patches,
        args.random_samples,
        args.seed,
    )
    sheets = []
    for patch_id in patch_ids:
        sheets.append(
            str(
                save_patch_sheet(
                    args.processed_root,
                    args.embedding_root,
                    patch_id,
                    args.month,
                    construction_dir,
                    landcover_dir,
                    args.output_root / "patch_sheets",
                    names,
                    colors,
                )
            )
        )
    summary["patch_sheets"] = sheets
    (args.output_root / "label_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_root / "label_audit_summary.json")


if __name__ == "__main__":
    main()
