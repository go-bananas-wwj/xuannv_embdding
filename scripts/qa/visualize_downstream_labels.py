#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio

MONTH_SUFFIX_RE = re.compile(r"^(?P<patch>.+)_(?P<month>\d{6})$")


def valid_yyyymm(value: str) -> bool:
    year = int(value[:4])
    month = int(value[4:])
    return 1900 <= year <= 2100 and 1 <= month <= 12


def base_patch_id(mask_path: Path) -> str:
    match = MONTH_SUFFIX_RE.match(mask_path.stem)
    if match is not None and valid_yyyymm(match.group("month")):
        return match.group("patch")
    return mask_path.stem


def mask_month(mask_path: Path) -> str | None:
    match = MONTH_SUFFIX_RE.match(mask_path.stem)
    if match is not None and valid_yyyymm(match.group("month")):
        return match.group("month")
    return None


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


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(np.int64)
    if image.ndim == 2:
        return image[y_idx][:, x_idx]
    return image[y_idx][:, x_idx, :]


def choose_file(files: list[Path], preferred_month: str | None) -> Path | None:
    files = [path for path in files if not path.name.endswith("_mask.tif")]
    if not files:
        return None
    if preferred_month:
        candidates = [
            path for path in files if re.search(rf"_{preferred_month}(?:\d{{2}})?_", path.name)
        ]
        if candidates:
            return sorted(candidates)[-1]
    return sorted(files)[-1]


def find_highres(data_root: Path, patch_id: str, preferred_month: str | None) -> Path | None:
    root = data_root / "patches" / "highres_optical"
    return choose_file(sorted(root.glob(f"*{patch_id}.tif")), preferred_month)


def find_s2(data_root: Path, patch_id: str, preferred_month: str | None) -> Path | None:
    root = data_root / "patches" / "s2"
    return choose_file(sorted(root.glob(f"*{patch_id}.tif")), preferred_month)


def load_rgb(path: Path | None, source: str) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if path is None or not path.exists():
        return None, None
    with rasterio.open(path) as src:
        arr = src.read()
        meta = {
            "path": str(path),
            "shape": [src.count, src.height, src.width],
            "crs": str(src.crs) if src.crs else None,
            "transform": [float(v) for v in src.transform[:6]],
            "bounds": [float(v) for v in src.bounds],
        }
    if source == "s2" and arr.shape[0] >= 4:
        rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)
    else:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    return stretch(rgb), meta


def read_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = {
            "path": str(path),
            "shape": [src.height, src.width],
            "crs": str(src.crs) if src.crs else None,
            "transform": [float(v) for v in src.transform[:6]],
            "bounds": [float(v) for v in src.bounds],
            "positive_pixels": int((arr > 0).sum()),
            "positive_fraction": float((arr > 0).mean()),
        }
    return arr, meta


def overlay_mask(image: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if image is None:
        return None
    image = resize_nearest(image, mask.shape)
    out = image.copy()
    positive = mask > 0
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=-1)
    color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[positive] = 0.45 * out[positive] + 0.55 * color
    return np.clip(out, 0.0, 1.0)


def select_masks(mask_dir: Path, top_k: int, random_k: int, seed: int) -> list[Path]:
    rows: list[tuple[int, Path]] = []
    for path in sorted(mask_dir.glob("*.tif")):
        with rasterio.open(path) as src:
            mask = src.read(1)
        positives = int((mask > 0).sum())
        if positives > 0:
            rows.append((positives, path))
    rows.sort(key=lambda item: item[0], reverse=True)
    selected = [path for _, path in rows[:top_k]]
    remaining = [path for _, path in rows[top_k:]]
    if random_k > 0 and remaining:
        rng = np.random.default_rng(seed)
        take = min(random_k, len(remaining))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return selected


def make_figure(
    task: str,
    mask_path: Path,
    data_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    patch_id = base_patch_id(mask_path)
    preferred_month = mask_month(mask_path)
    mask, mask_meta = read_mask(mask_path)
    highres_path = find_highres(data_root, patch_id, preferred_month)
    s2_path = find_s2(data_root, patch_id, preferred_month)
    highres, highres_meta = load_rgb(highres_path, "highres")
    s2, s2_meta = load_rgb(s2_path, "s2")

    highres_overlay = overlay_mask(highres, mask)
    s2_overlay = overlay_mask(s2, mask)
    panels = [
        ("High-res", resize_nearest(highres, mask.shape) if highres is not None else None),
        ("High-res + mask", highres_overlay),
        ("S2 RGB", resize_nearest(s2, mask.shape) if s2 is not None else None),
        ("S2 + mask", s2_overlay),
        ("Mask", mask),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.5))
    for ax, (title, image) in zip(axes, panels, strict=True):
        if image is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        elif title == "Mask":
            ax.imshow(image, cmap="Reds", vmin=0, vmax=max(1, int(mask.max())))
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{task} | {mask_path.stem} | positives={mask_meta['positive_pixels']}",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{mask_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": task,
        "patch_id": patch_id,
        "mask": mask_meta,
        "highres": highres_meta,
        "s2": s2_meta,
        "figure": str(out_path),
        "notes": [
            "Mask files currently carry no CRS/real-world transform when crs is null.",
            "Overlay is therefore a pixel-grid QA using matching patch ids.",
        ],
    }


def write_index(records: list[dict[str, Any]], out_dir: Path) -> None:
    index_path = out_dir / "index.md"
    lines = ["# Downstream Label QA", ""]
    for rec in records:
        fig = Path(rec["figure"])
        mask = rec["mask"]
        lines.extend(
            [
                f"## {rec['task']} / {Path(mask['path']).stem}",
                "",
                f"- positive_pixels: `{mask['positive_pixels']}`",
                f"- positive_fraction: `{mask['positive_fraction']:.6f}`",
                f"- mask_crs: `{mask['crs']}`",
                f"- mask_bounds: `{mask['bounds']}`",
                f"- highres: `{rec['highres']['path'] if rec['highres'] else None}`",
                f"- s2: `{rec['s2']['path'] if rec['s2'] else None}`",
                "",
                f"![{fig.name}]({fig.name})",
                "",
            ]
        )
    index_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--random-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    masks = select_masks(args.mask_dir, args.top_k, args.random_k, args.seed)
    if not masks:
        raise SystemExit(f"No positive masks found in {args.mask_dir}")
    records = [
        make_figure(args.task, mask_path, args.data_root, args.out_dir)
        for mask_path in masks
    ]
    write_index(records, args.out_dir)
    print(f"saved {len(records)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
