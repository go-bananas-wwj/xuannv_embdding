#!/usr/bin/env python3
"""Diagnose spatial drift in one exported embedding patch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-id", required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument(
        "--embedding-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32, copy=False)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return 0.0
    a = a[valid]
    b = b[valid]
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spatial_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx - xx.mean()) / max(float(xx.std()), 1.0)
    yy = (yy - yy.mean()) / max(float(yy.std()), 1.0)
    return xx, yy


def quadrant_stats(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.ndim == 3:
        # C,H,W -> H,W,C
        arr2 = np.moveaxis(arr, 0, -1)
    else:
        arr2 = arr[..., None]
    h, w = arr2.shape[:2]
    parts = {
        "top_left": arr2[: h // 2, : w // 2],
        "top_right": arr2[: h // 2, w // 2 :],
        "bottom_left": arr2[h // 2 :, : w // 2],
        "bottom_right": arr2[h // 2 :, w // 2 :],
    }
    return {name: np.nanmean(value.reshape(-1, arr2.shape[-1]), axis=0).tolist() for name, value in parts.items()}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def quadrant_cosine_distances(embedding: np.ndarray) -> dict[str, float]:
    means = quadrant_stats(embedding)
    keys = list(means)
    out: dict[str, float] = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            a = np.asarray(means[left], dtype=np.float64)
            b = np.asarray(means[right], dtype=np.float64)
            out[f"{left}_vs_{right}"] = 1.0 - cosine(a, b)
    return out


def source_files(root: Path, source: str, patch_id: str, month: str) -> list[Path]:
    source_root = root / "patches" / source
    if month:
        return sorted(source_root.glob(f"{source}_{month}*_{patch_id}.tif"))
    return sorted(source_root.glob(f"{source}_*_{patch_id}.tif"))


def source_summary(path: Path) -> dict[str, Any]:
    arr = read_tif(path)
    finite = np.isfinite(arr)
    valid = finite & (arr != 0)
    mean_image = np.nanmean(arr, axis=0)
    xx, yy = spatial_grids(mean_image.shape[0], mean_image.shape[1])
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "zero_ratio": float((arr == 0).mean()),
        "finite_ratio": float(finite.mean()),
        "valid_ratio": float(valid.mean()),
        "brightness_corr_x": corr(mean_image, xx),
        "brightness_corr_y": corr(mean_image, yy),
        "quadrant_mean": quadrant_stats(mean_image),
    }


def embedding_summary(embedding_path: Path) -> dict[str, Any]:
    emb = torch.load(embedding_path, map_location="cpu", weights_only=True).float().numpy()
    channels, height, width = emb.shape
    flat = emb.reshape(channels, -1).T
    pca = PCA(n_components=3)
    coords = pca.fit_transform(flat).reshape(height, width, 3)
    xx, yy = spatial_grids(height, width)
    channel_corr_x = [corr(emb[i], xx) for i in range(channels)]
    channel_corr_y = [corr(emb[i], yy) for i in range(channels)]
    pc_corr = {
        f"pc{i + 1}_corr_x": corr(coords[..., i], xx)
        for i in range(3)
    } | {
        f"pc{i + 1}_corr_y": corr(coords[..., i], yy)
        for i in range(3)
    }
    top_spatial_channels = sorted(
        [
            {
                "channel": i,
                "corr_x": channel_corr_x[i],
                "corr_y": channel_corr_y[i],
                "max_abs_corr": max(abs(channel_corr_x[i]), abs(channel_corr_y[i])),
            }
            for i in range(channels)
        ],
        key=lambda item: item["max_abs_corr"],
        reverse=True,
    )[:10]
    norm = np.linalg.norm(emb, axis=0)
    return {
        "path": str(embedding_path),
        "shape": list(emb.shape),
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std()),
        "channel_mean_std": float(emb.mean(axis=(1, 2)).std()),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc_spatial_correlation": pc_corr,
        "top_spatial_channels": top_spatial_channels,
        "quadrant_cosine_distance": quadrant_cosine_distances(emb),
    }


def main() -> None:
    args = parse_args()
    patch_id = args.patch_id
    if not re.match(r"patch_\d{6}$", patch_id):
        raise ValueError("--patch-id should look like patch_000183")

    embedding_path = args.embedding_root / "haidian" / patch_id / f"{args.month}_embedding_map.pt"
    report: dict[str, Any] = {
        "patch_id": patch_id,
        "month": args.month,
        "embedding": embedding_summary(embedding_path),
        "sources": {},
        "notes": [
            "High absolute PC/spatial correlation suggests low-frequency spatial drift in the embedding.",
            "High zero_ratio or strong brightness quadrant differences in optical inputs can be cloud/nodata/mosaic artifacts.",
        ],
    }
    for source in ("highres_optical", "s2", "landsat", "s1"):
        files = source_files(args.processed_root, source, patch_id, args.month)
        if not files and source == "highres_optical":
            files = source_files(args.processed_root, source, patch_id, "")
        report["sources"][source] = [source_summary(path) for path in files]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
