#!/usr/bin/env python3
"""可视化 patch：RGB、高分、SAR、Landsat、WorldCover、embedding PCA、下游预测与 GT mask。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter, zoom
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_s2_rgb(patch_id: str, root: Path) -> np.ndarray | None:
    files = sorted(root.glob(f"s2_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    im = np.nan_to_num(im, nan=0.0)
    rgb = np.stack([im[3], im[2], im[1]], axis=-1)  # R/G/B bands 3,2,1
    return _stretch(rgb)


def _load_highres_optical(patch_id: str, root: Path, month: str | None = None) -> np.ndarray | None:
    if month:
        files = sorted(root.glob(f"highres_optical_{month}*{patch_id}.tif"))
    else:
        files = sorted(root.glob(f"highres_optical_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    rgb = np.transpose(im[:3], (1, 2, 0))
    return _stretch(rgb)


def _load_highres_sar(patch_id: str, root: Path) -> np.ndarray | None:
    files = sorted(root.glob(f"highres_sar_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read(1)
    im = np.nan_to_num(im, nan=0.0)
    return _stretch(im)


def _load_s1(patch_id: str, root: Path) -> np.ndarray | None:
    files = sorted(root.glob(f"s1_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read(1)
    im = np.nan_to_num(im, nan=0.0)
    return _stretch(im)


def _load_landsat(patch_id: str, root: Path) -> np.ndarray | None:
    files = sorted(root.glob(f"landsat_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    im = np.nan_to_num(im, nan=0.0)
    rgb = np.stack([im[4], im[3], im[2]], axis=-1) if im.shape[0] >= 5 else np.transpose(im[:3], (1, 2, 0))
    return _stretch(rgb)


def _load_worldcover(patch_id: str, root: Path) -> np.ndarray | None:
    path = root / f"worldcover_20230101_{patch_id}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def _stretch(im: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    im = np.nan_to_num(im, nan=0.0)
    if im.ndim == 2:
        lo, hi = np.percentile(im, [lower, upper])
        if hi <= lo:
            return np.zeros_like(im)
        return np.clip((im - lo) / (hi - lo), 0, 1)
    out = np.zeros_like(im, dtype=np.float32)
    for c in range(im.shape[-1]):
        band = im[..., c]
        lo, hi = np.percentile(band, [lower, upper])
        if hi > lo:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """简单最近邻下采样到目标 shape (H, W)。"""
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    if arr.ndim == 3:
        factors = factors + (1,)
    return zoom(arr, factors, order=0)


def _compute_embedding_pca(emb: np.ndarray, n_components: int = 3, smooth_sigma: float = 0.0) -> np.ndarray:
    """emb: (C, H, W) -> (H, W, 3)"""
    c, h, w = emb.shape
    flat = emb.reshape(c, -1).T
    flat = flat - flat.mean(axis=0)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(flat)
    pcs = pcs.reshape(h, w, n_components)
    rgb = _stretch(pcs)
    if smooth_sigma > 0:
        rgb = gaussian_filter(rgb, sigma=(smooth_sigma, smooth_sigma, 0))
    return rgb


def _worldcover_cmap():
    colors = [
        "#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000",
        "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#000000"
    ]
    return mcolors.ListedColormap(colors)


def _load_prediction(patch_id: str, pred_dir: Path) -> np.ndarray | None:
    path = pred_dir / f"{patch_id}_prob.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def _load_mask(patch_id: str, mask_dir: Path) -> np.ndarray | None:
    path = mask_dir / f"{patch_id}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def visualize_patch(
    patch_id: str,
    emb_root: Path,
    data_root: Path,
    pred_dir: Path | None,
    mask_dir: Path,
    out_dir: Path,
    month: str = "202604",
    pca_smooth_sigma: float = 0.0,
) -> None:
    emb_path = emb_root / patch_id / f"{month}_embedding_map.pt"
    if not emb_path.exists():
        logger.warning("embedding 不存在: %s", emb_path)
        return
    emb = torch.load(emb_path, map_location="cpu", weights_only=True).numpy()
    pca_rgb = _compute_embedding_pca(emb, smooth_sigma=pca_smooth_sigma)

    s2_root = data_root / "patches" / "s2"
    highres_root = data_root / "patches" / "highres_optical"
    sar_root = data_root / "patches" / "highres_sar"
    s1_root = data_root / "patches" / "s1"
    landsat_root = data_root / "patches" / "landsat"
    wc_root = data_root / "labels" / "worldcover"

    s2_rgb = _load_s2_rgb(patch_id, s2_root)
    highres_rgb = _load_highres_optical(patch_id, highres_root)
    sar = _load_highres_sar(patch_id, sar_root)
    s1 = _load_s1(patch_id, s1_root)
    landsat = _load_landsat(patch_id, landsat_root)
    wc = _load_worldcover(patch_id, wc_root)
    pred = _load_prediction(patch_id, pred_dir) if pred_dir else None
    mask = _load_mask(patch_id, mask_dir)

    h, w = 128, 128
    top_images, top_titles = [], []
    bottom_images, bottom_titles = [], []
    if s2_rgb is not None:
        top_images.append(s2_rgb); top_titles.append("S2 RGB")
    if highres_rgb is not None:
        top_images.append(_resize_to(highres_rgb, (h, w))); top_titles.append("High-res Optical")
    if sar is not None:
        top_images.append(_resize_to(sar, (h, w))); top_titles.append("High-res SAR")
    if s1 is not None:
        top_images.append(_resize_to(s1, (h, w))); top_titles.append("S1 VV/VH")
    if landsat is not None:
        top_images.append(_resize_to(landsat, (h, w))); top_titles.append("Landsat FalseColor")
    if wc is not None:
        bottom_images.append(_resize_to(wc, (h, w))); bottom_titles.append("WorldCover")
    bottom_images.append(pca_rgb); bottom_titles.append("Embedding PCA")
    if pred is not None:
        bottom_images.append(pred); bottom_titles.append("Prediction Prob")
    if mask is not None:
        bottom_images.append(mask); bottom_titles.append("GT Mask")

    n_top = len(top_images)
    n_bottom = len(bottom_images)
    n_cols = max(n_top, n_bottom)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.0 * n_cols, 6.2))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    axes = np.atleast_2d(axes)

    def _show(ax, img, title):
        if title == "WorldCover":
            ax.imshow(img, cmap=_worldcover_cmap(), vmin=0, vmax=10)
        elif title in ("Prediction Prob", "GT Mask"):
            ax.imshow(img, cmap="jet", vmin=0, vmax=1)
        elif img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for idx, (img, title) in enumerate(zip(top_images, top_titles)):
        _show(axes[0, idx], img, title)
    for idx in range(n_top, n_cols):
        axes[0, idx].axis("off")
    for idx, (img, title) in enumerate(zip(bottom_images, bottom_titles)):
        _show(axes[1, idx], img, title)
    for idx in range(n_bottom, n_cols):
        axes[1, idx].axis("off")

    fig.suptitle(f"Patch {patch_id}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patch_id}_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--emb-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="processed/<region>")
    p.add_argument("--pred-dir", type=Path, default=None, help="含 *_prob.tif 的目录")
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--patch-ids", nargs="+", default=None)
    p.add_argument("--all-labeled", action="store_true", help="可视化 mask-dir 下所有有标注的 patch")
    p.add_argument("--n-samples", type=int, default=None, help="在 --all-labeled 时随机选 N 张")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--month", default="202604")
    p.add_argument("--pca-smooth-sigma", type=float, default=0.0, help="PCA 可视化高斯平滑 sigma（0 表示不平滑）")
    args = p.parse_args()

    if args.all_labeled:
        patch_ids = sorted([p.stem for p in args.mask_dir.glob("*.tif")])
        if args.n_samples and args.n_samples < len(patch_ids):
            rng = np.random.default_rng(seed=42)
            patch_ids = sorted(rng.choice(patch_ids, size=args.n_samples, replace=False).tolist())
    elif args.patch_ids:
        patch_ids = args.patch_ids
    else:
        p.error("请提供 --patch-ids 或 --all-labeled")

    for pid in patch_ids:
        visualize_patch(
            pid,
            args.emb_root,
            args.data_root,
            args.pred_dir,
            args.mask_dir,
            args.out_dir,
            args.month,
            args.pca_smooth_sigma,
        )


if __name__ == "__main__":
    main()
