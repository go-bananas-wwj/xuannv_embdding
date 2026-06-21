#!/usr/bin/env python3
"""
为同一 patch 生成不同版本 embedding 的可视化对比：
- 每个版本单独一张 comprehensive patch 可视化（类似 harbin_stage2_v1_fold0 风格）
- 一张汇总对比图：各版本 Embedding PCA + Downstream Prediction 并排
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("/root/workspace/report/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARBIN_DATA = Path("/data/xuannv_embedding/processed/harbin")
HARBIN_MASK = Path("/data/xuannv_embedding/processed/harbin/labels/construction/masks")

# 选定用于对比的 patch（这些 patch 在四个版本 embedding 与三个下游预测中均存在）
SELECTED_PATCHES = ["patch_000146", "patch_000256", "patch_000181", "patch_000193"]

VERSIONS = {
    "Stage2_V1": {
        "emb_root": Path("/data/xuannv_embedding/embeddings/harbin"),
        "pred_dirs": [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_{i}/predictions") for i in range(5)],
        "month": "202512",
    },
    "V1.1_Quick": {
        "emb_root": Path("/data/xuannv_embedding/embeddings/v1.1_quick_harbin_labeled/20260621_v1.1_extract_labeled_harbin_best/harbin"),
        "pred_dirs": [Path("/data/xuannv_embedding/experiments/v1.1_quick/construction_harbin/fold_0/predictions")],
        "month": "202512",
    },
    "AEF_Official": {
        "emb_root": Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual/harbin"),
        "pred_dirs": [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)],
        "month": "202512",
    },
    "AEF_Teacher": {
        "emb_root": Path("/data/xuannv_embedding/embeddings/aef_teacher_2025_annual/harbin"),
        "pred_dirs": [],  # 无对应下游预测，仅展示 embedding
        "month": "202512",
    },
}


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


def _load_s2_rgb(patch_id: str, month: str) -> np.ndarray | None:
    s2_root = HARBIN_DATA / "patches" / "s2"
    files = sorted(s2_root.glob(f"s2_{month}*{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    im = np.nan_to_num(im, nan=0.0)
    rgb = np.stack([im[3], im[2], im[1]], axis=-1)
    return _stretch(rgb)


def _load_s1(patch_id: str, month: str) -> np.ndarray | None:
    s1_root = HARBIN_DATA / "patches" / "s1"
    files = sorted(s1_root.glob(f"s1_{month}*{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read(1)
    return _stretch(np.nan_to_num(im, nan=0.0))


def _load_landsat(patch_id: str, month: str) -> np.ndarray | None:
    landsat_root = HARBIN_DATA / "patches" / "landsat"
    files = sorted(landsat_root.glob(f"landsat_{month}*{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    im = np.nan_to_num(im, nan=0.0)
    rgb = np.stack([im[4], im[3], im[2]], axis=-1) if im.shape[0] >= 5 else np.transpose(im[:3], (1, 2, 0))
    return _stretch(rgb)


def _load_worldcover(patch_id: str) -> np.ndarray | None:
    wc_root = HARBIN_DATA / "labels" / "worldcover"
    path = wc_root / f"worldcover_20230101_{patch_id}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def _load_mask(patch_id: str, time: str) -> np.ndarray | None:
    path = HARBIN_MASK / f"{patch_id}_{time}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def _load_pred(patch_id: str, pred_dirs: list[Path]) -> np.ndarray | None:
    for d in pred_dirs:
        path = d / f"{patch_id}_prob.tif"
        if path.exists():
            with rasterio.open(path) as src:
                return src.read(1)
    return None


def _compute_embedding_pca(patch_id: str, emb_root: Path, month: str) -> np.ndarray | None:
    path = emb_root / patch_id / f"{month}_embedding_map.pt"
    if not path.exists():
        return None
    emb = torch.load(path, map_location="cpu", weights_only=True).numpy()
    c, h, w = emb.shape
    flat = emb.reshape(c, -1).T
    flat = flat - flat.mean(axis=0)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(flat)
    pcs = pcs.reshape(h, w, 3)
    return _stretch(pcs)


def _show(ax, img, title, cmap=None, vmin=None, vmax=None):
    if img is None:
        ax.axis("off")
        ax.set_title(title, fontsize=9)
        return
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    elif img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def generate_comprehensive(patch_id: str, version_name: str, emb_root: Path, month: str, pred_dirs: list[Path]) -> Path:
    """生成类似 harbin_stage2_v1_fold0 风格的单版本 comprehensive 图。"""
    s2 = _load_s2_rgb(patch_id, month)
    s1 = _load_s1(patch_id, month)
    landsat = _load_landsat(patch_id, month)
    wc = _load_worldcover(patch_id)
    emb_pca = _compute_embedding_pca(patch_id, emb_root, month)
    pred = _load_pred(patch_id, pred_dirs)
    mask_t1 = _load_mask(patch_id, "202512")
    mask_t2 = _load_mask(patch_id, "202605")
    mask_t1_arr = mask_t1 if mask_t1 is not None else np.zeros((128, 128), dtype=np.uint8)
    mask_t2_arr = mask_t2 if mask_t2 is not None else np.zeros((128, 128), dtype=np.uint8)
    mask = np.maximum(mask_t1_arr, mask_t2_arr)

    top_images = [s2, s1, landsat]
    top_titles = ["S2 RGB", "S1 VV/VH", "Landsat FalseColor"]
    bottom_images = [wc, emb_pca, pred if pred is not None else mask]
    bottom_titles = ["WorldCover", "Embedding PCA", "Prediction Prob" if pred is not None else "GT Mask"]

    n_cols = 3
    fig, axes = plt.subplots(2, n_cols, figsize=(9.6, 6.4))
    for idx, (img, title) in enumerate(zip(top_images, top_titles)):
        _show(axes[0, idx], img, title)
    for idx, (img, title) in enumerate(zip(bottom_images, bottom_titles)):
        if title == "WorldCover":
            _show(axes[1, idx], img, title)
        elif title in ("Prediction Prob", "GT Mask"):
            _show(axes[1, idx], img, title, cmap="jet", vmin=0, vmax=1)
        else:
            _show(axes[1, idx], img, title)

    safe_name = version_name.replace(".", "_").replace(" ", "_")
    fig.suptitle(f"{version_name} - {patch_id}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"comprehensive_{safe_name}_{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("saved %s", out_path)
    return out_path


def generate_version_comparison_grid(patch_id: str) -> Path:
    """生成各版本 Embedding PCA + Prediction 的汇总对比图。"""
    s2_t1 = _load_s2_rgb(patch_id, "202512")
    s2_t2 = _load_s2_rgb(patch_id, "202605")
    mask_t1 = _load_mask(patch_id, "202512")
    mask_t2 = _load_mask(patch_id, "202605")

    n_versions = len(VERSIONS)
    # 每行 4 列：1 行参考 + n_versions 行版本
    n_rows = 1 + n_versions
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 2.8 * n_rows))

    # 参考行 0
    _show(axes[0, 0], s2_t1, "S2 202512")
    _show(axes[0, 1], s2_t2, "S2 202605")
    _show(axes[0, 2], mask_t1, "GT 202512", cmap="jet", vmin=0, vmax=1)
    _show(axes[0, 3], mask_t2, "GT 202605", cmap="jet", vmin=0, vmax=1)

    # 版本行
    for row_idx, (version_name, cfg) in enumerate(VERSIONS.items(), start=1):
        emb = _compute_embedding_pca(patch_id, cfg["emb_root"], cfg["month"])
        pred = _load_pred(patch_id, cfg["pred_dirs"])
        axes[row_idx, 0].text(0.5, 0.5, version_name, ha="center", va="center", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        _show(axes[row_idx, 1], emb, f"{version_name} Embedding PCA")
        if pred is not None:
            _show(axes[row_idx, 2], pred, f"{version_name} Prediction", cmap="jet", vmin=0, vmax=1)
        else:
            _show(axes[row_idx, 2], mask_t1, "GT 202512 (no pred)", cmap="jet", vmin=0, vmax=1)
        # 第 4 列放 GT Mask 作为参照
        _show(axes[row_idx, 3], mask_t1, "GT 202512", cmap="jet", vmin=0, vmax=1)

    fig.suptitle(f"Embedding Version Comparison - {patch_id}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / f"embedding_version_compare_{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("saved %s", out_path)
    return out_path


def main() -> None:
    # 为每个 patch 生成版本对比汇总图（包含各版本 Embedding PCA 与下游 Prediction）
    for patch_id in SELECTED_PATCHES:
        generate_version_comparison_grid(patch_id)


if __name__ == "__main__":
    main()
