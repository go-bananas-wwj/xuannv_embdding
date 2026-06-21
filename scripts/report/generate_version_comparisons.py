#!/usr/bin/env python3
"""生成多版本 / 多模型 / AEF 同 patch 对比可视化图。"""
from __future__ import annotations

import logging
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


def _load_s2_rgb(patch_id: str, root: Path, month: str | None = None) -> np.ndarray | None:
    """root: processed/<region>/patches/s2"""
    if month:
        files = sorted(root.glob(f"s2_{month}*{patch_id}.tif"))
    else:
        files = sorted(root.glob(f"s2_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    im = np.nan_to_num(im, nan=0.0)
    rgb = np.stack([im[3], im[2], im[1]], axis=-1)
    return _stretch(rgb)


def _load_s1(patch_id: str, root: Path, month: str | None = None) -> np.ndarray | None:
    """root: processed/<region>/patches/s1"""
    if month:
        files = sorted(root.glob(f"s1_{month}*{patch_id}.tif"))
    else:
        files = sorted(root.glob(f"s1_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read(1)
    return _stretch(np.nan_to_num(im, nan=0.0))


def _load_input(patch_id: str, data_root: Path, month: str) -> tuple[np.ndarray | None, str]:
    """优先读取 S2 RGB，缺失则回退到 S1 VV/VH。返回 (image, title)。"""
    s2_root = data_root / "patches" / "s2"
    s1_root = data_root / "patches" / "s1"
    s2 = _load_s2_rgb(patch_id, s2_root, month=month)
    if s2 is not None:
        return s2, f"S2 {month}"
    s1 = _load_s1(patch_id, s1_root, month=month)
    if s1 is not None:
        return s1, f"S1 {month}"
    return None, f"S2 {month}"


def _load_highres_optical(patch_id: str, root: Path) -> np.ndarray | None:
    files = sorted(root.glob(f"highres_optical_*_{patch_id}.tif"))
    if not files:
        return None
    with rasterio.open(files[-1]) as src:
        im = src.read()
    rgb = np.transpose(im[:3], (1, 2, 0))
    return _stretch(rgb)


def _load_mask(patch_id: str, mask_dir: Path, suffix: str = ".tif") -> np.ndarray | None:
    path = mask_dir / f"{patch_id}{suffix}"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


def _load_pred(patch_id: str, pred_dir: Path | list[Path]) -> np.ndarray | None:
    dirs = [pred_dir] if isinstance(pred_dir, Path) else pred_dir
    for d in dirs:
        path = d / f"{patch_id}_prob.tif"
        if path.exists():
            with rasterio.open(path) as src:
                return src.read(1)
    return None


def _load_embedding_pca(patch_id: str, emb_root: Path, month: str = "202604") -> np.ndarray | None:
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
        ax.set_title(title, fontsize=10)
        return
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    elif img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_haidian_construction_versions(patch_id: str, data_root: Path, mask_dir: Path,
                                        versions: list[tuple[str, Path | list[Path]]],
                                        emb_root: Path, month: str = "202505") -> None:
    """Haidian construction: S2, high-res, GT, embedding + v1/v2/v3 predictions."""
    s2_root = data_root / "patches" / "s2"
    highres_root = data_root / "patches" / "highres_optical"

    s2 = _load_s2_rgb(patch_id, s2_root, month=month)
    highres = _load_highres_optical(patch_id, highres_root)
    mask = _load_mask(patch_id, mask_dir)
    emb_pca = _load_embedding_pca(patch_id, emb_root, month=month)

    n_versions = len(versions)
    n_cols = max(4, n_versions)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.2))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    axes = np.atleast_2d(axes)

    _show(axes[0, 0], s2, "S2 RGB")
    _show(axes[0, 1], highres, "High-res Optical")
    _show(axes[0, 2], mask, "GT Mask", cmap="jet", vmin=0, vmax=1)
    _show(axes[0, 3], emb_pca, "Embedding PCA")
    for idx in range(4, n_cols):
        axes[0, idx].axis("off")

    for idx, (name, pred_dir) in enumerate(versions):
        pred = _load_pred(patch_id, pred_dir)
        _show(axes[1, idx], pred, f"{name}", cmap="jet", vmin=0, vmax=1)
    for idx in range(n_versions, n_cols):
        axes[1, idx].axis("off")

    fig.suptitle(f"Haidian Construction - {patch_id}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"compare_haidian_construction_{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("saved %s", out_path)


def plot_harbin_task_comparison(patch_id: str, task: str, data_root: Path, mask_dir: Path,
                                our_pred_dir: Path, aef_pred_dir: Path,
                                our_name: str = "Ours") -> None:
    """Harbin task: t1/t2 输入（S2 优先，缺失用 S1 回退），GT t1/t2，our pred，AEF pred。"""
    input_t1, title_t1 = _load_input(patch_id, data_root, month="202512")
    input_t2, title_t2 = _load_input(patch_id, data_root, month="202605")
    mask_t1 = _load_mask(f"{patch_id}_202512", mask_dir)
    mask_t2 = _load_mask(f"{patch_id}_202605", mask_dir)
    our_pred = _load_pred(patch_id, our_pred_dir)
    aef_pred = _load_pred(patch_id, aef_pred_dir)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    _show(axes[0, 0], input_t1, title_t1)
    _show(axes[0, 1], input_t2, title_t2)
    _show(axes[0, 2], mask_t1, "GT 202512", cmap="jet", vmin=0, vmax=1)
    _show(axes[0, 3], mask_t2, "GT 202605", cmap="jet", vmin=0, vmax=1)
    _show(axes[1, 0], our_pred, our_name, cmap="jet", vmin=0, vmax=1)
    _show(axes[1, 1], aef_pred, "AEF 2025", cmap="jet", vmin=0, vmax=1)
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")

    fig.suptitle(f"Harbin {task.replace('_', ' ').title()} - {patch_id}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"compare_harbin_{task}_{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("saved %s", out_path)


def main() -> None:
    # 1. Haidian construction: v1 / v2 / v3 on same patches
    haidian_data = Path("/data/xuannv_embedding/processed/haidian")
    haidian_mask = Path("/data/xuannv_embedding/processed/haidian/labels/construction/masks")
    haidian_emb = Path("/data/xuannv_embedding/embeddings/haidian")
    haidian_versions = [
        ("V1", [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_haidian_construction_frac1.0_20260620_1728/fold_{i}/predictions") for i in range(5)]),
        ("V2", [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_haidian_construction_v2_frac1.0_20260620_1744/fold_{i}/predictions") for i in range(5)]),
        ("V3", [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_haidian_construction_v3_frac1.0_20260620_1814/fold_{i}/predictions") for i in range(5)]),
    ]
    # 仅使用有标注的 patch，选取 GT 面积较大的前 3 个示例
    for pid in ["patch_000198", "patch_000090", "patch_000209"]:
        plot_haidian_construction_versions(pid, haidian_data, haidian_mask, haidian_versions, haidian_emb)

    # 2. Harbin construction: ours vs AEF
    harbin_data = Path("/data/xuannv_embedding/processed/harbin")
    harbin_construction_mask = Path("/data/xuannv_embedding/processed/harbin/labels/construction/masks")
    # 仅使用有标注且自研与 AEF 均有预测的 patch
    harbin_construction_patches = [
        ("patch_000217",
         [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)]),
        ("patch_000304",
         [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in harbin_construction_patches:
        plot_harbin_task_comparison(pid, "construction", harbin_data, harbin_construction_mask, our_dir, aef_dir)

    # 3. Harbin building_change: ours vs AEF
    building_change_mask = Path("/data/xuannv_embedding/processed/harbin/labels/building_change/masks")
    # 仅使用有标注且自研与 AEF 均有预测的 patch
    building_change_patches = [
        ("patch_000027",
         [Path(f"/data/xuannv_embedding/experiments/building_change_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/building_change/fold_{i}/predictions") for i in range(5)]),
        ("patch_000026",
         [Path(f"/data/xuannv_embedding/experiments/building_change_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/building_change/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in building_change_patches:
        plot_harbin_task_comparison(pid, "building_change", harbin_data, building_change_mask, our_dir, aef_dir)

    # 4. Harbin farm_change: ours vs AEF
    farm_change_mask = Path("/data/xuannv_embedding/processed/harbin/labels/farm_change/masks")
    # 修正预测路径；仅使用有标注且自研与 AEF 均有预测的 patch
    farm_change_patches = [
        ("patch_000034",
         [Path("/data/xuannv_embedding/experiments/farm_change_unet_harbin_fold0/fold_0/predictions")],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/farm_change/fold_{i}/predictions") for i in range(5)]),
        ("patch_000087",
         [Path("/data/xuannv_embedding/experiments/farm_change_unet_harbin_fold0/fold_0/predictions")],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/farm_change/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in farm_change_patches:
        plot_harbin_task_comparison(pid, "farm_change", harbin_data, farm_change_mask, our_dir, aef_dir)

    # 5. Harbin rubbish: ours vs AEF
    rubbish_mask = Path("/data/xuannv_embedding/processed/harbin/labels/rubbish/masks")
    # 仅使用有标注且自研与 AEF 均有预测的 patch
    rubbish_patches = [
        ("patch_000027",
         [Path(f"/data/xuannv_embedding/experiments/rubbish_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/rubbish/fold_{i}/predictions") for i in range(5)]),
        ("patch_000033",
         [Path(f"/data/xuannv_embedding/experiments/rubbish_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/rubbish/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in rubbish_patches:
        plot_harbin_task_comparison(pid, "rubbish", harbin_data, rubbish_mask, our_dir, aef_dir)


if __name__ == "__main__":
    main()
