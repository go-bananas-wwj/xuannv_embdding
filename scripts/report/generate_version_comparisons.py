#!/usr/bin/env python3
"""生成多版本 / 多模型 / AEF 同 patch 对比可视化图。"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter
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


def _norm_for_display(im: np.ndarray | None, lower: float = 2.0, upper: float = 98.0) -> np.ndarray | None:
    """对单通道概率图做 2-98 百分位拉伸，便于低置信度预测可视化。"""
    if im is None:
        return None
    im = np.nan_to_num(im, nan=0.0)
    lo, hi = np.percentile(im, [lower, upper])
    if hi <= lo:
        return np.zeros_like(im)
    return np.clip((im - lo) / (hi - lo), 0, 1)


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


def _load_input(patch_id: str, data_root: Path, month: str) -> tuple[np.ndarray | None, str]:
    """优先读取高分光学，缺失则回退到 S2 RGB，再缺失回退到 S1。返回 (image, title)。"""
    highres_root = data_root / "patches" / "highres_optical"
    s2_root = data_root / "patches" / "s2"
    s1_root = data_root / "patches" / "s1"
    highres = _load_highres_optical(patch_id, highres_root, month=month)
    if highres is not None:
        return highres, f"High-res Optical {month}"
    s2 = _load_s2_rgb(patch_id, s2_root, month=month)
    if s2 is not None:
        return s2, f"S2 {month}"
    s1 = _load_s1(patch_id, s1_root, month=month)
    if s1 is not None:
        return s1, f"S1 {month}"
    return None, f"High-res Optical {month}"


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


def _stripe_score(emb: np.ndarray) -> float:
    """embedding 行方向高频噪声强度，越小越平滑。"""
    row_mean = emb.mean(axis=(0, 2))
    spec = np.abs(np.fft.rfft(row_mean))
    low = spec[:5].sum()
    high = spec[5:].sum()
    return high / (low + 1e-8)


def _best_pca_month(emb_root: Path, patch_id: str) -> str:
    patch_dir = emb_root / patch_id
    best_month, best_score = None, float("inf")
    for path in patch_dir.glob("*_embedding_map.pt"):
        month = path.stem[:6]
        emb = torch.load(path, map_location="cpu", weights_only=True).numpy()
        score = _stripe_score(emb)
        if score < best_score:
            best_score, best_month = score, month
    return best_month if best_month else "202605"


def _load_embedding_pca(
    patch_id: str, emb_root: Path, month: str = "202604", smooth_sigma: float = 0.0
) -> np.ndarray | None:
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
    rgb = _stretch(pcs)
    if smooth_sigma > 0:
        rgb = gaussian_filter(rgb, sigma=(smooth_sigma, smooth_sigma, 0))
    return rgb


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
    emb_pca = _load_embedding_pca(patch_id, emb_root, month=month, smooth_sigma=1.0)

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


def plot_haidian_v3_vs_aef(
    patch_id: str,
    data_root: Path,
    mask_dir: Path,
    v3_pred_dir: Path | list[Path],
    aef_emb_root: Path,
    aef_pred_dir: Path | list[Path],
    our_emb_root: Path,
    our_month: str = "202505",
    aef_month: str = "202512",
) -> None:
    """海淀 construction：Stage2_V1 V3 预测 与 AEF Official 预测同 patch 对比。"""
    highres_root = data_root / "patches" / "highres_optical"

    highres_our = _load_highres_optical(patch_id, highres_root, month=our_month)
    highres_aef = _load_highres_optical(patch_id, highres_root, month=aef_month)
    mask = _load_mask(patch_id, mask_dir)
    our_emb_pca = _load_embedding_pca(patch_id, our_emb_root, month=our_month, smooth_sigma=1.0)
    aef_emb_pca = _load_embedding_pca(patch_id, aef_emb_root, month=aef_month, smooth_sigma=0.0)
    our_pred = _load_pred(patch_id, v3_pred_dir)
    aef_pred = _load_pred(patch_id, aef_pred_dir)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    _show(axes[0, 0], highres_our, f"High-res Optical {our_month}")
    _show(axes[0, 1], our_emb_pca, "Stage2_V1 Embedding PCA")
    _show(axes[0, 2], our_pred, "Stage2_V1 V3 Pred", cmap="jet", vmin=0, vmax=1)
    _show(axes[0, 3], mask, "GT Mask", cmap="jet", vmin=0, vmax=1)
    _show(axes[1, 0], highres_aef, f"High-res Optical {aef_month}")
    _show(axes[1, 1], aef_emb_pca, "AEF Embedding PCA")
    _show(axes[1, 2], aef_pred, "AEF Pred", cmap="jet", vmin=0, vmax=1)
    _show(axes[1, 3], mask, "GT Mask", cmap="jet", vmin=0, vmax=1)

    fig.suptitle(f"Haidian Construction V3 vs AEF - {patch_id}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"compare_haidian_construction_{patch_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("saved %s", out_path)


def plot_harbin_task_comparison(
    patch_id: str,
    task: str,
    data_root: Path,
    mask_dir: Path,
    our_pred_dir: Path,
    aef_pred_dir: Path,
    our_name: str = "Ours",
    our_emb_root: Path | None = None,
    aef_emb_root: Path | None = None,
    our_emb_month: str = "202605",
    aef_emb_month: str = "202512",
) -> None:
    """Harbin task: t1/t2 输入（S2 优先，缺失用 S1 回退）、embedding PCA、GT、预测对比。"""
    input_t1, title_t1 = _load_input(patch_id, data_root, month="202512")
    input_t2, title_t2 = _load_input(patch_id, data_root, month="202605")
    mask_t1 = _load_mask(f"{patch_id}_202512", mask_dir)
    mask_t2 = _load_mask(f"{patch_id}_202605", mask_dir)
    our_pred = _load_pred(patch_id, our_pred_dir)
    aef_pred = _load_pred(patch_id, aef_pred_dir)
    if our_emb_root is not None:
        our_emb_month = our_emb_month or _best_pca_month(our_emb_root, patch_id)
        our_emb = _load_embedding_pca(patch_id, our_emb_root, month=our_emb_month, smooth_sigma=0.5)
    else:
        our_emb = None
    if aef_emb_root is not None:
        aef_emb_month = aef_emb_month or "202512"
        aef_emb = _load_embedding_pca(patch_id, aef_emb_root, month=aef_emb_month, smooth_sigma=0.0)
    else:
        aef_emb = None

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    if our_emb is not None and aef_emb is not None:
        # 上排：输入 + embedding PCA；下排：GT + 预测
        _show(axes[0, 0], input_t1, title_t1)
        _show(axes[0, 1], input_t2, title_t2)
        _show(axes[0, 2], our_emb, f"{our_name} Embedding PCA")
        _show(axes[0, 3], aef_emb, "AEF Embedding PCA")
        _show(axes[1, 0], mask_t1, "GT 202512", cmap="jet", vmin=0, vmax=1)
        _show(axes[1, 1], mask_t2, "GT 202605", cmap="jet", vmin=0, vmax=1)
        _show(axes[1, 2], _norm_for_display(our_pred), f"{our_name} (norm)", cmap="jet", vmin=0, vmax=1)
        _show(axes[1, 3], _norm_for_display(aef_pred), "AEF 2025 (norm)", cmap="jet", vmin=0, vmax=1)
    else:
        _show(axes[0, 0], input_t1, title_t1)
        _show(axes[0, 1], input_t2, title_t2)
        _show(axes[0, 2], mask_t1, "GT 202512", cmap="jet", vmin=0, vmax=1)
        _show(axes[0, 3], mask_t2, "GT 202605", cmap="jet", vmin=0, vmax=1)
        _show(axes[1, 0], _norm_for_display(our_pred), f"{our_name} (norm)", cmap="jet", vmin=0, vmax=1)
        _show(axes[1, 1], _norm_for_display(aef_pred), "AEF 2025 (norm)", cmap="jet", vmin=0, vmax=1)
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
    aef_haidian_emb = Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual/haidian")
    v3_pred_dir = [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_haidian_construction_v3_frac1.0_20260620_1814/fold_{i}/predictions") for i in range(5)]
    aef_construction_pred_dir = [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)]
    # 仅使用有标注、且 Stage2_V1 / AEF embedding 与预测均存在的 patch
    for pid in ["patch_000198", "patch_000090", "patch_000209"]:
        plot_haidian_v3_vs_aef(
            pid,
            haidian_data,
            haidian_mask,
            v3_pred_dir,
            aef_haidian_emb,
            aef_construction_pred_dir,
            our_emb_root=haidian_emb,
            our_month="202601",
            aef_month="202512",
        )

    # 2. Harbin construction: ours vs AEF，使用高分光学前后时相
    harbin_data = Path("/data/xuannv_embedding/processed/harbin")
    harbin_construction_mask = Path("/data/xuannv_embedding/processed/harbin/labels/construction/masks")
    # 选取 GT 面积较大、且自研与 AEF 预测均有明显响应的 patch
    harbin_construction_patches = [
        ("patch_000255",
         [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)]),
        ("patch_000285",
         [Path(f"/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/construction/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in harbin_construction_patches:
        plot_harbin_task_comparison(pid, "construction", harbin_data, harbin_construction_mask, our_dir, aef_dir)

    # 3. Harbin building_change: ours vs AEF，选取有标注且 label 占比最大的 patch
    building_change_mask = Path("/data/xuannv_embedding/processed/harbin/labels/building_change/masks")
    our_building_change_emb = Path("/data/xuannv_embedding/embeddings/harbin")
    aef_building_change_emb = Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual/harbin")
    building_change_patches = [
        ("patch_000027",
         [Path(f"/data/xuannv_embedding/experiments/building_change_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/building_change/fold_{i}/predictions") for i in range(5)]),
        ("patch_000026",
         [Path(f"/data/xuannv_embedding/experiments/building_change_diff_unet_harbin_5fold/fold{i}/fold_{i}/predictions") for i in range(5)],
         [Path(f"/data/xuannv_embedding/experiments/aef_benchmark/building_change/fold_{i}/predictions") for i in range(5)]),
    ]
    for pid, our_dir, aef_dir in building_change_patches:
        plot_harbin_task_comparison(
            pid, "building_change", harbin_data, building_change_mask, our_dir, aef_dir,
            our_emb_root=our_building_change_emb,
            aef_emb_root=aef_building_change_emb,
            our_emb_month=None,
            aef_emb_month="202512",
        )

    # 4. Harbin farm_change: ours vs AEF，选取预测响应相对最好的 patch
    farm_change_mask = Path("/data/xuannv_embedding/processed/harbin/labels/farm_change/masks")
    farm_change_patches = [
        # 000034 是少数有 GT（15 像素）的 patch；000087 有 AEF 强响应
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
