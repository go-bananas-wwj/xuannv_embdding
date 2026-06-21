#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.metrics.visualization import overlay_prediction, plot_pr_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_rgb_path(rgb_source: Path, patch_id: str) -> Path | None:
    candidates = sorted(rgb_source.rglob(f"*{patch_id}*.tif"))
    if not candidates:
        candidates = sorted(rgb_source.glob(f"*{patch_id}*.tif"))
    return candidates[0] if candidates else None


def _read_rgb(path: Path) -> np.ndarray | None:
    with rasterio.open(path) as src:
        if src.count < 3:
            return None
        arr = src.read([1, 2, 3])
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    for i in range(3):
        band = arr[..., i]
        mn, mx = np.percentile(band, [2, 98])
        if mx > mn:
            arr[..., i] = np.clip((band - mn) / (mx - mn), 0, 1)
        else:
            arr[..., i] = np.zeros_like(band)
    return arr


def _collect_fold_probs(pred_dir: Path, mask_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    probs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for prob_path in sorted(pred_dir.glob("*_prob.tif")):
        patch_id = prob_path.stem.replace("_prob", "")
        mask_path = mask_dir / f"{patch_id}.tif"
        if not mask_path.exists():
            continue
        with rasterio.open(prob_path) as src:
            prob = src.read(1)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        probs.append(prob.ravel())
        masks.append(mask.ravel())
    if not probs:
        return None
    return np.concatenate(probs), np.concatenate(masks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--task", type=str, default="construction")
    p.add_argument("--rgb-source", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=10)
    args = p.parse_args()

    mask_dir = args.label_root / args.task / "masks"

    for fold_dir in sorted(args.output_root.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logger.info("%s: miou=%.3f f1=%.3f", fold_dir.name, metrics["miou"], metrics["f1_0.5"])

        pred_dir = fold_dir / "predictions"
        vis_dir = fold_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 汇总 PR 曲线
        collected = _collect_fold_probs(pred_dir, mask_dir)
        if collected is not None:
            probs, masks = collected
            fold_metrics = compute_segmentation_metrics(probs, masks, return_curve=True)
            plot_pr_curve(
                fold_metrics["precision_curve"],
                fold_metrics["recall_curve"],
                fold_metrics["ap"],
                vis_dir / "pr_curve.png",
            )
            logger.info(
                "%s fold-level AP=%.3f AUPRC=%.3f",
                fold_dir.name,
                fold_metrics["ap"],
                fold_metrics["auprc"],
            )

        # 样本叠加图
        if not pred_dir.exists():
            continue
        for prob_path in sorted(pred_dir.glob("*_prob.tif"))[: args.n_samples]:
            patch_id = prob_path.stem.replace("_prob", "")
            mask_path = mask_dir / f"{patch_id}.tif"
            if not mask_path.exists():
                continue
            with rasterio.open(prob_path) as src:
                prob = src.read(1)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            pred = (prob > 0.5).astype(np.uint8)

            rgb: np.ndarray | None = None
            if args.rgb_source:
                rgb_path = _find_rgb_path(args.rgb_source, patch_id)
                if rgb_path:
                    rgb = _read_rgb(rgb_path)
            if rgb is None or rgb.shape[:2] != pred.shape:
                # 退化为概率灰度图
                gray = np.stack([prob] * 3, axis=-1)
                mn, mx = gray[..., 0].min(), gray[..., 0].max()
                if mx > mn:
                    rgb = (gray - mn) / (mx - mn)
                else:
                    rgb = np.zeros_like(gray)

            overlay = overlay_prediction(rgb, pred, mask)
            out_png = vis_dir / f"{patch_id}_overlay.png"
            plt.imsave(out_png, overlay)
            logger.info("保存可视化 %s", out_png)

    logger.info("可视化完成")


if __name__ == "__main__":
    main()
