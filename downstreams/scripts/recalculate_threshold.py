#!/usr/bin/env python3
"""从已保存的概率图中重新搜索最优阈值并计算指标。

用于估计下游分割任务在固定 embedding + 当前 head 下的 F1 上限，
并验证验证集上选择的阈值是否在测试集上表现良好。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from downstreams.metrics.segmentation import compute_segmentation_metrics
from sklearn.metrics import precision_recall_curve


def load_prediction_mask(pred_dir: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stem = mask_path.stem
    # Try mask stem directly or strip month suffix
    pred_path = pred_dir / f"{stem}_prob.tif"
    if not pred_path.exists():
        # Strip _YYYYMM suffix and try base id
        base = stem
        for suffix in ["_202512", "_202605", "_202604"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        pred_path = pred_dir / f"{base}_prob.tif"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found for {stem}: {pred_path}")
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return pred, mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    all_scores: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    mask_paths = sorted(args.mask_dir.glob("*.tif"))
    for mask_path in mask_paths:
        try:
            pred, mask = load_prediction_mask(args.pred_dir, mask_path)
        except FileNotFoundError:
            continue
        all_scores.append(pred.flatten())
        all_targets.append((mask > 0).astype(np.int32).flatten())

    if not all_scores:
        print("No predictions/masks found.")
        return

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)

    # Search threshold on the combined predictions (oracle upper bound)
    p_arr, r_arr, thresholds = precision_recall_curve(targets, scores)
    f1s = 2 * p_arr[:-1] * r_arr[:-1] / (p_arr[:-1] + r_arr[:-1] + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    # Metrics at threshold 0.5 and best threshold
    metrics_05 = compute_segmentation_metrics(scores, targets, threshold=0.5)
    metrics_best = compute_segmentation_metrics(scores, targets, threshold=best_threshold)

    result = {
        "n_pixels": int(len(scores)),
        "n_positive": int(targets.sum()),
        "best_threshold": best_threshold,
        "f1_best_oracle": best_f1,
        "metrics_threshold_0.5": {k: v for k, v in metrics_05.items() if isinstance(v, (int, float))},
        "metrics_best_threshold": {k: v for k, v in metrics_best.items() if isinstance(v, (int, float))},
    }

    out_path = args.output or args.pred_dir.parent / "recalculated_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"n_pixels={result['n_pixels']}, n_positive={result['n_positive']}")
    print(f"threshold=0.5: F1={metrics_05['f1_0.5']:.4f}, AUC={metrics_05['auc_roc']:.4f}")
    print(f"best_threshold={best_threshold:.4f}: F1={metrics_best['f1_0.5']:.4f}, AUC={metrics_best['auc_roc']:.4f}")
    print(f"oracle f1_best={best_f1:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
