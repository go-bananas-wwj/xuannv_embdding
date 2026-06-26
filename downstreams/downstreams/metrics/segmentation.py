from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def _ensure_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_segmentation_metrics(
    pred_logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    ignore_index: int = -1,
    return_curve: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """pred_logits: (N, H, W) 或 (H, W)；target: 同 shape 的 int 标签。"""
    pred_logits = _ensure_numpy(pred_logits)
    target = _ensure_numpy(target)

    if pred_logits.ndim == 2:
        pred_logits = pred_logits[None]
        target = target[None]

    valid = target != ignore_index
    probs = 1.0 / (1.0 + np.exp(-pred_logits))
    preds_threshold = (probs > threshold).astype(np.uint8)
    preds_05 = (probs > 0.5).astype(np.uint8)

    tp = int(((preds_threshold == 1) & (target == 1) & valid).sum())
    fp = int(((preds_threshold == 1) & (target == 0) & valid).sum())
    fn = int(((preds_threshold == 0) & (target == 1) & valid).sum())
    tn = int(((preds_threshold == 0) & (target == 0) & valid).sum())
    tp_05 = int(((preds_05 == 1) & (target == 1) & valid).sum())
    fp_05 = int(((preds_05 == 1) & (target == 0) & valid).sum())
    fn_05 = int(((preds_05 == 0) & (target == 1) & valid).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    precision_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) > 0 else 0.0
    recall_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) > 0 else 0.0
    f1_05 = (
        2 * precision_05 * recall_05 / (precision_05 + recall_05)
        if (precision_05 + recall_05) > 0
        else 0.0
    )

    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 1.0

    # PR 曲线 / AP / AUPRC
    y_true = (target[valid] == 1).astype(np.int32)
    y_score = probs[valid]
    p_arr = np.array([])
    r_arr = np.array([])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        ap = auprc = auc_roc = 0.0
        best_f1 = 0.0
        best_threshold = 0.5
    else:
        p_arr, r_arr, thresholds = precision_recall_curve(y_true, y_score)
        auprc = auc(r_arr, p_arr)
        # AP via sklearn average_precision_score 更稳
        ap = average_precision_score(y_true, y_score)
        auc_roc = roc_auc_score(y_true, y_score)
        f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
        best_idx = int(f1s.argmax())
        best_f1 = float(f1s[best_idx])
        if thresholds.size == 0:
            best_threshold = 0.5
        else:
            threshold_idx = min(best_idx, thresholds.size - 1)
            best_threshold = float(thresholds[threshold_idx])

    result: dict[str, Any] = {
        "miou": float(iou),
        "f1_0.5": float(f1_05),
        "f1_at_threshold": float(f1),
        "threshold": float(threshold),
        "f1_best": best_f1,
        "best_threshold": best_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(ap),
        "auprc": float(auprc),
        "auc_roc": float(auc_roc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    if return_curve:
        result["precision_curve"] = p_arr
        result["recall_curve"] = r_arr
    return result
