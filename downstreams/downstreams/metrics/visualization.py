# downstreams/downstreams/metrics/visualization.py
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np


def overlay_prediction(
    rgb: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """rgb: (H, W, 3) [0,1]；pred/target: (H, W) bool/uint8。"""
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, f"rgb must be (H, W, 3), got {rgb.shape}"
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"
    overlay = np.clip(rgb, 0.0, 1.0).copy()
    overlay[pred > 0] = overlay[pred > 0] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha
    overlay[target > 0] = overlay[target > 0] * (1 - alpha) + np.array([0.0, 1.0, 0.0]) * alpha
    return np.clip(overlay, 0, 1)


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    if precision.size == 0 or recall.size == 0:
        plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        plt.title("PR Curve")
    else:
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.legend()
        plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()
