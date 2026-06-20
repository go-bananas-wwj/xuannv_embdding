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
    overlay = rgb.copy()
    overlay[pred > 0] = overlay[pred > 0] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha
    overlay[target > 0] = overlay[target > 0] * (1 - alpha) + np.array([0.0, 1.0, 0.0]) * alpha
    return np.clip(overlay, 0, 1)


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()
