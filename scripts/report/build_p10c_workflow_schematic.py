#!/usr/bin/env python3
"""Render a manuscript schematic for the P10C encoder and sparse PU reader."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, width, height, label, color, *, fontsize=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=color,
        edgecolor="#424242",
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=fontsize)
    return xy[0] + width, xy[1] + height / 2


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#424242"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(11.0, 5.7))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)
    axis.axis("off")

    axis.text(0.3, 7.55, "Training-time multimodal encoder", fontsize=12, fontweight="bold")
    inputs = box(axis, (0.3, 5.7), 2.15, 1.05, "Six temporal slots\nS2 (12), S1 (2),\nLandsat (7)", "#d7eaf8")
    highres = box(axis, (0.3, 3.95), 2.15, 0.95, "Availability-masked\nhigh-resolution optical/SAR", "#e6f3d1")
    stems = box(axis, (3.0, 5.25), 1.7, 1.1, "Sensor-specific\n3×3 stems", "#fde0c5")
    stp = box(axis, (5.25, 5.05), 2.0, 1.5, "Space-Time-Precision\nencoder\n6 blocks, 8 heads", "#f4cccc")
    fusion = box(axis, (7.8, 5.25), 1.7, 1.1, "High-resolution\nfusion", "#d9d2e9")
    bottleneck = box(axis, (10.05, 5.25), 1.45, 1.1, "vMF-style\nbottleneck", "#ead1dc")
    dense = box(axis, (12.0, 5.05), 1.65, 1.5, "Dense field\n64 × 128 × 128\n10 m grid", "#cfe2f3")
    arrow(axis, inputs, (3.0, 5.8))
    arrow(axis, highres, (7.8, 5.55))
    arrow(axis, stems, (5.25, 5.8))
    arrow(axis, stp, (7.8, 5.8))
    arrow(axis, fusion, (10.05, 5.8))
    arrow(axis, bottleneck, (12.0, 5.8))

    objective = box(axis, (4.7, 2.2), 3.1, 1.5, "Training objectives\nmasked continuous reconstruction\nmerged OSM land cover + 13 semantic probes\nstructured source corruption + uniformity", "#fff2cc", fontsize=8.4)
    arrow(axis, (6.25, 3.7), (6.25, 5.05))
    axis.text(4.7, 1.72, "OSM is auxiliary supervision only; it is not an encoder input.", fontsize=8, color="#555555")

    axis.text(0.3, 1.15, "Frozen-feature OSM-assisted sparse readout", fontsize=12, fontweight="bold")
    support = box(axis, (0.3, 0.15), 2.45, 0.8, "Three positive OSM polygons\n→ three target prototypes", "#d9ead3")
    background = box(axis, (3.25, 0.15), 2.25, 0.8, "Reliable background\nprototype mining", "#fce5cd")
    reader = box(axis, (6.05, 0.05), 2.8, 1.0, "Maximum target similarity\nminus background similarity\n(test-time Query disabled)", "#c9daf8", fontsize=8.5)
    calibration = box(axis, (9.4, 0.15), 1.95, 0.8, "Threshold on\n25 OSM-mask\nvalidation patches", "#ead1dc", fontsize=8.2)
    prediction = box(axis, (11.9, 0.15), 1.75, 0.8, "Test map\nF1 / AP / AUC", "#d9e1f2")
    arrow(axis, support, (6.05, 0.55))
    arrow(axis, background, (6.05, 0.55))
    arrow(axis, reader, (9.4, 0.55))
    arrow(axis, calibration, (11.9, 0.55))
    arrow(axis, (12.8, 5.05), (12.8, 1.05))

    figure.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
