#!/usr/bin/env python3
"""Render the P10C transductive sparse-readout study design for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def node(ax, x, y, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=color, edgecolor="#424242", linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x + width, y + height / 2)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#424242"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 7)
    axis.axis("off")

    axis.text(0.35, 6.35, "Haidian transductive sparse-readout study design", fontsize=13, fontweight="bold")
    archive = node(axis, 0.4, 4.25, 2.5, 1.05, "Haidian archive\n320 georeferenced patches\nDec. 2025--May 2026", "#d7eaf8")
    encoder = node(axis, 3.65, 4.25, 2.5, 1.05, "P10C encoder\ntrained on complete\nregional archive", "#f4cccc")
    export = node(axis, 6.9, 4.25, 2.4, 1.05, "Frozen April-indexed\n64-D embedding field\nfor all 320 patches", "#d9ead3")
    arrow(axis, archive, (3.65, 4.78))
    arrow(axis, encoder, (6.9, 4.78))

    split = node(axis, 1.15, 1.45, 3.0, 1.25, "For each of five random\npositive-ratio-stratified folds\n64 test + 25 OSM-mask validation\n+ 231 support-pool patches", "#fff2cc", fontsize=9)
    reader = node(axis, 5.05, 1.45, 2.7, 1.25, "Three deterministic\npositive-polygon selections\nper fold (seeds 41, 42, 43)", "#fce5cd", fontsize=9)
    evaluation = node(axis, 8.65, 1.45, 3.2, 1.25, "Same PU reader for\nXuannvEarth / AEF / traditional\nvalidation-only threshold → test metrics", "#c9daf8", fontsize=9)
    arrow(axis, (8.1, 4.25), (2.65, 2.7))
    arrow(axis, split, (5.05, 2.08))
    arrow(axis, reader, (8.65, 2.08))
    axis.text(1.15, 0.55, "Upstream encoder and downstream reader are intentionally distinct. The downstream split is not geographic blocking.", fontsize=8.5, color="#555555")
    figure.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
