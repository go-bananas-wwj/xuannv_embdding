#!/usr/bin/env python
"""计算 national 数据集的 source 统计量（mean/std）。

对 s2 与 s2_20m 分别计算每个波段的均值与标准差，
结果保存为 {statistics_dir}/{region}/{source}_stats.json。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCES = ["s2", "s2_20m"]


def _load_array(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as src:
            return src.read()
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def compute_source_statistics(
    output_root: Path,
    input_csv: Path,
    source: str,
    max_patches: int = 2000,
    eps: float = 1e-8,
) -> dict:
    """对单个 source 的已下载 patch 计算逐波段均值与标准差。"""
    df = pd.read_csv(input_csv)
    source_dir = output_root / "processed" / "national" / "patches" / source

    all_values: list[np.ndarray] = []
    count = 0
    for _, row in df.iterrows():
        if count >= max_patches:
            break
        patch_id = row["patch_id"]
        paths = sorted(source_dir.glob(f"*_{patch_id}.tif"))
        for p in paths:
            arr = _load_array(p)
            if arr is None:
                continue
            # 忽略精确 0 值，避免 nodata/占位零影响统计。
            mask = np.abs(arr) > eps
            if not mask.any():
                continue
            # 按波段收集有效值。
            bands = arr.shape[0]
            if not all_values:
                all_values = [[] for _ in range(bands)]
            for b in range(bands):
                vals = arr[b][mask[b]]
                if vals.size:
                    all_values[b].append(vals)
            count += 1

    if not all_values:
        raise ValueError(f"No valid data found for source {source}")

    means = []
    stds = []
    for b_vals in all_values:
        concat = np.concatenate(b_vals) if b_vals else np.array([0.0], dtype=np.float32)
        means.append(float(np.mean(concat)))
        stds.append(float(np.std(concat)))

    return {
        "mean": means,
        "std": stds,
        "count": int(sum(len(v) for v in all_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national",
        type=Path,
    )
    parser.add_argument(
        "--input-csv",
        default="/data2/xuannv_embedding/national/sample_national_patches.csv",
        type=Path,
    )
    parser.add_argument(
        "--statistics-dir",
        default="/data2/xuannv_embedding/national/statistics/national",
        type=Path,
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=2000,
        help="每个 source 最多采样的 patch 数",
    )
    args = parser.parse_args()

    args.statistics_dir.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        logger.info("Computing statistics for %s", source)
        stats = compute_source_statistics(
            args.output_root,
            args.input_csv,
            source,
            max_patches=args.max_patches,
        )
        out_path = args.statistics_dir / f"{source}_stats.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(
            "Saved %s stats: mean=%s, std=%s",
            source,
            [round(m, 4) for m in stats["mean"]],
            [round(s, 4) for s in stats["std"]],
        )


if __name__ == "__main__":
    main()
