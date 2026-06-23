#!/usr/bin/env python
"""将 haidian/harbin 的 12 波段 S2 TIFF 拆分为 s2（10m）与 s2_20m（20m）。

输出到 /data2/xuannv_embedding/national/processed_labeled_split，
并生成新的 manifest 与按 region 组织的统计量，供 national V1.2 模型提取 embedding。
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 12 波段 S2 的 band 顺序：B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
S2_10M_INDICES = [3, 2, 1, 7]  # red, green, blue, nir (与 Earth Search 10m 顺序一致)
S2_20M_INDICES = [4, 5, 6, 8, 10, 11]  # rededge1/2/3, nir08, swir16, swir22
SOURCES = {"s2": S2_10M_INDICES, "s2_20m": S2_20M_INDICES}


def split_and_save(src_path: Path, dst_root: Path) -> dict[str, Path]:
    """读取 12 波段 S2，拆分后保存，返回相对路径字典。"""
    with rasterio.open(src_path) as src:
        profile = src.profile
        data = src.read()  # (12, H, W)

    out_paths: dict[str, Path] = {}
    for source, indices in SOURCES.items():
        sub = data[indices].astype(np.float32)
        rel = Path(source) / src_path.name.replace("s2_", f"{source}_")
        out_path = dst_root / "patches" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=len(indices), dtype="float32", compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(sub)
        out_paths[source] = out_path
    return out_paths


def compute_stats(file_list: list[Path]) -> dict[str, Any]:
    """对文件列表计算每个波段的 mean/std。"""
    sums: np.ndarray | None = None
    sumsq: np.ndarray | None = None
    counts: np.ndarray | None = None
    for path in file_list:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)
        if sums is None:
            sums = np.zeros(arr.shape[0], dtype=np.float64)
            sumsq = np.zeros(arr.shape[0], dtype=np.float64)
            counts = np.zeros(arr.shape[0], dtype=np.float64)
        valid = arr != 0
        sums += (arr * valid).sum(axis=(1, 2))
        sumsq += ((arr**2) * valid).sum(axis=(1, 2))
        counts += valid.sum(axis=(1, 2))
    mean = (sums / np.maximum(counts, 1)).astype(np.float32)
    std = np.sqrt(np.maximum(sumsq / np.maximum(counts, 1) - mean**2, 0)).astype(
        np.float32
    )
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": int(counts.sum()) if counts is not None else 0,
    }


def _process_entry(
    entry: dict[str, Any],
    processed_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, list[Path]]]]:
    """处理单个 manifest 条目，返回新条目与文件列表。"""
    patch_id = entry["patch_id"]
    region = entry.get("region") or entry["s2"][0].split("/")[0]
    new_entry: dict[str, Any] = {"patch_id": patch_id, "region": region}
    region_files: dict[str, list[Path]] = {s: [] for s in SOURCES}

    for s2_path_str in entry.get("s2", []):
        s2_path = processed_root / s2_path_str
        if not s2_path.exists():
            logger.warning("Missing S2 file: %s", s2_path)
            continue
        out_paths = split_and_save(s2_path, output_root / region)
        for source, out_path in out_paths.items():
            rel = out_path.relative_to(output_root / region)
            new_entry.setdefault(source, []).append(str(rel))
            region_files[source].append(out_path)

    return new_entry, {region: region_files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default="/data/xuannv_embedding/processed/manifest_all_744.json",
        type=Path,
    )
    parser.add_argument(
        "--processed-root",
        default="/data/xuannv_embedding/processed",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national/processed_labeled_split",
        type=Path,
    )
    parser.add_argument(
        "--statistics-dir",
        default="/data2/xuannv_embedding/national/statistics_labeled_split",
        type=Path,
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    with args.input_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    new_manifest: list[dict[str, Any]] = []
    region_files: dict[str, dict[str, list[Path]]] = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _process_entry, entry, args.processed_root, args.output_root
            ): i
            for i, entry in enumerate(manifest)
        }
        results: list[tuple[int, dict[str, Any], dict[str, dict[str, list[Path]]]]] = []
        for future in as_completed(futures):
            idx = futures[future]
            new_entry, files = future.result()
            results.append((idx, new_entry, files))
            for region, src_files in files.items():
                if region not in region_files:
                    region_files[region] = {s: [] for s in SOURCES}
                for source, paths in src_files.items():
                    region_files[region][source].extend(paths)

    results.sort(key=lambda x: x[0])
    new_manifest = [r[1] for r in results]

    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote manifest with %d patches to %s", len(new_manifest), manifest_path
    )

    # 统计量
    args.statistics_dir.mkdir(parents=True, exist_ok=True)
    for region, sources in region_files.items():
        region_stat_dir = args.statistics_dir / region
        region_stat_dir.mkdir(parents=True, exist_ok=True)
        for source, files in sources.items():
            if not files:
                continue
            stats = compute_stats(files)
            out_path = region_stat_dir / f"{source}_stats.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(
                "%s %s stats: mean=%s std=%s",
                region,
                source,
                [round(x, 1) for x in stats["mean"]],
                [round(x, 1) for x in stats["std"]],
            )


if __name__ == "__main__":
    main()
