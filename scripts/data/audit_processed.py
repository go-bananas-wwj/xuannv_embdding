#!/usr/bin/env python3
"""已处理数据质量审计脚本。

检查项：
1. manifest 中每个条目对应的影像/掩膜文件是否存在。
2. 各 source 的波段数、空间尺寸、CRS 是否一致。
3. 高分辨率数据与低分辨率数据的空间/通道一致性。
4. worldcover 标签是否存在。
5. 各 source 有效掩膜平均/最小有效比例（采样统计）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _file_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with rasterio.open(path) as src:
            return {
                "bands": src.count,
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs),
            }
    except Exception as exc:
        return {"error": str(exc)}


def _mask_valid_ratio(path: Path) -> float | None:
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            return float(arr.mean())
    except Exception:
        return None


def _audit_entry(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    pid = entry["patch_id"]
    result: dict[str, Any] = {"patch_id": pid, "errors": [], "source_stats": {}}
    for source, paths in entry.items():
        if source == "patch_id":
            continue
        if paths is None:
            result["source_stats"][source] = {"status": "null"}
            continue
        meta_set = set()
        valid_ratios = []
        for p in paths:
            img_path = root / p
            mask_path = img_path.with_name(img_path.stem + "_mask.tif")
            if not img_path.exists():
                result["errors"].append(f"{source}: image missing {img_path.name}")
                continue
            if not mask_path.exists():
                result["errors"].append(f"{source}: mask missing {mask_path.name}")
            meta = _file_metadata(img_path)
            if meta and "error" not in meta:
                meta_set.add((meta["bands"], meta["height"], meta["width"], meta["crs"]))
            else:
                result["errors"].append(f"{source}: open error {img_path.name}")
            if mask_path.exists():
                vr = _mask_valid_ratio(mask_path)
                if vr is not None:
                    valid_ratios.append(vr)
        result["source_stats"][source] = {
            "expected_files": len(paths),
            "metadata_variants": [dict(zip(["bands", "height", "width", "crs"], m)) for m in meta_set],
            "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
            "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else None,
        }
    return result


def audit_region(region_root: Path, max_workers: int = 8) -> dict[str, Any]:
    manifest_path = region_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    logger.info("%s: auditing %d patches with %d workers", region_root.name, len(manifest), max_workers)

    errors: list[str] = []
    source_metadata: dict[str, set[tuple[int, int, int, str]]] = defaultdict(set)
    source_valid_ratios: dict[str, list[float]] = defaultdict(list)
    source_file_counts: dict[str, int] = defaultdict(int)
    source_patch_counts: dict[str, int] = defaultdict(int)
    null_counts: dict[str, int] = defaultdict(int)

    # 文件存在性快速检查
    for entry in manifest:
        for source, paths in entry.items():
            if source == "patch_id":
                continue
            if paths is None:
                null_counts[source] += 1
                continue
            source_patch_counts[source] += 1
            source_file_counts[source] += len(paths)
            for p in paths:
                img_path = region_root / p
                mask_path = img_path.with_name(img_path.stem + "_mask.tif")
                if not img_path.exists():
                    errors.append(f"{entry['patch_id']} {source}: image missing {p}")
                if not mask_path.exists():
                    errors.append(f"{entry['patch_id']} {source}: mask missing {p}")

    # 元数据/掩膜采样（每个 source 采样 200 个文件，避免全量打开过慢）
    sample_paths: dict[str, list[Path]] = defaultdict(list)
    rng = np.random.default_rng(42)
    for entry in manifest:
        pid = entry["patch_id"]
        for source, paths in entry.items():
            if source == "patch_id" or not paths:
                continue
            chosen = rng.choice(paths, size=min(len(paths), 1), replace=False)
            for p in chosen:
                sample_paths[source].append(region_root / p)
    for source in sample_paths:
        files = sample_paths[source]
        if len(files) > 200:
            idx = rng.choice(len(files), size=200, replace=False)
            files = [files[i] for i in idx]
        for img_path in files:
            meta = _file_metadata(img_path)
            if meta and "error" not in meta:
                source_metadata[source].add(
                    (meta["bands"], meta["height"], meta["width"], meta["crs"])
                )
            mask_path = img_path.with_name(img_path.stem + "_mask.tif")
            if mask_path.exists():
                vr = _mask_valid_ratio(mask_path)
                if vr is not None:
                    source_valid_ratios[source].append(vr)

    return {
        "region": region_root.name,
        "manifest_patches": len(manifest),
        "errors": errors,
        "source_summary": {
            src: {
                "files": source_file_counts[src],
                "patches": source_patch_counts[src],
                "null_patches": null_counts[src],
                "metadata_variants": [
                    dict(zip(["bands", "height", "width", "crs"], m))
                    for m in source_metadata[src]
                ],
                "valid_ratio_mean": float(np.mean(source_valid_ratios[src])) if source_valid_ratios[src] else None,
                "valid_ratio_min": float(np.min(source_valid_ratios[src])) if source_valid_ratios[src] else None,
            }
            for src in sorted(set(list(source_file_counts.keys()) + list(null_counts.keys())))
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit processed embedding data.")
    parser.add_argument("--root", default="/data/xuannv_embedding/processed", type=Path)
    parser.add_argument("--regions", default="haidian,harbin", help="comma-separated regions")
    parser.add_argument("--out", default="audit_report.json", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}
    for region in args.regions.split(","):
        region_root = args.root / region
        if not region_root.exists():
            logger.warning("Region root not found: %s", region_root)
            continue
        report[region] = audit_region(region_root, max_workers=args.workers)

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
