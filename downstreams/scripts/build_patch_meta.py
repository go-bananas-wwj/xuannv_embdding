#!/usr/bin/env python3
"""Build patch metadata JSON for AEF benchmark from mask GeoTIFFs.

For each region, scan the mask directory for *.tif files, extract bbox, CRS and
shape with rasterio, and write a JSON keyed by patch_id (filename stem).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_patch_meta(tif_path: Path) -> dict[str, Any]:
    """Return bbox, crs and shape for a GeoTIFF mask."""
    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        return {
            "bbox": [left, bottom, right, top],
            "crs": src.crs.to_string() if src.crs else None,
            "shape": (src.height, src.width),
        }


def build_region_meta(mask_dir: Path) -> dict[str, dict[str, Any]]:
    """Scan mask_dir for *.tif and build per-patch metadata."""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    meta: dict[str, dict[str, Any]] = {}
    tif_paths = sorted(mask_dir.glob("*.tif"))
    if not tif_paths:
        raise ValueError(f"No *.tif files found in {mask_dir}")

    for tif_path in tif_paths:
        patch_id = tif_path.stem
        meta[patch_id] = extract_patch_meta(tif_path)

    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Build AEF patch metadata JSONs")
    p.add_argument(
        "--harbin-masks",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/harbin/masks"),
        help="Directory containing harbin mask GeoTIFFs",
    )
    p.add_argument(
        "--haidian-masks",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/masks"),
        help="Directory containing haidian mask GeoTIFFs",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
        help="Root directory for output JSON files",
    )
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    regions = {
        "harbin": args.harbin_masks,
        "haidian": args.haidian_masks,
    }

    for region, mask_dir in regions.items():
        logger.info("Building metadata for %s from %s", region, mask_dir)
        meta = build_region_meta(mask_dir)
        out_path = args.output_root / f"{region}_patch_meta.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("%s: wrote %d patches to %s", region, len(meta), out_path)


if __name__ == "__main__":
    main()
