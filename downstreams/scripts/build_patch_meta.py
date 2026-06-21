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


def build_region_meta(mask_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Scan multiple mask directories for *.tif and build per-patch metadata.

    Mask filenames may include a month suffix (e.g. patch_000001_202512.tif);
    the patch_id is the base name before the first underscore after patch_*.
    """
    meta: dict[str, dict[str, Any]] = {}
    seen_files: set[Path] = set()

    for mask_dir in mask_dirs:
        if not mask_dir.is_dir():
            logger.warning("Skipping missing mask dir: %s", mask_dir)
            continue
        for tif_path in sorted(mask_dir.glob("*.tif")):
            if tif_path in seen_files:
                continue
            seen_files.add(tif_path)
            # patch_id is the stem with any trailing _YYYYMM removed
            stem = tif_path.stem
            import re
            if re.fullmatch(r"patch_\d+_\d{6}", stem):
                base_id = stem.rsplit("_", 1)[0]
            else:
                base_id = stem
            if base_id not in meta:
                meta[base_id] = extract_patch_meta(tif_path)

    if not meta:
        raise ValueError(f"No *.tif files found in any of {mask_dirs}")

    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Build AEF patch metadata JSONs")
    p.add_argument(
        "--harbin-mask-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("/data/xuannv_embedding/processed/harbin/labels/construction/masks"),
            Path("/data/xuannv_embedding/processed/harbin/labels/building_change/masks"),
            Path("/data/xuannv_embedding/processed/harbin/labels/farm_change/masks"),
            Path("/data/xuannv_embedding/processed/harbin/labels/rubbish/masks"),
        ],
        help="Directories containing harbin mask GeoTIFFs",
    )
    p.add_argument(
        "--haidian-mask-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("/data/xuannv_embedding/processed/haidian/labels/construction/masks"),
        ],
        help="Directories containing haidian mask GeoTIFFs",
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
        "harbin": args.harbin_mask_dirs,
        "haidian": args.haidian_mask_dirs,
    }

    for region, mask_dirs in regions.items():
        logger.info("Building metadata for %s from %s", region, mask_dirs)
        meta = build_region_meta(mask_dirs)
        out_path = args.output_root / f"{region}_patch_meta.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("%s: wrote %d patches to %s", region, len(meta), out_path)


if __name__ == "__main__":
    main()
