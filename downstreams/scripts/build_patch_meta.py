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


def extract_patch_meta(tif_path: Path, fallback_path: Path | None = None) -> dict[str, Any]:
    """Return bbox, crs and shape for a GeoTIFF mask.

    If the mask itself is not georeferenced (crs is None) and a fallback
    georeferenced patch file is provided, use the fallback's CRS and bounds
    while keeping the mask's pixel shape.
    """
    with rasterio.open(tif_path) as src:
        shape = (src.height, src.width)
        if src.crs is not None:
            left, bottom, right, top = src.bounds
            crs = src.crs.to_string()
        elif fallback_path is not None and fallback_path.exists():
            with rasterio.open(fallback_path) as fallback_src:
                left, bottom, right, top = fallback_src.bounds
                crs = fallback_src.crs.to_string() if fallback_src.crs else None
        else:
            left, bottom, right, top = src.bounds
            crs = None
        return {
            "bbox": [left, bottom, right, top],
            "crs": crs,
            "shape": shape,
        }


def load_fallback_map(manifest_path: Path | None, region_root: Path | None) -> dict[str, Path]:
    """Load a mapping from patch_id to the first existing patch file.

    The manifest is expected to be a list of dicts with a 'patch_id' key and
    modality keys containing lists of relative file paths.  Relative paths are
    resolved against ``region_root``; if ``region_root`` is None, the manifest's
    parent directory is used.
    """
    fallback: dict[str, Path] = {}
    if manifest_path is None or not manifest_path.exists():
        return fallback

    root = region_root if region_root is not None else manifest_path.parent
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        patch_id = entry.get("patch_id")
        if patch_id is None:
            continue
        for key, files in entry.items():
            if key == "patch_id" or not isinstance(files, list):
                continue
            for rel in files:
                candidate = root / rel
                if candidate.exists():
                    fallback[patch_id] = candidate
                    break
            if patch_id in fallback:
                break

    return fallback


def build_region_meta(
    mask_dirs: list[Path],
    manifest_path: Path | None = None,
    region_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Scan multiple mask directories for *.tif and build per-patch metadata.

    Mask filenames may include a month suffix (e.g. patch_000001_202512.tif);
    the patch_id is the base name before the first underscore after patch_*.
    Masks without a CRS are recovered from the optional manifest's georeferenced
    patch files.
    """
    meta: dict[str, dict[str, Any]] = {}
    seen_files: set[Path] = set()
    fallback = load_fallback_map(manifest_path, region_root)

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
                meta[base_id] = extract_patch_meta(tif_path, fallback.get(base_id))

    if not meta:
        raise ValueError(f"No *.tif files found in any of {mask_dirs}")

    return meta


def build_region_meta_from_manifest(
    manifest_path: Path,
    region_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build metadata for every patch in a region manifest.

    Uses the first existing georeferenced source file for each patch to obtain
    bbox, crs and shape. This works for both labeled and unlabeled patches.
    """
    manifest_path = Path(manifest_path)
    root = region_root if region_root is not None else manifest_path.parent
    fallback = load_fallback_map(manifest_path, root)

    meta: dict[str, dict[str, Any]] = {}
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        patch_id = entry.get("patch_id")
        if patch_id is None:
            continue
        source_path = fallback.get(patch_id)
        if source_path is None:
            logger.warning("No georeferenced source found for %s", patch_id)
            continue
        meta[patch_id] = extract_patch_meta(source_path)

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
        "--harbin-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/harbin/manifest_labeled.json"),
        help="Harbin manifest JSON used to recover CRS from patch files",
    )
    p.add_argument(
        "--haidian-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_labeled.json"),
        help="Haidian manifest JSON used to recover CRS from patch files",
    )
    p.add_argument(
        "--harbin-full-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/harbin/manifest.json"),
        help="Full Harbin manifest for all-patch metadata generation",
    )
    p.add_argument(
        "--haidian-full-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest.json"),
        help="Full Haidian manifest for all-patch metadata generation",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
        help="Root directory for output JSON files",
    )
    p.add_argument(
        "--from-full-manifest",
        action="store_true",
        help="Build metadata from full region manifests instead of mask dirs",
    )
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.from_full_manifest:
        regions = {
            "harbin": args.harbin_full_manifest,
            "haidian": args.haidian_full_manifest,
        }
        for region, manifest_path in regions.items():
            logger.info("Building metadata for %s from %s", region, manifest_path)
            region_root = manifest_path.parent if manifest_path.exists() else None
            meta = build_region_meta_from_manifest(manifest_path, region_root=region_root)
            out_path = args.output_root / f"{region}_patch_meta.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("%s: wrote %d patches to %s", region, len(meta), out_path)
        return

    regions = {
        "harbin": (args.harbin_mask_dirs, args.harbin_manifest),
        "haidian": (args.haidian_mask_dirs, args.haidian_manifest),
    }

    for region, (mask_dirs, manifest_path) in regions.items():
        logger.info("Building metadata for %s from %s", region, mask_dirs)
        region_root = manifest_path.parent if manifest_path.exists() else None
        meta = build_region_meta(mask_dirs, manifest_path=manifest_path, region_root=region_root)
        out_path = args.output_root / f"{region}_patch_meta.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("%s: wrote %d patches to %s", region, len(meta), out_path)


if __name__ == "__main__":
    main()
