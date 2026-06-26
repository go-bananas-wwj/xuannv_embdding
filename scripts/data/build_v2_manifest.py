#!/usr/bin/env python3
"""Build the V2 mixed-region manifest for the 202512-202605 embedding run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MONTH_RE = re.compile(r"(20\d{4})")
DEFAULT_MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")
DEFAULT_REGIONS = ("haidian", "harbin")
DEFAULT_SOURCES = (
    "s2",
    "s1",
    "landsat",
    "worldcover",
    "highres_optical_haidian",
    "highres_optical_harbin",
    "highres_sar_haidian",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the V2 202512-202605 mixed-region manifest."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed"),
        help="Root containing processed/{haidian,harbin}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605"),
        help="Directory for manifest_v2_202512_202605.json and metadata.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(DEFAULT_MONTHS),
        help="Allowed YYYYMM months.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Regions to include.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Canonical source names to include.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing files.",
    )
    return parser.parse_args()


def _load_region_manifest(region_dir: Path) -> list[dict[str, Any]]:
    for name in ("manifest_full.json", "manifest_stage2.json", "manifest.json"):
        path = region_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must contain a list of patch entries")
            return data
    raise FileNotFoundError(f"No manifest found under {region_dir}")


def _paths_for_source(entry: dict[str, Any], source: str, region: str) -> list[str]:
    value = entry.get(source)
    if value is None and source == "highres_optical_haidian" and region == "haidian":
        value = entry.get("highres_optical")
    if value is None and source == "highres_optical_harbin" and region == "harbin":
        value = entry.get("highres_optical")
    if value is None and source == "highres_sar_haidian" and region == "haidian":
        value = entry.get("highres_sar")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(p) for p in value if p]
    return [str(value)]


def _month_from_path(path: str) -> str | None:
    match = MONTH_RE.search(path)
    return match.group(1) if match else None


def _filter_paths(paths: list[str], source: str, allowed_months: set[str]) -> list[str]:
    if source == "worldcover":
        return sorted(paths)
    filtered = []
    for path in paths:
        month = _month_from_path(path)
        if month in allowed_months:
            filtered.append(path)
    return sorted(filtered)


def _file_shape(path: Path) -> list[int] | None:
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(path) as src:
            return [src.count, src.height, src.width]
    except Exception:
        return None


def _build_manifest(
    processed_root: Path,
    output_dir: Path,
    regions: list[str],
    sources: list[str],
    months: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_months = set(months)
    output_entries: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "months": months,
        "regions": regions,
        "sources": sources,
        "source_root": str(processed_root),
        "manifest_root": str(output_dir),
        "region_source_summary": {},
        "missing_files": [],
        "sample_shapes": {},
    }

    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for region in regions:
        region_dir = processed_root / region
        region_manifest = _load_region_manifest(region_dir)
        region_summary: dict[str, dict[str, Any]] = {}
        month_coverage: dict[str, set[str]] = {source: set() for source in sources}
        present_counts: dict[str, int] = {source: 0 for source in sources}
        frame_counts: dict[str, int] = {source: 0 for source in sources}

        for entry in region_manifest:
            patch_id = entry["patch_id"]
            out: dict[str, Any] = {
                "region": region,
                "patch_id": f"{region}_{patch_id}",
                "source_patch_id": patch_id,
            }
            for source in sources:
                raw_paths = _paths_for_source(entry, source, region)
                selected = _filter_paths(raw_paths, source, allowed_months)
                rel_paths = [str(Path("..") / region / p) for p in selected]
                out[source] = rel_paths

                if selected:
                    present_counts[source] += 1
                    frame_counts[source] += len(selected)
                for path in selected:
                    month = _month_from_path(path)
                    if month is not None:
                        month_coverage[source].add(month)
                    full_path = region_dir / path
                    if not full_path.exists():
                        meta["missing_files"].append(str(full_path))
                    elif source not in meta["sample_shapes"]:
                        shape = _file_shape(full_path)
                        if shape is not None:
                            meta["sample_shapes"][source] = shape
            output_entries.append(out)

        for source in sources:
            covered = sorted(month_coverage[source])
            missing_months = [] if source == "worldcover" else [m for m in months if m not in covered]
            region_summary[source] = {
                "patches_with_source": present_counts[source],
                "total_frames": frame_counts[source],
                "covered_months": covered,
                "missing_months": missing_months,
            }
        summary[region] = region_summary

    meta["num_entries"] = len(output_entries)
    meta["region_source_summary"] = summary
    meta["missing_file_count"] = len(meta["missing_files"])
    return output_entries, meta


def main() -> None:
    args = _parse_args()
    manifest, meta = _build_manifest(
        processed_root=args.processed_root,
        output_dir=args.output_dir,
        regions=args.regions,
        sources=args.sources,
        months=args.months,
    )

    print(json.dumps(meta["region_source_summary"], indent=2, ensure_ascii=False))
    print(f"entries: {meta['num_entries']}")
    print(f"missing files: {meta['missing_file_count']}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest_v2_202512_202605.json"
    meta_path = args.output_dir / "manifest_v2_202512_202605.meta.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {manifest_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
