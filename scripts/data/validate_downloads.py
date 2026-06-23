#!/usr/bin/env python
"""校验 national / labeled v1.2 下载结果，输出缺失文件列表与统计。

用法示例:
    python scripts/data/validate_downloads.py national
    python scripts/data/validate_downloads.py labeled --regions haidian harbin
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCES = ["s2", "s2_20m"]
DEFAULT_MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]


def _expected_tiff_exists(
    output_root: Path,
    region: str,
    source: str,
    month: str,
    patch_id: str,
) -> bool:
    date_str = month.replace("-", "") + "01"
    path = (
        output_root
        / "processed"
        / region
        / "patches"
        / source
        / f"{source}_{date_str}_{patch_id}.tif"
    )
    return path.exists() and path.stat().st_size > 0


def _tiff_is_valid(path: Path) -> bool:
    try:
        with rasterio.open(path) as src:
            data = src.read()
        return data.size > 0
    except Exception as exc:
        logger.debug("Invalid tiff %s: %s", path, exc)
        return False


def validate_national(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    input_csv = Path(args.input_csv)
    df = pd.read_csv(input_csv)

    complete = 0
    missing: list[dict] = []
    for _, row in df.iterrows():
        patch_id = row["patch_id"]
        emb_path = (
            output_root
            / "embeddings"
            / "national"
            / patch_id
            / "202512_embedding_map.pt"
        )
        emb_ok = emb_path.exists() and emb_path.stat().st_size > 0
        all_tiffs_ok = True
        for source in SOURCES:
            for month in args.months:
                ok = _expected_tiff_exists(output_root, "national", source, month, patch_id)
                if not ok:
                    all_tiffs_ok = False
                    missing.append(
                        {
                            "patch_id": patch_id,
                            "region": "national",
                            "source": source,
                            "month": month,
                            "type": "tiff_missing",
                        }
                    )
                else:
                    path = (
                        output_root
                        / "processed"
                        / "national"
                        / "patches"
                        / source
                        / f"{source}_{month.replace('-', '')}01_{patch_id}.tif"
                    )
                    if not _tiff_is_valid(path):
                        all_tiffs_ok = False
                        missing.append(
                            {
                                "patch_id": patch_id,
                                "region": "national",
                                "source": source,
                                "month": month,
                                "type": "tiff_invalid",
                            }
                        )
        if emb_ok and all_tiffs_ok:
            complete += 1

    logger.info(
        "National: %d / %d patches complete (%.1f%%)",
        complete,
        len(df),
        100.0 * complete / len(df) if len(df) else 0,
    )
    if missing:
        logger.info("Missing/invalid items: %d", len(missing))
        missing_path = output_root / "missing_national.csv"
        pd.DataFrame(missing).to_csv(missing_path, index=False)
        logger.info("Wrote %s", missing_path)


def validate_labeled(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    ref_manifest = Path(args.labeled_manifest)
    if not ref_manifest.exists():
        logger.warning("Reference manifest not found: %s", ref_manifest)
        return

    with ref_manifest.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    complete = 0
    missing: list[dict] = []
    for entry in entries:
        patch_id = entry["patch_id"]
        region = entry.get("region")
        if region is None:
            # infer region from first s2 path
            sample_path = entry.get("s2", [""])[0]
            region = sample_path.split("/")[0] if "/" in sample_path else "unknown"
        if args.regions and region not in args.regions:
            continue
        all_ok = True
        for source in SOURCES:
            for month in args.months:
                if not _expected_tiff_exists(output_root, region, source, month, patch_id):
                    all_ok = False
                    missing.append(
                        {
                            "patch_id": patch_id,
                            "region": region,
                            "source": source,
                            "month": month,
                            "type": "tiff_missing",
                        }
                    )
                else:
                    path = (
                        output_root
                        / "processed"
                        / region
                        / "patches"
                        / source
                        / f"{source}_{month.replace('-', '')}01_{patch_id}.tif"
                    )
                    if not _tiff_is_valid(path):
                        all_ok = False
                        missing.append(
                            {
                                "patch_id": patch_id,
                                "region": region,
                                "source": source,
                                "month": month,
                                "type": "tiff_invalid",
                            }
                        )
        if all_ok:
            complete += 1

    logger.info(
        "Labeled: %d / %d patches complete (%.1f%%)",
        complete,
        len(entries),
        100.0 * complete / len(entries) if entries else 0,
    )
    if missing:
        logger.info("Missing/invalid items: %d", len(missing))
        missing_path = output_root / "missing_labeled.csv"
        pd.DataFrame(missing).to_csv(missing_path, index=False)
        logger.info("Wrote %s", missing_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    national_p = subparsers.add_parser("national")
    national_p.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national",
        type=Path,
    )
    national_p.add_argument(
        "--input-csv",
        default="/data2/xuannv_embedding/national/sample_national_patches.csv",
        type=Path,
    )
    national_p.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)

    labeled_p = subparsers.add_parser("labeled")
    labeled_p.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national/processed_labeled_v1.2",
        type=Path,
    )
    labeled_p.add_argument(
        "--labeled-manifest",
        default="/data/xuannv_embedding/processed/manifest_labeled_all.json",
        type=Path,
    )
    labeled_p.add_argument("--regions", nargs="+", default=None)
    labeled_p.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)

    args = parser.parse_args()
    if args.command == "national":
        validate_national(args)
    else:
        validate_labeled(args)


if __name__ == "__main__":
    main()
