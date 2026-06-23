#!/usr/bin/env python
"""为已下载的 labeled v1.2 S2 数据生成 manifest 并准备统计量文件。"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCES = ["s2", "s2_20m"]
REGIONS = ["haidian", "harbin"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national/processed_labeled_v1.2",
        type=Path,
    )
    parser.add_argument(
        "--statistics-dir",
        default="/data2/xuannv_embedding/national/statistics_labeled_v1.2",
        type=Path,
    )
    parser.add_argument(
        "--national-statistics-dir",
        default="/data2/xuannv_embedding/national/statistics/national",
        type=Path,
    )
    args = parser.parse_args()

    processed_root = args.output_root / "processed"
    manifest_entries: list[dict[str, Any]] = []

    for region in REGIONS:
        region_dir = processed_root / region / "patches"
        if not region_dir.exists():
            logger.warning("Region dir not found: %s", region_dir)
            continue

        # 收集每个 patch 的月度文件
        patch_files: dict[str, dict[str, list[Path]]] = {}
        for source in SOURCES:
            source_dir = region_dir / source
            if not source_dir.exists():
                continue
            for path in source_dir.glob("*.tif"):
                # filename: {source}_YYYYMMDD_{patch_id}.tif
                parts = path.stem.split("_", 3)
                if len(parts) < 3:
                    continue
                patch_id = "_".join(parts[2:])
                patch_files.setdefault(patch_id, {}).setdefault(source, []).append(path)

        for patch_id in sorted(patch_files):
            entry = {
                "patch_id": patch_id,
                "region": region,
            }
            for source in SOURCES:
                paths = sorted(patch_files[patch_id].get(source, []))
                entry[source] = [str(p.relative_to(processed_root / region)) for p in paths]
            manifest_entries.append(entry)

    manifest_path = processed_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d patches to %s", len(manifest_entries), manifest_path)

    # 拷贝 national 统计量到各 region
    if args.national_statistics_dir.exists():
        args.statistics_dir.mkdir(parents=True, exist_ok=True)
        for region in REGIONS:
            dst = args.statistics_dir / region
            dst.mkdir(parents=True, exist_ok=True)
            for stat_file in args.national_statistics_dir.glob("*_stats.json"):
                shutil.copy2(stat_file, dst / stat_file.name)
        logger.info("Copied national stats to %s", args.statistics_dir)
    else:
        logger.warning("National stats not found: %s", args.national_statistics_dir)


if __name__ == "__main__":
    main()
