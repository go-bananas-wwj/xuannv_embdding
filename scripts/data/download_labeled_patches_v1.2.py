#!/usr/bin/env python
"""为 haidian/harbin 的 744 个标注 patch 下载 2025-12 至 2026-05 的 S2 输入。

使用 AEF official 2025 annual 的 patch meta（bbox + crs）确定每个 patch 的
UTM 边界，调用与 national 下载相同的 Earth Search 流程，输出到
/data2/xuannv_embedding/national/processed_labeled_v1.2。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyproj

# 复用 national 下载的核心函数
from scripts.data.download_national_patches import (
    DEFAULT_MONTHS,
    SOURCES,
    _download_source_month,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")


def load_patch_meta(meta_root: Path) -> pd.DataFrame:
    """加载 haidian/harbin 的 patch meta，转换为下载脚本使用的 DataFrame 格式。"""
    records: list[dict] = []
    for region in ["haidian", "harbin"]:
        meta_path = meta_root / f"{region}_patch_meta.json"
        if not meta_path.exists():
            logger.warning("Meta file not found: %s", meta_path)
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        for patch_id, info in meta.items():
            left, bottom, right, top = info["bbox"]
            epsg = int(info["crs"].replace("EPSG:", ""))
            records.append(
                {
                    "patch_id": patch_id,
                    "region": region,
                    "utm_epsg": epsg,
                    "bbox_left_utm": left,
                    "bbox_bottom_utm": bottom,
                    "bbox_right_utm": right,
                    "bbox_top_utm": top,
                    "aef_height": info["shape"][0],
                    "aef_width": info["shape"][1],
                }
            )
    return pd.DataFrame(records)


def process_patch(row: pd.Series, args: argparse.Namespace) -> bool:
    """处理单个 labeled patch：仅下载 S2 各月（教师 embedding 已存在）。"""
    from pystac_client import Client

    patch_id = row["patch_id"]
    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    ok = True
    for month in args.months:
        for source, assets in SOURCES.items():
            if not _download_source_month(
                row,
                month,
                source,
                assets,
                catalog,
                Path(args.output_root),
                region=row["region"],
                max_retries=args.max_retries,
                zero_fill_on_failure=args.zero_fill_on_failure,
                cloud_cover_threshold=args.cloud_cover_threshold,
                max_items=args.max_items,
            ):
                ok = False
    if ok:
        logger.info("Finished patch %s", patch_id)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-root",
        default="/data/xuannv_embedding/embeddings/aef_official_2025_annual",
        type=Path,
    )
    parser.add_argument(
        "--labeled-manifest",
        default="/data/xuannv_embedding/processed/manifest_labeled_all.json",
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        default="/data2/xuannv_embedding/national/processed_labeled_v1.2",
        type=Path,
    )
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="仅下载指定 region（默认全部）",
    )
    parser.add_argument(
        "--zero-fill-on-failure",
        action="store_true",
        help="下载/重试失败后使用零值填充，保证 patch 可被加入训练",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每个 source-month 失败后的重试次数",
    )
    parser.add_argument(
        "--cloud-cover-threshold",
        type=float,
        default=80.0,
        help="STAC 云量过滤阈值",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="每月每个 source 最多使用的 STAC item 数量",
    )
    args = parser.parse_args()

    labeled_ids: set[str] = set()
    if args.labeled_manifest.exists():
        with args.labeled_manifest.open("r", encoding="utf-8") as f:
            labeled_ids = {entry["patch_id"] for entry in json.load(f)}
        logger.info("Limiting to %d labeled patches", len(labeled_ids))

    df = load_patch_meta(args.meta_root)
    if args.regions:
        df = df[df["region"].isin(args.regions)]
    if labeled_ids:
        df = df[df["patch_id"].isin(labeled_ids)]
    if args.limit:
        df = df.head(args.limit)
    logger.info("Processing %d labeled patches", len(df))

    if args.max_workers <= 1:
        results = [process_patch(row, args) for _, row in df.iterrows()]
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_patch, row, args): i
                for i, row in df.iterrows()
            }
            results = [False] * len(df)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(
                        "Worker failed for patch %s: %s", df.iloc[idx]["patch_id"], exc
                    )

    success = sum(results)
    logger.info("Completed %d / %d patches successfully", success, len(df))


if __name__ == "__main__":
    main()
