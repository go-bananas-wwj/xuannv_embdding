#!/usr/bin/env python
"""根据已下载的 national patch 生成 manifest.json。

扫描 processed/national/patches/{s2,s2_20m} 中的月度 GeoTIFF，
并检查 embeddings/national/{patch_id}/202512_embedding_map.pt 是否存在，
输出符合 MonthlyEmbeddingDataset 约定的 manifest。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCES = ["s2", "s2_20m"]


def _collect_source_paths(source_dir: Path, patch_id: str) -> list[Path]:
    """收集某个 patch 在 source 目录下的所有月度 TIFF，按文件名排序。"""
    if not source_dir.exists():
        return []
    paths = sorted(source_dir.glob(f"*_{patch_id}.tif"))
    return paths


def generate_manifest(
    output_root: Path,
    input_csv: Path,
    require_embedding: bool = True,
) -> list[dict]:
    """生成 national manifest 列表。"""
    df = pd.read_csv(input_csv)
    processed_root = output_root / "processed" / "national"
    manifest_entries: list[dict] = []

    for _, row in df.iterrows():
        patch_id = row["patch_id"]
        entry: dict = {"patch_id": patch_id, "region": "national"}

        if require_embedding:
            emb_path = (
                output_root
                / "embeddings"
                / "national"
                / patch_id
                / "202512_embedding_map.pt"
            )
            if not emb_path.exists():
                logger.debug("Skipping %s: teacher embedding missing", patch_id)
                continue

        has_any = False
        for source in SOURCES:
            source_dir = processed_root / "patches" / source
            paths = _collect_source_paths(source_dir, patch_id)
            if paths:
                has_any = True
                entry[source] = [str(p.relative_to(processed_root)) for p in paths]
            else:
                entry[source] = []

        if has_any:
            manifest_entries.append(entry)

    return manifest_entries


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
        "--no-require-embedding",
        action="store_true",
        help="即使教师 embedding 缺失也包含 patch（用于尚未下载完时测试）",
    )
    args = parser.parse_args()

    manifest = generate_manifest(
        args.output_root,
        args.input_csv,
        require_embedding=not args.no_require_embedding,
    )

    manifest_path = args.output_root / "processed" / "national" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %d patches to %s",
        len(manifest),
        manifest_path,
    )


if __name__ == "__main__":
    main()
