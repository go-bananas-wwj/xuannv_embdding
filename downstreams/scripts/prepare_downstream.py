#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import rasterio
from downstreams.data.label_loaders import (
    get_reference_patch_path,
    parse_patch_id_from_labelme_name,
    rasterize_labelme,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解压 labelme 标注并生成对齐 mask")
    p.add_argument("--task", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--labelme-rar", type=Path, required=True)
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--class-map", type=str, default='{"jiazhudongdi": 1}')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    class_map = json.loads(args.class_map)

    out_dir = args.out_dir
    raw_dir = out_dir / "labelme_raw"
    mask_dir = out_dir / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    crs: Any = None
    transform: list[float] | None = None

    # 解压 rar
    logger.info("解压 %s -> %s", args.labelme_rar, raw_dir)
    shutil.rmtree(raw_dir, ignore_errors=True)
    subprocess.run(
        ["bsdtar", "xf", str(args.labelme_rar), "-C", str(raw_dir)],
        check=True,
    )

    label_files = sorted(raw_dir.rglob("*.json"))
    logger.info("找到 %d 个 labelme json", len(label_files))

    processed = 0
    for label_path in label_files:
        try:
            patch_id = parse_patch_id_from_labelme_name(label_path)
        except ValueError as exc:
            logger.warning("跳过 %s: %s", label_path, exc)
            continue

        ref_path = get_reference_patch_path(args.patch_dir, patch_id)
        if ref_path is None:
            logger.warning("找不到参考影像: %s", patch_id)
            continue

        with rasterio.open(ref_path) as src:
            height, width = src.height, src.width
            crs = src.crs.to_string() if src.crs else None
            transform = list(src.transform)

        mask = rasterize_labelme(label_path, (height, width), class_map)
        out_mask = mask_dir / f"{patch_id}.tif"
        with rasterio.open(
            out_mask,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(mask, 1)
        processed += 1

    label_meta = {
        "task": args.task,
        "region": args.region,
        "labelme_rar": str(args.labelme_rar),
        "patch_dir": str(args.patch_dir),
        "class_map": class_map,
        "num_patches": processed,
        "mask_dir": str(mask_dir),
        "crs": crs,
        "transform": transform,
    }
    with open(out_dir / "label_meta.json", "w", encoding="utf-8") as f:
        json.dump(label_meta, f, ensure_ascii=False, indent=2)

    logger.info("完成：处理 %d 张 mask，元数据写入 %s", processed, out_dir / "label_meta.json")


if __name__ == "__main__":
    main()
