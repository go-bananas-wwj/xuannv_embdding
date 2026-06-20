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
    try:
        class_map = json.loads(args.class_map)
    except json.JSONDecodeError as exc:
        logger.error("--class-map 必须是合法 JSON，例如 '{\"jiazhudongdi\": 1}'")
        raise SystemExit(1) from exc

    out_dir = args.out_dir
    raw_dir = out_dir / "labelme_raw"
    mask_dir = out_dir / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ref_crs: Any = None
    ref_transform: list[float] | None = None
    per_patch: list[dict[str, Any]] = []

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
            crs_obj = src.crs.to_string() if src.crs else None
            transform_obj = list(src.transform)

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

        if processed == 0:
            ref_crs = crs_obj
            ref_transform = transform_obj
        elif crs_obj != ref_crs or transform_obj != ref_transform:
            logger.warning(
                "%s 的 crs/transform 与第一张 patch 不一致",
                patch_id,
            )

        per_patch.append(
            {
                "patch_id": patch_id,
                "crs": crs_obj,
                "transform": transform_obj,
                "height": height,
                "width": width,
            }
        )
        processed += 1

    if processed == 0:
        logger.warning("未处理任何 patch，请检查 labelme 文件名、class-map 与 patch_dir")

    label_meta = {
        "task": args.task,
        "region": args.region,
        "labelme_rar": str(args.labelme_rar),
        "patch_dir": str(args.patch_dir),
        "class_map": class_map,
        "num_patches": processed,
        "mask_dir": str(mask_dir),
        "crs": ref_crs,
        "transform": ref_transform,
        "per_patch": per_patch,
    }
    with open(out_dir / "label_meta.json", "w", encoding="utf-8") as f:
        json.dump(label_meta, f, ensure_ascii=False, indent=2)

    logger.info("完成：处理 %d 张 mask，元数据写入 %s", processed, out_dir / "label_meta.json")


if __name__ == "__main__":
    main()
