#!/usr/bin/env python3
"""解压并处理哈尔滨新区带标注的 patch 数据，生成与当前 pipeline 兼容的 mask 与目录结构。"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def _group_by_patch_id(paths: list[Path], root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in paths:
        rel = p.relative_to(root)
        # patch_000xxx/...
        m = re.match(r"(patch_\d+)", rel.parts[0])
        if not m:
            continue
        pid = m.group(1)
        groups.setdefault(pid, []).append(p)
    return groups


_QUADRANT_ORDER = {
    "01": (0, 0),
    "02": (0, 1),
    "03": (1, 0),
    "04": (1, 1),
}


def _parse_label_stem(stem: str) -> tuple[str, str, str] | None:
    """Parse '01_20260501' -> ('01', '20260501', '202605')."""
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    quadrant, date = parts[0], parts[1]
    return quadrant, date, date[:6]


def _group_by_patch_and_date(paths: list[Path], root: Path) -> dict[str, dict[str, list[Path]]]:
    """{patch_id: {month: [label_paths]}}."""
    groups: dict[str, dict[str, list[Path]]] = {}
    for p in paths:
        rel = p.relative_to(root)
        m = re.match(r"(patch_\d+)", rel.parts[0])
        if not m:
            continue
        pid = m.group(1)
        parsed = _parse_label_stem(p.stem)
        if parsed is None:
            continue
        quadrant, date, month = parsed
        if quadrant not in _QUADRANT_ORDER:
            continue
        groups.setdefault(pid, {}).setdefault(month, []).append(p)
    return groups


def _reference_image(images: list[Path]) -> Path | None:
    """优先选择第一个 .tif 参考影像；不存在则选第一个可用图像。"""
    candidates = [
        p for p in images if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    ]
    if not candidates:
        return None
    # 优先 tif
    tifs = [p for p in candidates if p.suffix.lower() in (".tif", ".tiff")]
    if tifs:
        return tifs[0]
    return candidates[0]


def _infer_full_size(quadrant_dims: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Infer full image (W, H) from present quadrant dimensions.

    Assumes 2x2 layout: 01 top-left, 02 top-right, 03 bottom-left, 04 bottom-right.
    """
    left_w = max([quadrant_dims[q][0] for q in ("01", "03") if q in quadrant_dims] or [0])
    right_w = max([quadrant_dims[q][0] for q in ("02", "04") if q in quadrant_dims] or [0])
    top_h = max([quadrant_dims[q][1] for q in ("01", "02") if q in quadrant_dims] or [0])
    bottom_h = max([quadrant_dims[q][1] for q in ("03", "04") if q in quadrant_dims] or [0])

    if left_w == 0 and right_w == 0:
        raise ValueError("no quadrant widths available")
    if top_h == 0 and bottom_h == 0:
        raise ValueError("no quadrant heights available")

    if left_w and right_w and abs(left_w - right_w) / max(left_w, right_w) > 0.05:
        logger.warning(
            "Left/right quadrant widths differ significantly (%d vs %d)", left_w, right_w
        )
    if top_h and bottom_h and abs(top_h - bottom_h) / max(top_h, bottom_h) > 0.05:
        logger.warning(
            "Top/bottom quadrant heights differ significantly (%d vs %d)", top_h, bottom_h
        )

    full_w = (left_w if left_w else right_w) + (right_w if right_w else left_w)
    full_h = (top_h if top_h else bottom_h) + (bottom_h if bottom_h else top_h)
    return full_w, full_h


def _rasterize_quadrant_labelme(
    label_path: Path,
    quadrant: str,
    full_size: tuple[int, int],
    out_shape: tuple[int, int],
    class_map: dict[str, int] | None = None,
) -> np.ndarray:
    if class_map is None:
        class_map = {
            "jiazhudongdi": 1,
            "gongdi": 1,
            "construction site": 1,
        }
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_h = data.get("imageHeight", 1)
    img_w = data.get("imageWidth", 1)

    q_row, q_col = _QUADRANT_ORDER[quadrant]
    full_w, full_h = full_size

    if q_col == 0:
        x0 = 0
        x1 = max(img_w, full_w // 2)
    else:
        x1 = full_w
        x0 = full_w - max(img_w, full_w - (full_w // 2))
    if q_row == 0:
        y0 = 0
        y1 = max(img_h, full_h // 2)
    else:
        y1 = full_h
        y0 = full_h - max(img_h, full_h - (full_h // 2))

    q_w = x1 - x0
    q_h = y1 - y0

    scale_x = out_shape[1] * (q_w / full_w) / max(img_w, 1)
    scale_y = out_shape[0] * (q_h / full_h) / max(img_h, 1)
    offset_x = out_shape[1] * (x0 / full_w)
    offset_y = out_shape[0] * (y0 / full_h)

    geometries = []
    for s in data.get("shapes", []):
        label = s.get("label", "").strip().lower()
        if label not in class_map:
            continue
        points = s.get("points", [])
        if len(points) < 3:
            continue
        pts = [(x * scale_x + offset_x, y * scale_y + offset_y) for x, y in points]
        geom = Polygon(pts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid and not geom.is_empty:
            geometries.append((geom, class_map[label]))

    mask = np.zeros(out_shape, dtype=np.uint8)
    if geometries:
        mask = rasterize(
            geometries,
            out_shape=out_shape,
            fill=0,
            default_value=0,
            dtype=np.uint8,
            all_touched=True,
        )
    return mask


def _save_mask(mask: np.ndarray, out_path: Path, profile_template: dict | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": mask.dtype,
        "crs": None,
        "transform": rasterio.Affine.identity(),
        "compress": "lzw",
    }
    if profile_template:
        profile.update(profile_template)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)


def process_harbin(
    raw_root: Path,
    output_root: Path,
    target_mask_size: tuple[int, int] = (128, 128),
    class_map: dict[str, int] | None = None,
) -> None:
    labelme_dir = output_root / "labels" / "construction" / "labelme_raw"
    mask_dir = output_root / "labels" / "construction" / "masks"
    img_out_dir = output_root / "patches" / "highres_optical"

    label_paths = _find_files(raw_root, "*.json")
    image_paths = (
        _find_files(raw_root, "*.tif")
        + _find_files(raw_root, "*.png")
        + _find_files(raw_root, "*.jpg")
    )
    label_groups = _group_by_patch_and_date(label_paths, raw_root)
    image_groups = _group_by_patch_id(image_paths, raw_root)

    stats = []
    for pid in sorted(label_groups):
        month_dict = label_groups[pid]
        images = image_groups.get(pid, [])
        ref_img = _reference_image(images)

        for month in sorted(month_dict):
            labels = month_dict[month]
            if not labels:
                continue

            # 复制 LabelMe 原始标签
            patch_label_dir = labelme_dir / pid / month
            patch_label_dir.mkdir(parents=True, exist_ok=True)
            for lp in labels:
                shutil.copy2(lp, patch_label_dir / lp.name)

            # 从各象限 JSON 的 imageWidth/imageHeight 推断完整尺寸
            quadrant_dims: dict[str, tuple[int, int]] = {}
            for lp in labels:
                parsed = _parse_label_stem(lp.stem)
                if parsed is None:
                    continue
                quadrant = parsed[0]
                with open(lp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                quadrant_dims[quadrant] = (
                    data.get("imageWidth", 1),
                    data.get("imageHeight", 1),
                )
            try:
                full_size = _infer_full_size(quadrant_dims)
            except ValueError as e:
                logger.warning("%s %s 无法推断完整尺寸: %s", pid, month, e)
                continue

            # 合并所有象限标签为多边形，栅格化到 128x128
            mask = np.zeros(target_mask_size, dtype=np.uint8)
            for lp in labels:
                parsed = _parse_label_stem(lp.stem)
                if parsed is None:
                    continue
                quadrant = parsed[0]
                m = _rasterize_quadrant_labelme(
                    lp, quadrant, full_size, target_mask_size, class_map
                )
                mask = np.maximum(mask, m)

            mask_path = mask_dir / f"{pid}_{month}.tif"
            _save_mask(mask, mask_path)

            # 可选：复制参考影像到 highres_optical 目录
            dst_img = None
            if ref_img:
                patch_img_dir = img_out_dir / pid
                patch_img_dir.mkdir(parents=True, exist_ok=True)
                dst_img = patch_img_dir / ref_img.name
                if not dst_img.exists():
                    shutil.copy2(ref_img, dst_img)

            ratio = float((mask > 0).sum() / mask.size)
            stat = {
                "patch_id": pid,
                "month": month,
                "label_count": len(labels),
                "mask": str(mask_path.relative_to(output_root)),
                "positive_ratio": ratio,
            }
            if dst_img is not None:
                stat["image"] = str(dst_img.relative_to(output_root))
            stats.append(stat)
            logger.info("%s %s 处理完成: 正样本比例 %.3f", pid, month, ratio)

    summary_path = output_root / "harbin_patches_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("共处理 %d 个 patch-month，汇总保存至 %s", len(stats), summary_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="解压后的 haerbin_patches 根目录")
    p.add_argument(
        "--output-root", type=Path, required=True, help="输出到 processed/harbin_patches"
    )
    p.add_argument(
        "--mask-size", type=int, nargs=2, default=[128, 128], help="输出 mask 尺寸 (H W)"
    )
    args = p.parse_args()
    process_harbin(args.raw_root, args.output_root, tuple(args.mask_size))


if __name__ == "__main__":
    main()
