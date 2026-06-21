#!/usr/bin/env python3
"""解压并处理哈尔滨新区带标注的 patch 数据，生成与当前 pipeline 兼容的 mask 与目录结构。

支持多任务：通过 --task 与 --class-map 为每个类别独立生成 mask。"""

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


_TASK_DEFAULT_CLASS_MAP: dict[str, dict[str, int]] = {
    "construction": {
        "construction site": 1,
        "jiazhudongdi": 1,
        "gongdi": 1,
    },
    "building_change": {"building change": 1},
    "farm_change": {"farm_change": 1},
    "river_building": {"river_building": 1},
    "river_farm": {"river_farm": 1},
    "river_in": {"river_in": 1},
    "river_ship": {"river_ship": 1},
    "rubbish": {"rubbish": 1},
}


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
        class_map = _TASK_DEFAULT_CLASS_MAP["construction"]
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


def compute_label_distribution(raw_root: Path, out_path: Path | None = None) -> dict:
    """统计 raw_root 下所有 LabelMe JSON 中每个 shape label 的出现次数。"""
    label_counts: dict[str, int] = {}
    total_files = 0
    total_shapes = 0
    for p in _find_files(raw_root, "*.json"):
        total_files += 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("无法读取 %s: %s", p, e)
            continue
        for s in data.get("shapes", []):
            label = str(s.get("label", "")).strip().lower()
            if not label:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
            total_shapes += 1

    distribution = {
        "raw_root": str(raw_root),
        "total_files": total_files,
        "total_shapes": total_shapes,
        "label_counts": dict(sorted(label_counts.items())),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(distribution, f, ensure_ascii=False, indent=2)
        logger.info("标签分布已保存至 %s", out_path)
    return distribution


def process_harbin(
    raw_root: Path,
    output_root: Path,
    target_mask_size: tuple[int, int] = (128, 128),
    class_map: dict[str, int] | None = None,
) -> None:
    """处理哈尔滨标注数据，按 task 输出 mask 与原始 LabelMe 备份。

    输出目录结构：
        output_root/masks/         - 栅格化 mask
        output_root/labelme_raw/   - 原始 JSON 备份
    """
    if class_map is None:
        class_map = _TASK_DEFAULT_CLASS_MAP["construction"]

    labelme_dir = output_root / "labelme_raw"
    mask_dir = output_root / "masks"
    # 共享高分辨率影像目录，位于 processed/harbin/patches/highres_optical
    img_out_dir = output_root.parent.parent / "patches" / "highres_optical"

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

        # 先收集该 patch 实际有标注的月份
        present_months: list[str] = []
        generated_masks: dict[str, Path] = {}

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

            # 合并所有象限标签为多边形，栅格化到 target_mask_size
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
            present_months.append(month)
            generated_masks[month] = mask_path

            # 可选：复制参考影像到共享 highres_optical 目录
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
                "source": "annotated",
            }
            if dst_img is not None:
                stat["image"] = str(dst_img.relative_to(output_root))
            stats.append(stat)
            logger.info("%s %s 处理完成: 正样本比例 %.3f", pid, month, ratio)

        # 补齐缺失月份：如果同一个 patch 只标注了一个月份，把 mask 复制到另一个月份
        known_months = {"202512", "202605"}
        annotated_months = set(present_months)
        missing_months = known_months - annotated_months
        if len(annotated_months) == 1 and missing_months:
            src_month = present_months[0]
            src_path = generated_masks[src_month]
            for missing_month in missing_months:
                dst_path = mask_dir / f"{pid}_{missing_month}.tif"
                shutil.copy2(src_path, dst_path)
                with rasterio.open(dst_path) as src:
                    mask = src.read(1)
                ratio = float((mask > 0).sum() / mask.size)
                stat = {
                    "patch_id": pid,
                    "month": missing_month,
                    "label_count": 0,
                    "mask": str(dst_path.relative_to(output_root)),
                    "positive_ratio": ratio,
                    "source": f"copied_from_{src_month}",
                }
                stats.append(stat)
                logger.info(
                    "%s %s 缺失标注，已从 %s 复制 mask: 正样本比例 %.3f",
                    pid, missing_month, src_month, ratio,
                )

    summary_path = output_root / "harbin_patches_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("共处理 %d 个 patch-month，汇总保存至 %s", len(stats), summary_path)

    # 统计所有原始 JSON 的标签分布，保存到 labels/label_distribution.json
    distribution_path = output_root.parent / "label_distribution.json"
    compute_label_distribution(raw_root, distribution_path)


def _default_class_map(task: str) -> dict[str, int]:
    key = task.strip().lower()
    if key not in _TASK_DEFAULT_CLASS_MAP:
        raise ValueError(
            f"未知 task '{task}'。已知任务: {list(_TASK_DEFAULT_CLASS_MAP.keys())}"
        )
    return _TASK_DEFAULT_CLASS_MAP[key]


def _parse_class_map_arg(json_str: str) -> dict[str, int]:
    try:
        class_map = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--class-map 必须是合法 JSON 字符串: {e}")
    if not isinstance(class_map, dict):
        raise argparse.ArgumentTypeError("--class-map 必须是一个 JSON object")
    return {str(k).strip().lower(): int(v) for k, v in class_map.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True, help="解压后的 haerbin_patches 根目录")
    p.add_argument(
        "--output-root", type=Path, required=True, help="输出根目录，例如 processed/harbin/labels/<task>"
    )
    p.add_argument(
        "--mask-size", type=int, nargs=2, default=[128, 128], help="输出 mask 尺寸 (H W)"
    )
    p.add_argument(
        "--task",
        type=str,
        default="construction",
        help="任务名称（用于默认 class-map，当未提供 --class-map 时）",
    )
    p.add_argument(
        "--class-map",
        type=str,
        default=None,
        help='JSON 字符串，例如：\'{"construction site":1,"jiazhudongdi":1,"gongdi":1}\'',
    )
    args = p.parse_args()

    if args.class_map is not None:
        class_map = _parse_class_map_arg(args.class_map)
    else:
        class_map = _default_class_map(args.task)

    process_harbin(args.raw_root, args.output_root, tuple(args.mask_size), class_map)


if __name__ == "__main__":
    main()
