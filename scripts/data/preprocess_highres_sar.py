#!/usr/bin/env python3
"""预处理海淀高分辨率 SAR ORG 产品，按低分辨率 S2 网格切分为 patches。

输入为 ModelScope 下载的 radiometrically-calibrated、geocoded ORG 产品：
  BC*-SM-ORG-*-2SVV-*.zip
每个 zip 解压后包含 VV 极化 GeoTIFF（部分产品可能同时包含 VH）。

处理流程：
1. 按日期分组（同一日期可能包含多景相邻 swath，需要合并）。
2. 将每个 zip 解压到 _unpacked/{zip_stem}/。
3. 将 VV（及 VH）重投影到 EPSG:32650、约 3 m 分辨率，并裁剪到
   1280 m × 1280 m 的低分辨率 patch 边界（427×427 像素）。
4. 线性幅度转 dB、按典型范围归一化到 [0, 1]、生成有效掩膜。
5. 写出 highres_sar_{YYYYMMDD}_{patch_id}.tif 与对应 _mask.tif。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
logging.getLogger("rasterio.errors").setLevel(logging.ERROR)

NODATA = 0.0
MIN_VALID_RATIO = 0.3
MASTER_RES = 10.0
SAR_RES = 3.0
SAR_OUT_SIZE = 427  # ceil(1280 / 3)

# 典型 dB 范围，用于归一化
POL_RANGES = {
    "VV": {"low": -25.0, "high": 0.0},
    "HH": {"low": -25.0, "high": 0.0},
    "VH": {"low": -35.0, "high": 0.0},
    "HV": {"low": -35.0, "high": 0.0},
}


def load_config(path: Path) -> dict[str, Any]:
    """加载低分辨率预处理配置。"""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.setdefault("patch_size_m", 1280.0)
    config.setdefault("aoi_path", f"configs/regions/{config['region']}.geojson")
    return config


def _generate_patch_grid_from_aoi(
    aoi_path: Path,
    patch_size_m: float,
    dst_crs: CRS,
) -> list[dict[str, Any]]:
    """基于 AOI GeoJSON 生成 patch 网格（与 preprocess.py 一致）。"""
    aoi = gpd.read_file(aoi_path)
    aoi = aoi.to_crs(dst_crs)
    left, bottom, right, top = aoi.total_bounds

    # 吸附到 10 m 主网格
    left = math.floor(left / MASTER_RES) * MASTER_RES
    bottom = math.floor(bottom / MASTER_RES) * MASTER_RES
    right = math.ceil(right / MASTER_RES) * MASTER_RES
    top = math.ceil(top / MASTER_RES) * MASTER_RES

    n_rows = int(np.ceil((top - bottom) / patch_size_m))
    patches: list[dict[str, Any]] = []
    for idx, pbounds in enumerate(
        make_patch_grid((left, bottom, right, top), patch_size_m)
    ):
        col = idx // n_rows
        row = idx % n_rows
        patch_id = f"p{col:03d}_r{row:03d}"
        patches.append({"patch_id": patch_id, "bounds": pbounds, "col": col, "row": row})
    return patches


def _generate_patch_grid_from_s2_patches(
    s2_patch_dir: Path,
) -> list[dict[str, Any]]:
    """基于已存在的 S2 patches 生成参考网格，确保严格对齐。"""
    files = sorted(s2_patch_dir.glob("s2_*.tif"))
    # 优先使用有最多 patches 的日期
    date_counts: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = re.search(r"s2_(\d{8})_p\d{3}_r\d{3}\.tif$", f.name)
        if m:
            date_counts[m.group(1)].append(f)

    if not date_counts:
        raise ValueError(f"在 {s2_patch_dir} 未找到 S2 patch 文件")

    ref_date = max(date_counts, key=lambda d: len(date_counts[d]))
    ref_files = sorted(date_counts[ref_date])

    patches: list[dict[str, Any]] = []
    for f in ref_files:
        m = re.search(r"s2_\d{8}_(p\d{3}_r\d{3})\.tif$", f.name)
        if not m:
            continue
        patch_id = m.group(1)
        with rasterio.open(f) as src:
            bounds = tuple(src.bounds)
        # 从 patch_id 解析 col/row
        cm = re.match(r"p(\d{3})_r(\d{3})", patch_id)
        col = int(cm.group(1)) if cm else 0
        row = int(cm.group(2)) if cm else 0
        patches.append({"patch_id": patch_id, "bounds": bounds, "col": col, "row": row})

    return patches


def generate_patch_grid(
    config: dict[str, Any],
    dst_crs: CRS,
    output_root: Path,
) -> list[dict[str, Any]]:
    """生成 patch 网格，优先使用已存在的 S2 patches 作为参考。"""
    s2_patch_dir = output_root / "patches" / "s2"
    if s2_patch_dir.exists() and any(s2_patch_dir.glob("s2_*.tif")):
        logger.info("使用 S2 patches 作为参考网格: %s", s2_patch_dir)
        return _generate_patch_grid_from_s2_patches(s2_patch_dir)

    aoi_path = Path(config["aoi_path"])
    logger.info("使用 AOI 生成参考网格: %s", aoi_path)
    return _generate_patch_grid_from_aoi(
        aoi_path, float(config["patch_size_m"]), dst_crs
    )


def discover_org_zips(input_root: Path) -> dict[str, list[Path]]:
    """递归发现 ORG 产品 zip，并按日期分组。"""
    pattern = re.compile(r"^BC\d+-SM-ORG-2SVV-.*\.zip$", re.IGNORECASE)
    zips = [f for f in input_root.rglob("*.zip") if pattern.match(f.name)]

    dates: dict[str, list[Path]] = defaultdict(list)
    for zf in zips:
        m = re.search(r"(\d{8})T", zf.name)
        if not m:
            logger.warning("无法从文件名解析日期: %s", zf.name)
            continue
        dates[m.group(1)].append(zf)

    for date_str in dates:
        dates[date_str] = sorted(dates[date_str])

    logger.info(
        "发现 %d 个 ORG zip，涵盖 %d 个日期", sum(len(v) for v in dates.values()), len(dates)
    )
    return dict(dates)


def extract_zip(zip_path: Path, unpack_root: Path, overwrite: bool) -> Path:
    """解压 zip 到 _unpacked/{zip_stem}/，返回解压目录。"""
    unpack_dir = unpack_root / zip_path.stem
    tiff_files = list(unpack_dir.glob("*.tif*")) if unpack_dir.exists() else []
    if unpack_dir.exists() and not overwrite and tiff_files:
        logger.debug("已解压且包含影像，跳过: %s", zip_path.name)
        return unpack_dir

    unpack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(unpack_dir)
    logger.debug("解压完成: %s -> %s", zip_path.name, unpack_dir)
    return unpack_dir


def classify_polarization(tiff_path: Path) -> str | None:
    """根据文件名判断极化方式（VV/VH）。"""
    name = tiff_path.stem.lower()
    if "-vv-" in name or "_vv_" in name or name.endswith("_vv"):
        return "VV"
    if "-vh-" in name or "_vh_" in name or name.endswith("_vh"):
        return "VH"
    # 兜底：通过路径中出现顺序判断
    if "vv" in name:
        return "VV"
    if "vh" in name:
        return "VH"
    logger.warning("无法判断极化方式: %s", tiff_path.name)
    return None


def sar_to_db(linear: np.ndarray) -> np.ndarray:
    """线性幅度转 dB。"""
    return 10.0 * np.log10(np.maximum(linear, 0.0) + 1e-10)


def normalize_sar(db: np.ndarray, pol: str) -> np.ndarray:
    """将 dB 值按典型范围归一化到 [0, 1] 并 clip。"""
    cfg = POL_RANGES.get(pol, POL_RANGES["VV"])
    low, high = cfg["low"], cfg["high"]
    return np.clip((db - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _reproject_patch(
    src_path: Path,
    bounds: tuple[float, float, float, float],
    dst_crs: CRS,
) -> np.ndarray:
    """将单极化源影像裁剪/重投影到指定 patch 边界。"""
    patch_array = np.empty((1, SAR_OUT_SIZE, SAR_OUT_SIZE), dtype="float32")
    dst_transform = from_bounds(*bounds, width=SAR_OUT_SIZE, height=SAR_OUT_SIZE)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=patch_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata if src.nodata is not None else NODATA,
            dst_nodata=NODATA,
            num_threads=2,
        )
    return patch_array[0]


def _process_patch(
    patch: dict[str, Any],
    pol_files: dict[str, list[Path]],
    dst_crs: CRS,
) -> dict[str, Any] | None:
    """处理单个 patch：合并多 swath、转 dB、归一化、生成 mask。"""
    bounds = patch["bounds"]
    patch_id = patch["patch_id"]

    pol_arrays: dict[str, np.ndarray] = {}
    pol_masks: dict[str, np.ndarray] = {}

    for pol in ["VV", "VH"]:
        files = pol_files.get(pol, [])
        if not files:
            continue

        # 合并同一极化多景数据：逐源重投影后累加并计数
        sum_arr = np.zeros((SAR_OUT_SIZE, SAR_OUT_SIZE), dtype="float64")
        count_arr = np.zeros((SAR_OUT_SIZE, SAR_OUT_SIZE), dtype="uint8")

        for src_path in files:
            # 快速包围盒过滤
            with rasterio.open(src_path) as src:
                src_bounds_dst = transform_bounds(src.crs, dst_crs, *src.bounds)
            if (
                src_bounds_dst[0] >= bounds[2]
                or src_bounds_dst[2] <= bounds[0]
                or src_bounds_dst[1] >= bounds[3]
                or src_bounds_dst[3] <= bounds[1]
            ):
                continue

            raw_patch = _reproject_patch(src_path, bounds, dst_crs)
            db_patch = sar_to_db(raw_patch)
            norm_patch = normalize_sar(db_patch, pol)

            finite = np.isfinite(norm_patch)
            valid = finite & (norm_patch != NODATA)
            sum_arr[valid] += norm_patch[valid]
            count_arr[valid] += 1

        if count_arr.max() == 0:
            # 该极化无有效数据
            pol_arrays[pol] = np.zeros((SAR_OUT_SIZE, SAR_OUT_SIZE), dtype="float32")
            pol_masks[pol] = np.zeros((SAR_OUT_SIZE, SAR_OUT_SIZE), dtype="bool")
        else:
            avg = np.divide(
                sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr > 0
            ).astype("float32")
            pol_arrays[pol] = avg
            pol_masks[pol] = count_arr > 0

    if "VV" not in pol_arrays:
        return {
            "patch_id": patch_id,
            "status": "skipped",
            "reason": "no_VV",
        }

    # 有效掩膜：所有可用极化均有效；若 VH 不存在则只要求 VV 有效
    valid_mask = pol_masks["VV"].copy()
    if "VH" in pol_masks:
        valid_mask &= pol_masks["VH"]

    valid_ratio = float(valid_mask.mean())
    if valid_ratio < MIN_VALID_RATIO:
        return {
            "patch_id": patch_id,
            "status": "skipped",
            "reason": "low_valid_ratio",
            "valid_ratio": valid_ratio,
        }

    # 组装输出数组：VV 必备，VH 可选
    bands = [pol_arrays["VV"]]
    band_descriptions = ["VV"]
    if "VH" in pol_arrays:
        bands.append(pol_arrays["VH"])
        band_descriptions.append("VH")

    out_array = np.stack(bands, axis=0).astype("float32")
    return {
        "patch_id": patch_id,
        "status": "written",
        "valid_ratio": valid_ratio,
        "array": out_array,
        "mask": valid_mask.astype("uint8"),
        "band_descriptions": band_descriptions,
        "bounds": bounds,
    }


def _write_patch(
    patch: dict[str, Any],
    out_dir: Path,
    date_str: str,
    dst_crs: CRS,
    overwrite: bool,
) -> dict[str, Any]:
    """将 patch 结果写出为 GeoTIFF 与 mask。"""
    patch_id = patch["patch_id"]
    out_path = out_dir / f"highres_sar_{date_str}_{patch_id}.tif"
    mask_path = out_dir / f"highres_sar_{date_str}_{patch_id}_mask.tif"

    if out_path.exists() and mask_path.exists() and not overwrite:
        return {"patch_id": patch_id, "status": "exists"}

    out_dir.mkdir(parents=True, exist_ok=True)

    array = patch["array"]
    bounds = patch["bounds"]
    dst_transform = from_bounds(*bounds, width=SAR_OUT_SIZE, height=SAR_OUT_SIZE)

    profile = {
        "driver": "GTiff",
        "height": SAR_OUT_SIZE,
        "width": SAR_OUT_SIZE,
        "count": array.shape[0],
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array)
        for idx, desc in enumerate(patch["band_descriptions"], start=1):
            dst.set_band_description(idx, desc)

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(patch["mask"], 1)

    return {
        "patch_id": patch_id,
        "status": "written",
        "valid_ratio": patch["valid_ratio"],
    }


def process_date(
    date_str: str,
    zip_paths: list[Path],
    patches: list[dict[str, Any]],
    dst_crs: CRS,
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """处理单个日期的所有 ORG zip。"""
    logger.info("[%s] 开始处理 %d 个 zip", date_str, len(zip_paths))
    unpack_root = input_root / "_unpacked"

    # 解压并收集极化文件
    pol_files: dict[str, list[Path]] = defaultdict(list)
    for zf in zip_paths:
        unpack_dir = extract_zip(zf, unpack_root, overwrite)
        for tiff in unpack_dir.rglob("*.tif*"):
            pol = classify_polarization(tiff)
            if pol:
                pol_files[pol].append(tiff)

    for pol in pol_files:
        pol_files[pol] = sorted(set(pol_files[pol]))

    if "VV" not in pol_files or not pol_files["VV"]:
        logger.warning("[%s] 未找到 VV 极化文件，跳过", date_str)
        return {"date": date_str, "written": 0, "skipped": 0, "pols": list(pol_files.keys())}

    logger.info(
        "[%s] 极化文件: VV=%d%s",
        date_str,
        len(pol_files.get("VV", [])),
        f", VH={len(pol_files.get('VH', []))}" if "VH" in pol_files else "",
    )

    out_dir = output_root / "patches" / "highres_sar"

    written = 0
    skipped = 0
    for patch in patches:
        res = _process_patch(patch, pol_files, dst_crs)
        if res is None or res.get("status") == "skipped":
            skipped += 1
            continue
        if res.get("status") == "written":
            write_res = _write_patch(res, out_dir, date_str, dst_crs, overwrite)
            if write_res["status"] in ("written", "exists"):
                written += 1
            else:
                skipped += 1

    logger.info("[%s] 完成: written=%d, skipped=%d", date_str, written, skipped)
    return {
        "date": date_str,
        "written": written,
        "skipped": skipped,
        "pols": list(pol_files.keys()),
    }


def process_date_wrapper(task: dict[str, Any]) -> dict[str, Any]:
    """多进程入口。"""
    return process_date(
        date_str=task["date_str"],
        zip_paths=task["zip_paths"],
        patches=task["patches"],
        dst_crs=task["dst_crs"],
        input_root=task["input_root"],
        output_root=task["output_root"],
        overwrite=task["overwrite"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="预处理海淀高分辨率 SAR ORG 产品为对齐 patches",
    )
    parser.add_argument(
        "--input-root",
        default="/data/xuannv_embedding/raw/haidian/highres_sar",
        help="高分辨率 SAR zip 根目录",
    )
    parser.add_argument(
        "--output-root",
        default="/data/xuannv_embedding/processed/haidian",
        help="输出根目录",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="低分辨率预处理 JSON 配置（如 scripts/data/preprocess_haidian.json）",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="调试选项：最多处理的日期数",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="仅处理指定日期，逗号分隔，如 20250821,20251107",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    dst_crs = CRS.from_string(cfg.get("crs", "EPSG:32650"))

    if not input_root.exists():
        logger.error("Input root does not exist: %s", input_root)
        return 1

    patches = generate_patch_grid(cfg, dst_crs, output_root)
    logger.info("生成 %d 个参考 patches", len(patches))

    dates = discover_org_zips(input_root)
    if not dates:
        logger.warning("未找到任何 ORG zip 文件")
        return 0

    if args.dates:
        date_filter = set(args.dates.split(","))
        dates = {k: v for k, v in dates.items() if k in date_filter}
        if not dates:
            logger.warning("没有匹配的日期: %s", args.dates)
            return 0

    if args.max_scenes:
        dates = dict(list(dates.items())[: args.max_scenes])

    tasks = [
        {
            "date_str": date_str,
            "zip_paths": zip_paths,
            "patches": patches,
            "dst_crs": dst_crs,
            "input_root": input_root,
            "output_root": output_root,
            "overwrite": args.overwrite,
        }
        for date_str, zip_paths in sorted(dates.items())
    ]

    total_written = 0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_date_wrapper, tasks):
            total_written += result.get("written", 0)
            total_skipped += result.get("skipped", 0)

    logger.info(
        "全部完成: dates=%d, written=%d, skipped=%d",
        len(tasks),
        total_written,
        total_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
