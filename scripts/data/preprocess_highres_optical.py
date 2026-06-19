#!/usr/bin/env python3
"""预处理高分辨率光学影像，按低分辨率网格切分为 patches。

支持：
- 哈尔滨 0.5 m DOM（CGCS2000 / EPSG:4490），单文件每月；
- 海淀 3 m PlanetScope SR（EPSG:32650），多景每月，附带 UDM2 云影掩膜。

输出与 scripts/data/preprocess.py 生成的 1280 m × 1280 m patch 网格严格对齐，
文件命名格式为 highres_optical_{YYYYMMDD}_{patch_id}.tif 及对应 _mask.tif。

为提升对未分块大文件（哈尔滨 DOM）的处理效率，本脚本按 patch 行一次性读取
并整行重投影，再切出各个 patch，避免对同一源数据行重复读取。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# 抑制 rasterio 对 DOM 元数据警告的重复输出
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
logging.getLogger("rasterio.errors").setLevel(logging.ERROR)

MASTER_RES = 10.0
NODATA = 0.0
MIN_VALID_RATIO = 0.3

# 区域特定参数：目标 UTM CRS、目标分辨率、输出像素尺寸
REGION_CFG: dict[str, dict[str, Any]] = {
    "harbin": {
        "crs": "EPSG:32652",
        "resolution": 0.5,
        "out_size": 2560,  # 1280 m / 0.5 m
    },
    "haidian": {
        "crs": "EPSG:32650",
        "resolution": 3.0,
        "out_size": 427,  # round(1280 m / 3 m)
    },
}


def load_config(path: Path) -> dict[str, Any]:
    """加载低分辨率预处理配置。"""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.setdefault("patch_size_m", 1280.0)
    config.setdefault("aoi_path", f"configs/regions/{config['region']}.geojson")
    return config


def generate_patch_grid(
    aoi_path: str | Path,
    patch_size_m: float,
    crs: str | CRS,
) -> list[dict[str, Any]]:
    """基于 AOI 生成与低分辨率 patch 严格一致的网格。"""
    aoi = gpd.read_file(aoi_path)
    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs
    aoi = aoi.to_crs(crs_obj)

    left, bottom, right, top = aoi.total_bounds

    # 与 preprocess.py 保持一致：外扩吸附到 10 m 主网格
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


def discover_harbin_inputs(input_root: Path) -> dict[str, dict[str, Any]]:
    """发现哈尔滨 DOM 输入：每个月子目录下的 GeoTIFF。"""
    months: dict[str, dict[str, Any]] = {}
    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue
        m = re.search(r"(\d{6})", subdir.name)
        if not m:
            continue
        month_key = m.group(1)
        files = sorted(subdir.rglob("*.tif"))
        if not files:
            continue
        months[month_key] = {"sr": files, "mask": None}
        logger.info("Harbin %s: found %d DOM file(s)", month_key, len(files))
    return months


def discover_haidian_inputs(input_root: Path) -> dict[str, dict[str, Any]]:
    """发现海淀 PlanetScene 输入：按月份分组 SR 与 UDM2 文件。"""
    scene_dir = input_root / "_unzipped" / "PSScene"
    if not scene_dir.exists():
        scene_dir = input_root / "_raw"
    if not scene_dir.exists():
        logger.warning(
            "Haidian high-res optical scene directory not found: %s", scene_dir
        )
        return {}

    sr_files = sorted(scene_dir.glob("*_AnalyticMS_SR_clip.tif"))
    months: dict[str, dict[str, Any]] = {}
    for f in sr_files:
        date_str = f.name[:8]
        if not date_str.isdigit():
            continue
        month_key = date_str[:6]
        udm_file = f.parent / f.name.replace(
            "_AnalyticMS_SR_clip.tif", "_udm2_clip.tif"
        )
        months.setdefault(month_key, {"sr": [], "mask": []})
        months[month_key]["sr"].append(f)
        if udm_file.exists():
            months[month_key]["mask"].append(udm_file)

    for month_key, files in months.items():
        logger.info(
            "Haidian %s: %d SR scene(s), %d UDM2 mask(s)",
            month_key,
            len(files["sr"]),
            len(files["mask"]),
        )
    return months


def build_mosaic(
    input_files: list[Path],
    out_path: Path,
    resampling: Resampling,
    nodata: float = NODATA,
    indexes: list[int] | None = None,
    method: str = "first",
) -> Path:
    """将多个输入文件合并为单个临时 GeoTIFF（输入需已处于同一 CRS）。"""
    if len(input_files) == 1:
        return input_files[0]

    datasets = [rasterio.open(f) for f in input_files]
    mosaic, transform = merge(
        datasets,
        resampling=resampling,
        nodata=nodata,
        method=method,
        indexes=indexes,
    )

    profile = datasets[0].profile
    for ds in datasets:
        ds.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=mosaic.shape[0],
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        dtype=mosaic.dtype,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    return out_path


def _read_source_strip(
    src_path: Path,
    bounds: tuple[float, float, float, float],
    dst_crs: CRS,
    src_nodata: float = NODATA,
) -> tuple[np.ndarray, rasterio.Affine, CRS, list[str]]:
    """读取覆盖指定目标范围的一整条源数据到内存。"""
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_bounds = transform_bounds(dst_crs, src_crs, *bounds)

        # 增加少量缓冲区，避免边缘重采样不足
        x_res = max(abs(float(src.transform.a)), 1e-12)
        y_res = max(abs(float(src.transform.e)), 1e-12)
        buf_x = x_res * 10
        buf_y = y_res * 10
        src_bounds_buf = (
            src_bounds[0] - buf_x,
            src_bounds[1] - buf_y,
            src_bounds[2] + buf_x,
            src_bounds[3] + buf_y,
        )
        win = window_from_bounds(*src_bounds_buf, transform=src.transform)
        src_array = src.read(window=win, boundless=True, fill_value=src_nodata)
        win_transform = src.window_transform(win)
        band_descriptions = list(src.descriptions)
    return src_array, win_transform, src_crs, band_descriptions


def _write_patch(
    patch: dict[str, Any],
    patch_array: np.ndarray,
    clear_mask: np.ndarray,
    patch_transform: rasterio.Affine,
    dst_crs: CRS,
    out_dir: Path,
    date_str: str,
    band_descriptions: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    """写出单个 patch 及其掩膜。"""
    patch_id = patch["patch_id"]
    out_path = out_dir / f"highres_optical_{date_str}_{patch_id}.tif"
    mask_path = out_dir / f"highres_optical_{date_str}_{patch_id}_mask.tif"

    if out_path.exists() and mask_path.exists() and not overwrite:
        return {"patch_id": patch_id, "status": "exists"}

    finite_mask = np.isfinite(patch_array[0])
    for b in range(1, patch_array.shape[0]):
        finite_mask = finite_mask & np.isfinite(patch_array[b])
    nodata_mask = np.all(patch_array != NODATA, axis=0)
    valid = clear_mask & finite_mask & nodata_mask
    valid_ratio = float(valid.mean())

    if valid_ratio < MIN_VALID_RATIO:
        return {
            "patch_id": patch_id,
            "status": "skipped",
            "reason": "low_valid_ratio",
            "valid_ratio": valid_ratio,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": patch_array.shape[-2],
        "width": patch_array.shape[-1],
        "count": patch_array.shape[0],
        "dtype": patch_array.dtype,
        "crs": dst_crs,
        "transform": patch_transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(patch_array)
        for idx, desc in enumerate(band_descriptions, start=1):
            if desc:
                dst.set_band_description(idx, desc)

    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(valid.astype(np.uint8), 1)

    return {
        "patch_id": patch_id,
        "status": "written",
        "valid_ratio": valid_ratio,
    }


def process_row_wrapper(task: dict[str, Any]) -> list[dict[str, Any]]:
    """多进程入口：解包 task 并调用 process_row。"""
    return process_row(
        row_patches=task["row_patches"],
        sr_mosaic=task["sr_mosaic"],
        mask_mosaic=task["mask_mosaic"],
        dst_crs=task["dst_crs"],
        out_size=task["out_size"],
        out_dir=task["out_dir"],
        date_str=task["date_str"],
        overwrite=task["overwrite"],
    )


def _row_intersects_source(
    row_bounds: tuple[float, float, float, float],
    src_path: Path,
    dst_crs: CRS,
) -> bool:
    """判断一行 patch 是否与源数据范围相交。"""
    with rasterio.open(src_path) as src:
        src_bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
    return (
        row_bounds[0] < src_bounds[2]
        and row_bounds[2] > src_bounds[0]
        and row_bounds[1] < src_bounds[3]
        and row_bounds[3] > src_bounds[1]
    )


def process_row(
    row_patches: list[dict[str, Any]],
    sr_mosaic: Path,
    mask_mosaic: Path | None,
    dst_crs: CRS,
    out_size: int,
    out_dir: Path,
    date_str: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """处理一整行 patch：一次性读取源条带，再逐 patch 重投影写出。"""
    row_patches = sorted(row_patches, key=lambda p: p["col"])
    n_cols = len(row_patches)
    if n_cols == 0:
        return []

    left = row_patches[0]["bounds"][0]
    bottom = row_patches[0]["bounds"][1]
    right = row_patches[-1]["bounds"][2]
    top = row_patches[-1]["bounds"][3]

    if not _row_intersects_source((left, bottom, right, top), sr_mosaic, dst_crs):
        return [
            {"patch_id": p["patch_id"], "status": "skipped", "reason": "no_intersection"}
            for p in row_patches
        ]

    # 一次性读取一整行所需源数据，避免对未分块大文件重复全宽读取
    src_strip, src_transform, src_crs, band_descriptions = _read_source_strip(
        sr_mosaic, (left, bottom, right, top), dst_crs, src_nodata=NODATA
    )

    if mask_mosaic and Path(mask_mosaic).exists():
        mask_strip, mask_transform, mask_src_crs, _ = _read_source_strip(
            Path(mask_mosaic), (left, bottom, right, top), dst_crs, src_nodata=0
        )
    else:
        mask_strip = None
        mask_transform = None
        mask_src_crs = None

    results: list[dict[str, Any]] = []
    for patch in row_patches:
        bounds = patch["bounds"]
        dst_transform = from_bounds(*bounds, width=out_size, height=out_size)

        patch_array = np.empty((src_strip.shape[0], out_size, out_size), dtype="float32")
        reproject(
            source=src_strip,
            destination=patch_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=NODATA,
            dst_nodata=NODATA,
            num_threads=4,
        )

        if mask_strip is not None:
            mask_array = np.empty((1, out_size, out_size), dtype="uint8")
            reproject(
                source=mask_strip,
                destination=mask_array,
                src_transform=mask_transform,  # type: ignore[arg-type]
                src_crs=mask_src_crs,  # type: ignore[arg-type]
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
                num_threads=4,
            )
            clear_mask = mask_array[0] == 1
        else:
            clear_mask = np.ones((out_size, out_size), dtype=bool)

        res = _write_patch(
            patch=patch,
            patch_array=patch_array,
            clear_mask=clear_mask,
            patch_transform=dst_transform,
            dst_crs=dst_crs,
            out_dir=out_dir,
            date_str=date_str,
            band_descriptions=band_descriptions,
            overwrite=overwrite,
        )
        results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="预处理高分辨率光学影像为对齐 patches",
    )
    parser.add_argument("--region", required=True, choices=["harbin", "haidian"])
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="低分辨率预处理 JSON 配置（如 scripts/data/preprocess_harbin.json）",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="高分辨率光学原始数据根目录（默认 "
        "/data/xuannv_embedding/raw/{region}/highres_optical）",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="输出根目录（默认 /data/xuannv_embedding/processed/{region}）",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--months",
        default=None,
        help="仅处理指定月份，逗号分隔，如 202512,202601",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="调试选项：最多处理的 patch 数",
    )
    parser.add_argument(
        "--patch-offset",
        type=int,
        default=0,
        help="调试选项：从第几个 patch 开始处理（配合 --max-patches 分块）",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    region = args.region
    region_cfg = REGION_CFG[region]

    dst_crs = CRS.from_string(cfg.get("crs", region_cfg["crs"]))
    if str(dst_crs) != region_cfg["crs"]:
        logger.warning(
            "Config CRS %s differs from expected high-res CRS %s; using config CRS",
            dst_crs,
            region_cfg["crs"],
        )

    input_root = (
        Path(args.input_root)
        if args.input_root
        else Path(f"/data/xuannv_embedding/raw/{region}/highres_optical")
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(f"/data/xuannv_embedding/processed/{region}")
    )

    if not input_root.exists():
        logger.error("Input root does not exist: %s", input_root)
        return 1

    aoi_path = Path(cfg["aoi_path"])
    if not aoi_path.exists():
        logger.error("AOI GeoJSON not found: %s", aoi_path)
        return 1

    patches = generate_patch_grid(aoi_path, float(cfg["patch_size_m"]), dst_crs)
    if args.max_patches or args.patch_offset:
        start = args.patch_offset
        end = start + args.max_patches if args.max_patches else len(patches)
        patches = patches[start:end]
    logger.info("Region %s: generated %d patches", region, len(patches))

    # 发现输入
    if region == "harbin":
        months = discover_harbin_inputs(input_root)
    elif region == "haidian":
        months = discover_haidian_inputs(input_root)
    else:
        months = {}

    if not months:
        logger.warning("No usable high-res optical data found for region %s", region)
        return 0

    month_filter = None
    if args.months:
        month_filter = set(args.months.split(","))
        months = {k: v for k, v in months.items() if k in month_filter}
        if not months:
            logger.warning("No months matched filter: %s", args.months)
            return 0

    out_dir = output_root / "patches" / "highres_optical"

    with tempfile.TemporaryDirectory(prefix=f"highres_optical_{region}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        total_written = 0
        total_skipped = 0

        for month_key in sorted(months.keys()):
            files = months[month_key]
            date_str = f"{month_key}01"
            logger.info("Processing month %s (%s)", month_key, date_str)

            # 构建 SR mosaic
            sr_files = files["sr"]
            if len(sr_files) == 1 and region == "harbin":
                sr_mosaic = sr_files[0]
            else:
                sr_mosaic = tmpdir / f"{region}_{month_key}_sr_mosaic.tif"
                build_mosaic(
                    sr_files,
                    sr_mosaic,
                    resampling=Resampling.bilinear,
                )

            # 构建掩膜 mosaic（如可用）
            mask_mosaic: Path | None = None
            mask_files = files.get("mask") or []
            if mask_files:
                if len(mask_files) == 1 and region == "harbin":
                    mask_mosaic = mask_files[0]
                else:
                    mask_mosaic = tmpdir / f"{region}_{month_key}_mask_mosaic.tif"
                    build_mosaic(
                        mask_files,
                        mask_mosaic,
                        resampling=Resampling.nearest,
                        indexes=[1],
                        method="max",
                    )

            # 按行分组
            rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for p in patches:
                rows[p["row"]].append(p)

            written = 0
            skipped = 0
            row_tasks = [
                {
                    "row_patches": rows[row_idx],
                    "sr_mosaic": sr_mosaic,
                    "mask_mosaic": mask_mosaic,
                    "dst_crs": dst_crs,
                    "out_size": region_cfg["out_size"],
                    "out_dir": out_dir,
                    "date_str": date_str,
                    "overwrite": args.overwrite,
                }
                for row_idx in sorted(rows.keys())
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for results in executor.map(process_row_wrapper, row_tasks):
                    for res in results:
                        if res.get("status") == "written":
                            written += 1
                        elif res.get("status") == "skipped":
                            skipped += 1

            logger.info(
                "Month %s: written=%d, skipped=%d",
                month_key,
                written,
                skipped,
            )
            total_written += written
            total_skipped += skipped

    logger.info(
        "Done: region=%s total_written=%d total_skipped=%d",
        region,
        total_written,
        total_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
