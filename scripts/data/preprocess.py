"""对齐多源 NetCDF 时序数据到统一 10 m UTM 网格并切分为 patches。

核心改进：
- 一次性加载整时相到内存；
- 按 AOI 主网格对齐（整数像素偏移或 warp）；
- 批量整数窗口切片并补 nodata；
- 每源独立有效掩膜；
- 按 (nc_file, time_idx) 多进程并行。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TOLERANCE = 1e-3
MASTER_RES = 10.0

# S2 SCL 有效类别：植被、裸土、水、低云、雪
S2_VALID_SCL = {4, 5, 6, 7, 11}


def _find_data_variable(ds: xr.Dataset) -> str:
    """从 Dataset 中找出具有 (time, band, y, x) 维度的数据变量。"""
    target_dims = {"time", "band", "y", "x"}
    candidates = [name for name, var in ds.data_vars.items() if set(var.dims) == target_dims]
    if not candidates:
        raise ValueError(
            f"NetCDF 中未找到维度为 (time, band, y, x) 的数据变量，"
            f"实际变量: {list(ds.data_vars.keys())}"
        )
    return candidates[0]


def _extract_epsg(ds: xr.Dataset) -> int:
    """从 Dataset 坐标或属性中提取 EPSG 代码。"""
    if "epsg" in ds.coords:
        epsg = int(ds.coords["epsg"].values)
        if epsg > 0:
            return epsg
    if "proj:code" in ds.coords:
        code = str(ds.coords["proj:code"].values)
        if code.upper().startswith("EPSG:"):
            return int(code.split(":", 1)[1])
    for key in ("epsg", "crs"):
        if key in ds.attrs:
            val = ds.attrs[key]
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.upper().startswith("EPSG:"):
                return int(val.split(":", 1)[1])
    raise ValueError(f"无法从 NetCDF 提取 EPSG: {list(ds.coords.keys())}, {dict(ds.attrs)}")


def _build_src_transform(x: np.ndarray, y: np.ndarray) -> tuple[Affine, bool]:
    """根据 xarray 的 x/y 坐标构建 rasterio 仿射变换。

    返回:
        (src_transform, needs_vertical_flip)
    """
    x_vals = np.asarray(x, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)

    x_res = float(np.abs(np.diff(x_vals).mean()))
    y_res = float(np.abs(np.diff(y_vals).mean()))

    left = float(x_vals.min()) - x_res / 2.0
    top = float(y_vals.max()) + y_res / 2.0

    src_transform = from_origin(left, top, x_res, y_res)
    needs_vertical_flip = bool(y_vals[1] > y_vals[0])
    return src_transform, needs_vertical_flip


def load_config(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件。"""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("min_valid_ratio", 0.3)
    config.setdefault("workers", 8)
    config.setdefault("nodata", 0.0)
    config.setdefault("aoi_path", f"configs/regions/{config['region']}.geojson")
    return config


def generate_patch_grid(
    aoi_path_or_file: str | Path,
    patch_size_m: float,
    crs: str | CRS,
) -> tuple[gpd.GeoDataFrame, Affine]:
    """基于 AOI 生成 patch 网格与主变换。

    返回:
        (patch_gdf, master_transform)，其中 master_transform 的 a=10, e=-10。
    """
    aoi = gpd.read_file(aoi_path_or_file)
    crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs
    aoi = aoi.to_crs(crs_obj)

    left, bottom, right, top = aoi.total_bounds

    # 将边界向外吸附到 10 m 网格，确保 transform 严格为 10 m
    left = math.floor(left / MASTER_RES) * MASTER_RES
    bottom = math.floor(bottom / MASTER_RES) * MASTER_RES
    right = math.ceil(right / MASTER_RES) * MASTER_RES
    top = math.ceil(top / MASTER_RES) * MASTER_RES

    width = int(round((right - left) / MASTER_RES))
    height = int(round((top - bottom) / MASTER_RES))

    master_transform = from_bounds(left, bottom, right, top, width=width, height=height)

    raw_patches = make_patch_grid((left, bottom, right, top), patch_size_m)
    n_rows = int(np.ceil((top - bottom) / patch_size_m))

    records: list[dict[str, Any]] = []
    for idx, pbounds in enumerate(raw_patches):
        col = idx // n_rows
        row = idx % n_rows
        patch_id = f"p{col:03d}_r{row:03d}"
        records.append({"patch_id": patch_id, "geometry": shapely.geometry.box(*pbounds)})

    patch_gdf = gpd.GeoDataFrame(records, crs=crs_obj)
    return patch_gdf, master_transform


def _compute_valid_mask(slice_arr: np.ndarray, source: str, nodata: float) -> np.ndarray:
    """计算 uint8 有效像素掩膜。"""
    ref = slice_arr[0]
    valid = np.isfinite(ref) & (ref != nodata)

    if source == "s2":
        scl = slice_arr[-1]
        scl_uint8 = np.nan_to_num(scl, nan=0.0).astype(np.uint8)
        valid = valid & np.isin(scl_uint8, list(S2_VALID_SCL))
    elif source == "landsat":
        qa = slice_arr[-1].astype(np.uint16)
        valid = valid & ((qa & 0b11111) == 0)

    return valid.astype(np.uint8)


def _slice_with_padding(
    arr: np.ndarray,
    row_off: int,
    col_off: int,
    win_h: int,
    win_w: int,
    nodata: float,
) -> np.ndarray:
    """从数组中切出指定窗口，越界部分用 nodata 填充。"""
    src_h, src_w = arr.shape[-2:]
    pad_top = max(0, -row_off)
    pad_left = max(0, -col_off)
    pad_bottom = max(0, row_off + win_h - src_h)
    pad_right = max(0, col_off + win_w - src_w)

    slice_arr = arr[
        :,
        max(0, row_off) : min(row_off + win_h, src_h),
        max(0, col_off) : min(col_off + win_w, src_w),
    ]

    if pad_top or pad_left or pad_bottom or pad_right:
        slice_arr = np.pad(
            slice_arr,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=nodata,
        )
    return slice_arr


def _extract_patch_bounded(
    arr: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    master_transform: Affine,
    master_crs: CRS,
    window: rasterio.windows.Window,
    nodata: float,
) -> np.ndarray:
    """按 master 网格窗口提取一个 patch，不分配完整 AOI 数组。

    对 S2/S1 等已匹配 10 m 网格的数据直接整数像素切片；对 Landsat 等
    分辨率不同的数据，先切出覆盖该 patch 的源区域再重采样到 128x128。
    """
    bands = arr.shape[0]
    patch_h = int(round(window.height))
    patch_w = int(round(window.width))
    patch_transform = rasterio.windows.transform(window, master_transform)

    src_res = abs(src_transform.a)
    src_y_res = abs(src_transform.e)
    same_crs = src_crs == master_crs
    res_match = abs(src_res - MASTER_RES) < TOLERANCE and abs(src_y_res - MASTER_RES) < TOLERANCE
    no_skew = abs(src_transform.b) < TOLERANCE and abs(src_transform.d) < TOLERANCE

    # 计算覆盖 patch 地理范围的源窗口
    src_window = window_from_bounds(
        *rasterio.windows.bounds(window, master_transform),
        transform=src_transform,
    )
    row_off = int(round(src_window.row_off))
    col_off = int(round(src_window.col_off))
    win_h = int(round(src_window.height))
    win_w = int(round(src_window.width))

    slice_arr = _slice_with_padding(arr, row_off, col_off, win_h, win_w, nodata)

    if same_crs and res_match and no_skew and slice_arr.shape[-2:] == (patch_h, patch_w):
        return slice_arr

    dst = np.full((bands, patch_h, patch_w), nodata, dtype=arr.dtype)
    reproject(
        source=slice_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=patch_transform,
        dst_crs=master_crs,
        resampling=Resampling.bilinear,
        dst_nodata=nodata,
    )
    return dst


def process_one_time(args: tuple[Path, int, dict[str, Any]]) -> int:
    """处理单个 NetCDF 的单个时间步，返回写入的 patch 数量。"""
    nc_path, time_idx, config = args

    try:
        source = config["source"]
        nodata = float(config["nodata"])
        min_valid_ratio = float(config["min_valid_ratio"])
        master_transform = Affine(*config["master_transform"])
        master_crs = CRS.from_string(config["crs"])
        patches = config["patches"]

        out_dir = Path(config["output_root"]) / "patches" / source
        out_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
            data_var = _find_data_variable(ds)
            arr = ds[data_var].isel(time=time_idx).values.astype(np.float32)

            y_coords = np.asarray(ds.y.values, dtype=np.float64)
            src_transform, needs_vertical_flip = _build_src_transform(
                np.asarray(ds.x.values, dtype=np.float64), y_coords
            )
            if needs_vertical_flip:
                arr = arr[:, ::-1, :]

            src_crs = CRS.from_epsg(_extract_epsg(ds))
            timestamp = pd.Timestamp(ds["time"].values[time_idx])
            date_str = timestamp.strftime("%Y%m%d")

        base_profile = {
            "driver": "GTiff",
            "height": config["patch_size_px"],
            "width": config["patch_size_px"],
            "crs": master_crs,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
        }

        for patch in patches:
            bounds = patch["bounds"]
            patch_id = patch["patch_id"]
            window = window_from_bounds(*bounds, transform=master_transform)

            slice_arr = _extract_patch_bounded(
                arr=arr,
                src_transform=src_transform,
                src_crs=src_crs,
                master_transform=master_transform,
                master_crs=master_crs,
                window=window,
                nodata=nodata,
            )
            valid_mask = _compute_valid_mask(slice_arr, source, nodata)

            if float(valid_mask.mean()) < min_valid_ratio:
                continue

            out_path = out_dir / f"{source}_{date_str}_{patch_id}.tif"
            mask_path = out_dir / f"{source}_{date_str}_{patch_id}_mask.tif"

            if out_path.exists() and not config.get("overwrite", False):
                continue

            patch_transform = rasterio.windows.transform(window, master_transform)
            profile = {
                **base_profile,
                "count": slice_arr.shape[0],
                "dtype": slice_arr.dtype,
                "transform": patch_transform,
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(slice_arr)
                for idx, name in enumerate(config.get("band_names", []), start=1):
                    dst.set_band_description(idx, name)

            mask_profile = {
                **base_profile,
                "count": 1,
                "dtype": "uint8",
                "nodata": None,
                "transform": patch_transform,
            }
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(valid_mask, 1)

            written += 1

        return written
    except Exception as exc:  # noqa: BLE001
        logger.exception("处理失败: nc=%s time_idx=%s - %s", nc_path, time_idx, exc)
        return 0


def process_file(nc_path: Path, config: dict[str, Any], max_times: int | None = None) -> int:
    """处理单个 NetCDF 文件的所有时间步（或前 max_times 个）。"""
    with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
        n_times = ds.sizes["time"]

    n_times = min(n_times, max_times) if max_times is not None else n_times
    tasks = [(nc_path, t, config) for t in range(n_times)]
    total_written = 0
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        for time_idx, written in enumerate(executor.map(process_one_time, tasks)):
            total_written += written
            if written == 0:
                logger.warning(
                    "%s: time_idx=%d 未写入任何 patch（可能处理异常或全部被过滤）",
                    nc_path.name,
                    time_idx,
                )

    return total_written


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="将 NetCDF 时序数据对齐到统一 10 m 网格并切分为 patches",
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON 配置文件路径")
    parser.add_argument("--source", default=None, help="仅处理指定 source（调试用）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 patch 文件")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理的 NetCDF 文件数")
    parser.add_argument("--max-times", type=int, default=None, help="每个文件最多处理的时间步数")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    region = config["region"]
    raw_root = Path(config["raw_root"])
    output_root = Path(config["output_root"])
    crs = CRS.from_string(config["crs"])
    patch_size_m = float(config["patch_size_m"])
    sources = config["sources"]

    if args.source is not None:
        sources = [args.source]

    aoi_path = Path(config["aoi_path"])
    patch_gdf, master_transform = generate_patch_grid(aoi_path, patch_size_m, crs)

    # 从 snapped bounds 计算主网格形状
    left, bottom, right, top = patch_gdf.total_bounds
    master_shape = (
        int(round((top - bottom) / MASTER_RES)),
        int(round((right - left) / MASTER_RES)),
    )

    config["master_transform"] = tuple(master_transform)
    config["master_shape"] = master_shape
    # 多进程传递：转换为纯 Python 结构
    config["patches"] = [
        {"patch_id": row["patch_id"], "bounds": row.geometry.bounds}
        for _, row in patch_gdf.iterrows()
    ]

    logger.info(
        "开始预处理: region=%s raw_root=%s output_root=%s crs=%s patches=%d shape=%s",
        region,
        raw_root,
        output_root,
        crs,
        len(config["patches"]),
        master_shape,
    )

    total_written = 0
    t0 = time.time()
    for source in sources:
        source_dir = raw_root / source
        if not source_dir.exists():
            logger.warning("source 目录不存在，跳过: %s", source_dir)
            continue

        nc_files = sorted(source_dir.glob("*.nc"))
        if not nc_files:
            logger.warning("未找到 NetCDF 文件: %s", source_dir)
            continue

        if args.max_files is not None:
            nc_files = nc_files[: args.max_files]

        config["source"] = source
        config["overwrite"] = args.overwrite

        with xr.open_dataset(nc_files[0], chunks={"time": 1}) as ds:
            config["band_names"] = [str(b) for b in ds.band.values]

        written = 0
        for nc_path in nc_files:
            logger.info("%s: 处理 %s", source, nc_path.name)
            written += process_file(nc_path, config, args.max_times)

        total_written += written
        logger.info("%s: 写入 %d 个 patches", source, written)

    elapsed = time.time() - t0
    logger.info(
        "预处理完成: 共写入 %d 个 patches, 耗时 %.1f s (%.2f patches/s)",
        total_written,
        elapsed,
        total_written / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
