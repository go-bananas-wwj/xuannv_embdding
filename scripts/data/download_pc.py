"""从 Planetary Computer 下载 S1/S2/Landsat 时序影像并保存为 NetCDF。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

# 降低 vsicurl 并发与增加 HTTP 重试，缓解大规模 stackstac 读取时的 DNS/连接失败。
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "10")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_CONCURRENCY", "4")
os.environ.setdefault(
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS",
    ".tif,.tiff,.TIF,.jp2",
)

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rasterio._err
import rasterio.errors
import stackstac
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds as rio_transform_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/raw")

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

F = TypeVar("F", bound=Callable[..., Any])


def _retry(
    max_retries: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF_SECONDS
) -> Callable[[F], F]:
    """对网络 IO 操作进行基础指数退避重试的装饰器。"""

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        sleep_time = backoff * (2**attempt)
                        logger.warning(
                            "%s 第 %d 次失败，%.1f 秒后重试: %s",
                            func.__name__,
                            attempt + 1,
                            sleep_time,
                            exc,
                        )
                        time.sleep(sleep_time)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def _retry_io(
    max_retries: int = 3, backoff: float = 5.0
) -> Callable[[F], F]:
    """对 rasterio / stackstac 底层 IO 错误进行重试的装饰器。"""

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    RuntimeError,
                    rasterio.errors.RasterioIOError,
                    rasterio._err.CPLE_AppDefinedError,
                ) as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        sleep_time = backoff * (2**attempt)
                        logger.warning(
                            "%s 第 %d 次失败，%.1f 秒后重试: %s",
                            func.__name__,
                            attempt + 1,
                            sleep_time,
                            exc,
                        )
                        time.sleep(sleep_time)
                        # 非最终尝试前清理可能存在的部分/损坏输出文件
                        out_path = kwargs.get("out_path")
                        if out_path is None and len(args) >= 2:
                            out_path = args[1]
                        if isinstance(out_path, Path) and out_path.exists():
                            logger.warning(
                                "删除部分/损坏输出文件: %s",
                                out_path,
                            )
                            out_path.unlink()
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def _parse_time(value: Any) -> pd.Timestamp:
    """将 numpy/xarray 时间值统一转换为 pandas Timestamp。"""
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if hasattr(value, "to_pydatetime"):
        return pd.Timestamp(value.to_pydatetime())
    raise ValueError(f"无法解析时间值: {value} ({type(value)})")


def _monthly_composite_s1(da: xr.DataArray) -> xr.DataArray:
    """将 S1 同一月份内的多次采集合成为单月均值影像。

    S1 RTC 产品以 0 作为 nodata；合成时把 0 与 NaN 统一视为无效，
    沿 time 维度做 nanmean，可提升月度覆盖范围并避免单次采集覆盖不足
    导致整月下載失败。
    """
    if da.sizes["time"] <= 1:
        return da
    valid = np.isfinite(da) & (da != 0)
    masked = da.where(valid)
    comp = masked.mean(dim="time", skipna=True)
    t0 = _parse_time(da.time.values[0])
    month_start = np.datetime64(f"{t0.year:04d}-{t0.month:02d}-01")
    comp = comp.expand_dims("time")
    comp["time"] = [month_start]
    return comp


def _build_target_grid(
    aoi: gpd.GeoDataFrame,
    epsg: int,
    resolution: float,
) -> tuple[CRS, Affine, tuple[int, int], tuple[float, float, float, float]]:
    """根据 AOI 与目标 CRS/分辨率生成规则网格。"""
    aoi_dst = aoi.to_crs(f"EPSG:{epsg}")
    left, bottom, right, top = aoi_dst.total_bounds.tolist()
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))
    dst_transform = rio_transform_from_bounds(
        left, bottom, right, top, width=width, height=height
    )
    dst_crs = CRS.from_epsg(epsg)
    return dst_crs, dst_transform, (height, width), (left, bottom, right, top)


def _reproject_asset_to_grid(
    href: str,
    dst_crs: CRS,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    aoi_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """用 rasterio 将单个 asset 的 AOI 窗口重投影到目标网格，返回 (1, H, W) 数组。"""
    with rasterio.open(href) as src:
        # 先读取 AOI 在源坐标系下的窗口，避免 reproject 拉取整景大文件。
        win = window_from_bounds(*aoi_bounds, transform=src.transform)
        src_array = src.read(1, window=win)
        src_transform = window_transform(win, src.transform)
        src_crs = src.crs
        src_nodata = src.nodata

        dst_array = np.empty((1, dst_shape[0], dst_shape[1]), dtype=np.float32)
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    # 将 nodata/0 统一为 NaN
    invalid = ~np.isfinite(dst_array) | (dst_array == 0)
    dst_array[invalid] = np.nan
    return dst_array


def _stack_s1_manual(
    items: list[Any],
    aoi: gpd.GeoDataFrame,
    epsg: int,
    resolution: float,
    assets: list[str],
) -> xr.Dataset:
    """当 stackstac 读取 S1 RTC 失败时，逐景手工重投影到统一网格。

    Sentinel-1 RTC 在 Planetary Computer 上偶发 WarpedVRT 块级读取失败，
    但直接用 rasterio 读取 AOI 窗口通常正常。本函数绕过 stackstac，
    为每个 item 的每个 asset 做重投影并堆叠成 xarray Dataset。
    """
    dst_crs, dst_transform, (height, width), aoi_bounds = _build_target_grid(
        aoi, epsg, resolution
    )
    x_coords = np.arange(width) * dst_transform.a + dst_transform.c + dst_transform.a / 2
    y_coords = np.arange(height) * dst_transform.e + dst_transform.f + dst_transform.e / 2

    times: list[pd.Timestamp] = []
    stacked: list[np.ndarray] = []
    for item in items:
        times.append(pd.Timestamp(item.datetime))
        item_arrays: list[np.ndarray] = []
        for asset in assets:
            href = planetary_computer.sign(item.assets[asset].href)
            arr = _reproject_asset_to_grid(
                href, dst_crs, dst_transform, (height, width), aoi_bounds
            )
            item_arrays.append(arr[0])
        stacked.append(np.stack(item_arrays, axis=0))

    data = np.stack(stacked, axis=0).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.array([pd.Timestamp(t).tz_localize(None) for t in times], dtype="datetime64[s]"),
            "band": list(assets),
            "y": y_coords,
            "x": x_coords,
        },
        name="stackstac",
    )
    da = da.assign_coords(epsg=epsg)
    return da.to_dataset(name="stackstac")


COLLECTIONS: dict[str, str] = {
    "s2": "sentinel-2-l2a",
    "s1": "sentinel-1-rtc",
    "landsat": "landsat-c2-l2",
}

# 各数据源默认空间分辨率（米），用于 stackstac 重采样
DEFAULT_RESOLUTION: dict[str, float] = {
    "s2": 10.0,
    "s1": 10.0,
    "landsat": 30.0,
}

# 默认读取的 asset 名称，避免 collection 中包含无空间 CRS 的辅助 asset
DEFAULT_ASSETS: dict[str, list[str] | None] = {
    "s2": [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "SCL",
    ],
    "s1": ["vv", "vh"],
    "landsat": [
        "blue",
        "green",
        "red",
        "nir08",
        "swir16",
        "swir22",
        "qa_pixel",
    ],
}


def _validate_date(date_str: str, name: str) -> None:
    """校验日期字符串是否为 YYYY-MM-DD 格式；非法时抛出 ValueError。"""
    from datetime import datetime

    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"{name} 必须是 YYYY-MM-DD 格式，例如 2025-01-01，收到: {date_str}"
        ) from exc


def _extract_epsg(item: Any) -> int | None:
    """从 STAC item 的 projection 扩展字段中提取 EPSG 代码。"""
    # 新版 projection extension 使用 proj:code，例如 "EPSG:32652"
    code = item.properties.get("proj:code")
    if code is not None:
        if isinstance(code, str) and code.upper().startswith("EPSG:"):
            return int(code.split(":", 1)[1])
        if isinstance(code, int):
            return code
    # 旧版字段
    epsg = item.properties.get("proj:epsg")
    if isinstance(epsg, int):
        return epsg
    return None


def _get_aoi_utm_epsg(aoi: gpd.GeoDataFrame) -> int:
    """返回 AOI 中心点所在 UTM 带的 EPSG 代码。"""
    centroid = aoi.to_crs("EPSG:4326").union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _load_aoi(region_file: Path) -> gpd.GeoDataFrame:
    """读取区域范围文件并转换为 WGS84。"""
    aoi = gpd.read_file(region_file)
    if aoi.crs is None:
        logger.warning("区域文件未指定 CRS，按 EPSG:4326 解析")
        aoi.set_crs("EPSG:4326", inplace=True)
    return aoi.to_crs("EPSG:4326")


@_retry()
def _search_items(
    catalog: pystac_client.Client,
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_items: int | None = None,
) -> list[Any]:
    """在 Planetary Computer 中搜索满足时空范围的 items。"""
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        max_items=max_items,
    )
    return list(search.items())


def _build_output_path(output_dir: Path, source: str, start: str, end: str) -> Path:
    """构造输出 NetCDF 文件路径。"""
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return output_dir / f"{source}_{safe_start}_{safe_end}.nc"


def download_source(
    region: str,
    source: str,
    start: str,
    end: str,
    region_file: Path,
    output_root: Path,
    resolution: float | None = None,
    overwrite: bool = False,
    max_items: int | None = None,
    workers: int = 12,
    min_valid_ratio: float = 0.05,
) -> Path | None:
    """下载指定区域、数据源、时间范围的 PC 影像，保存为 NetCDF。

    参数:
        region: 区域名称，仅用于构造输出目录。
        source: 数据源，s2 / s1 / landsat。
        start: 起始日期，ISO-8601 格式。
        end: 结束日期，ISO-8601 格式。
        region_file: 区域范围 GeoJSON/SHP 等路径。
        output_root: 原始数据根目录。
        resolution: 输出分辨率（米），None 则使用默认值。
        overwrite: 是否覆盖已存在的 NetCDF 文件。
        max_items: 最多搜索的 item 数量，None 表示不限制。
        workers: dask 线程数，控制读取与写入并发。
        min_valid_ratio: 每个时间切片有效像素比例阈值，低于则拒绝。

    返回:
        保存的 NetCDF 路径；若未搜索到有效 item 或文件已存在且不覆盖，则返回 None。
    """
    _validate_date(start, "--start")
    _validate_date(end, "--end")

    collection = COLLECTIONS[source]
    output_dir = output_root / region / source
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = _build_output_path(output_dir, source, start, end)
    if out_path.exists() and not overwrite:
        logger.warning(
            "输出文件已存在: %s，如需覆盖请添加 --overwrite",
            out_path,
        )
        return None

    aoi = _load_aoi(region_file)
    bbox = tuple(aoi.total_bounds.tolist())
    logger.info(
        "开始下载: region=%s source=%s collection=%s bbox=%s date=%s/%s",
        region,
        source,
        collection,
        bbox,
        start,
        end,
    )

    catalog = pystac_client.Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )
    items = _search_items(catalog, collection, bbox, start, end, max_items)
    logger.info("搜索完成: 找到 %d 个 items", len(items))

    if not items:
        logger.warning("未找到任何 items，跳过保存")
        return None

    res = resolution if resolution is not None else DEFAULT_RESOLUTION[source]
    logger.info("使用分辨率 %.1f 米进行 stack", res)

    # 使用 AOI 中心点所在 UTM 带作为固定投影，避免跨带数据 CRS 不一致
    epsg = _get_aoi_utm_epsg(aoi)
    logger.info("使用固定 UTM CRS: EPSG:%s", epsg)

    # 使用 stackstac 堆叠为 xarray Dataset；这里用 latlon bounds 裁剪到 AOI
    assets = DEFAULT_ASSETS.get(source)
    logger.info("读取 assets: %s", assets)
    if source == "s1":
        # S1 RTC 在 PC 端偶发 WarpedVRT 块级读取失败，逐景手工重投影更稳定。
        ds = _stack_s1_manual(items, aoi, epsg, res, assets or ["vv", "vh"])
    else:
        ds = _retry()(stackstac.stack)(
            items,
            bounds_latlon=bbox,
            resolution=res,
            epsg=epsg,
            dtype="float32",
            fill_value=np.float32(np.nan),
            rescale=False,
            assets=assets,
            errors_as_nodata=(),
        )

    logger.info("堆叠结果维度: %s", dict(ds.sizes))

    # stackstac.stack 返回 DataArray，统一转换为 Dataset 以便后续处理
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name="stackstac")

    # S1 同一月份通常有多景数据，部分单次采集覆盖不足会导致按时间切片
    # 校验失败；将其合成为月度均值影像，可提升覆盖并统一时间步长。
    if source == "s1" and ds.sizes["time"] > 1:
        data_var = list(ds.data_vars)[0]
        logger.info("S1 合成月度均值影像: %s", data_var)
        comp = _monthly_composite_s1(ds[data_var])
        # 用新的单时间步 DataArray 重建 Dataset，避免旧 time 坐标对齐导致全 NaN。
        ds = comp.to_dataset(name=data_var)

    # 清理无法序列化为 NetCDF 的 stackstac 内部属性
    # stackstac 会在 Dataset / 坐标 / 变量上留下 RasterSpec 等对象，netCDF 写入时会失败
    _allowed_attr_types = (str, int, float, np.generic, list, tuple, np.ndarray, bytes)
    for obj in [ds, *ds.coords.values(), *ds.data_vars.values()]:
        for attr_key in list(obj.attrs.keys()):
            val = obj.attrs[attr_key]
            if attr_key == "spec" or not isinstance(val, _allowed_attr_types):
                obj.attrs.pop(attr_key, None)
            else:
                try:
                    _ = str(val)
                except Exception:
                    obj.attrs.pop(attr_key, None)

    # 丢弃 object 类型坐标（如 proj:bbox / proj:transform 为 set/list，无法序列化为 NetCDF）
    drop_coords = [name for name, coord in ds.coords.items() if coord.dtype == object]
    if drop_coords:
        logger.info("丢弃非序列化坐标: %s", drop_coords)
        ds = ds.drop_vars(drop_coords)

    # 将时间坐标规范化为秒级精度，避免 netCDF3 后端因 int64 纳秒时间编码失败
    if "time" in ds.coords:
        ds["time"] = ds["time"].astype("datetime64[s]")

    _validate_and_write(ds, out_path, workers, min_valid_ratio)
    return out_path


def _validate_coverage(ds: Any, min_valid_ratio: float = 0.05) -> None:
    """校验每个时间切片有效像素比例，低于阈值时抛出 ValueError。"""
    arr = ds[list(ds.data_vars)[0]] if hasattr(ds, "data_vars") else ds
    for t in range(arr.sizes["time"]):
        band0 = arr.isel(time=t, band=0)
        # 有效 = 非 NaN 且非 0（覆盖 NaN fill 与原始 0 nodata）
        values = band0.values
        valid_mask = np.isfinite(values) & (values != 0)
        valid = float(valid_mask.mean())
        if valid < min_valid_ratio:
            raise ValueError(
                f"time {t} coverage {valid:.2%} < {min_valid_ratio:.2%}"
            )


def _write_dataset(
    ds: Any, out_path: Path, num_workers: int = 12
) -> None:
    """将 xarray Dataset 写入 NetCDF。

    先把 dask 数组物化到内存（``ds.load()``），再同步写入 NetCDF。
    这样可以避免 ``to_netcdf`` 流式写入时 dask 图调度长时间不刷盘的问题。
    本环境内存充足（>900 GB 可用），对哈尔滨/海淀这种尺度的 S2 季度/半年数据是安全的。
    """
    logger.info("开始物化并写入 NetCDF: %s (workers=%d)", out_path, num_workers)
    with dask.config.set(scheduler="threads", num_workers=num_workers):
        ds.load()
    ds.to_netcdf(
        out_path,
        encoding={
            "time": {"dtype": "int32", "units": "seconds since 1970-01-01"},
        },
    )
    logger.info("保存成功: %s", out_path)


@_retry_io(max_retries=3, backoff=5.0)
def _validate_and_write(
    ds: Any, out_path: Path, workers: int, min_valid_ratio: float = 0.05
) -> None:
    """校验覆盖度并将 Dataset 写入 NetCDF，IO 错误时支持重试。

    覆盖率不足会抛出 ``ValueError``，不会被 ``@_retry_io`` 重试；
    只有读取/写入阶段的 IO 错误会被重试。
    """
    _validate_coverage(ds, min_valid_ratio=min_valid_ratio)
    _write_dataset(ds, out_path, num_workers=workers)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="从 Planetary Computer 下载 S1/S2/Landsat 影像",
    )
    parser.add_argument(
        "--region",
        required=True,
        choices=["harbin", "haidian"],
        help="目标区域名称",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(COLLECTIONS.keys()),
        help="数据源：s2 / s1 / landsat",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="起始日期，例如 2025-01-01",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="结束日期，例如 2025-01-31",
    )
    parser.add_argument(
        "--region-file",
        required=True,
        type=Path,
        help="区域范围 GeoJSON/SHP 文件路径",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        type=Path,
        help=f"原始数据输出根目录（默认 {DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="输出分辨率（米），默认 s2/s1=10, landsat=30",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 NetCDF 文件",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="限制搜索返回的最大 item 数量（调试用）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="dask 线程并发数（默认 4）",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.05,
        help="每个时间切片最小有效像素比例（默认 0.05）",
    )
    args = parser.parse_args(argv)

    if args.min_valid_ratio < 0.0 or args.min_valid_ratio > 1.0:
        logger.error(
            "--min-valid-ratio 必须在 [0, 1] 范围内，收到: %s",
            args.min_valid_ratio,
        )
        return 1

    try:
        download_source(
            region=args.region,
            source=args.source,
            start=args.start,
            end=args.end,
            region_file=args.region_file,
            output_root=args.output_root,
            resolution=args.resolution,
            overwrite=args.overwrite,
            max_items=args.max_items,
            workers=args.workers,
            min_valid_ratio=args.min_valid_ratio,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("下载过程发生错误: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
