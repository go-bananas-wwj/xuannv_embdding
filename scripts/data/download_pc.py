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
import planetary_computer
import pystac_client
import rasterio
import rasterio._err
import rasterio.errors
import stackstac

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

    # 清理无法序列化为 NetCDF 的 stackstac 内部属性
    ds.attrs.pop("spec", None)
    for attr_key in list(ds.attrs.keys()):
        try:
            _ = str(ds.attrs[attr_key])
        except Exception:
            ds.attrs.pop(attr_key, None)

    # 清理坐标上的非序列化属性
    for coord in ds.coords.values():
        for attr_key in list(coord.attrs.keys()):
            val = coord.attrs[attr_key]
            if isinstance(val, (list, tuple, dict, set)) or not isinstance(
                val, (str, int, float, np.generic)
            ):
                coord.attrs.pop(attr_key, None)

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
    """校验每个时间切片有效像素比例，低于阈值时抛出 RuntimeError。"""
    arr = ds[list(ds.data_vars)[0]] if hasattr(ds, "data_vars") else ds
    for t in range(arr.sizes["time"]):
        band0 = arr.isel(time=t, band=0)
        # 有效 = 非 NaN 且非 0（覆盖 NaN fill 与原始 0 nodata）
        values = band0.values
        valid_mask = np.isfinite(values) & (values != 0)
        valid = float(valid_mask.mean())
        if valid < min_valid_ratio:
            raise RuntimeError(
                f"time {t} coverage {valid:.2%} < {min_valid_ratio:.2%}"
            )


@_retry_io(max_retries=3, backoff=5.0)
def _write_dataset(
    ds: Any, *, out_path: Path, num_workers: int = 12
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


def _validate_and_write(
    ds: Any, out_path: Path, workers: int, min_valid_ratio: float = 0.05
) -> None:
    """校验覆盖度并将 Dataset 写入 NetCDF。

    覆盖度校验在 IO 重试路径之外执行；只有写入操作会被 ``@_retry_io`` 重试。
    """
    _validate_coverage(ds, min_valid_ratio=min_valid_ratio)
    _write_dataset(ds, out_path=out_path, num_workers=workers)


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
