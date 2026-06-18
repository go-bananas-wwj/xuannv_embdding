"""从 Planetary Computer 下载 ESRI 10m Annual Land Cover 2023 作为分类目标。

集合使用 Planetary Computer 上的 `io-lulc-annual-v02`（ESRI 10m Annual Land Use Land Cover V2）。
脚本读取区域 GeoJSON 的 WGS84 bbox，检索与该 bbox 相交的 2023 年 tile(s)，
将每个 tile 的 `data` asset 下载到本地缓存后使用 stackstac 读取并 mosaic，
最终输出为单张 Cloud-Optimized GeoTIFF（uint8，nodata=0）。

说明：ESRI LULC tile 为较大的条带式 GeoTIFF，直接通过 HTTP 范围请求流式读取
容易触发 `_TIFFPartialReadStripArray` 错误，因此脚本先将 tile 下载到本地缓存
再进行后续处理。若 stackstac 因 item 数量、内存或其他原因失败，则回退到
rasterio.warp.reproject 手动 mosaic。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import stackstac
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "io-lulc-annual-v02"
TARGET_YEAR = 2023
TARGET_RESOLUTION = 10.0

DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/raw")
DEFAULT_LOG_ROOT = Path("/data/xuannv_embedding")


def _setup_file_logger(log_path: Path) -> logging.Handler:
    """将日志同时写入指定文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return fh


def _load_aoi(region_file: Path) -> gpd.GeoDataFrame:
    """读取区域范围文件并转换为 WGS84。"""
    aoi = gpd.read_file(region_file)
    if aoi.crs is None:
        logger.warning("区域文件未指定 CRS，按 EPSG:4326 解析")
        aoi.set_crs("EPSG:4326", inplace=True)
    return aoi.to_crs("EPSG:4326")


def _search_items(
    catalog: pystac_client.Client,
    bbox: tuple[float, float, float, float],
    max_items: int | None = None,
) -> list[Any]:
    """搜索 ESRI LULC 2023 与 bbox 相交的 items。"""
    start = f"{TARGET_YEAR}-01-01"
    end = f"{TARGET_YEAR + 1}-01-01"
    search = catalog.search(
        collections=[COLLECTION_ID],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        max_items=max_items,
    )
    return list(search.items())


def _select_target_epsg(items: list[Any], fallback: int = 4326) -> int:
    """从首个 item 的 proj:code 推断目标投影，优先使用该 tile 的 UTM CRS。"""
    if not items:
        return fallback
    code = items[0].properties.get("proj:code")
    if isinstance(code, str) and code.upper().startswith("EPSG:"):
        return int(code.split(":", 1)[1])
    epsg = items[0].properties.get("proj:epsg")
    if isinstance(epsg, int):
        return epsg
    return fallback


def _download_tile(item: Any, cache_dir: Path) -> Path:
    """将 item 的 `data` asset 下载到本地缓存并返回路径。

    使用 wget -c 支持断点续传，避免网络慢或超时时需要从头下载。
    下载过程中先写入 .tmp 文件，完成后重命名为最终缓存文件。
    """
    asset = item.assets["data"]
    href = asset.href
    cache_path = cache_dir / f"{item.id}.tif"
    if cache_path.exists():
        logger.info("使用已缓存 tile: %s", cache_path)
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f"{item.id}.tif.tmp"
    logger.info("下载 tile %s -> %s", item.id, cache_path)
    start_time = time.time()
    try:
        subprocess.run(
            ["wget", "-c", "-O", str(tmp_path), href],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"下载 tile {item.id} 失败: {exc.stdout}") from exc
    tmp_path.replace(cache_path)
    elapsed = time.time() - start_time
    logger.info("下载完成: %s (%.1f MB, %.1f s)", cache_path, cache_path.stat().st_size / 1e6, elapsed)
    return cache_path


def _patch_item_asset_href(item: Any, local_path: Path) -> Any:
    """将 item 的 data asset href 替换为本地路径，使 stackstac 读取本地文件。"""
    item.assets["data"].href = str(local_path)
    return item


def _stackstac_read(
    items: list[Any],
    bbox: tuple[float, float, float, float],
    epsg: int,
) -> tuple[np.ndarray, rasterio.Affine, CRS] | None:
    """使用 stackstac.stack 读取本地 tile 并 mosaic。

    stackstac 当前版本对整数 fill_value 支持有限，因此先以 float64 读取
    （NaN 作为 nodata），再在内存中转为 uint8 并以 0 作为 nodata。
    """
    logger.info("使用 stackstac 读取 %d 个本地 tile(s)，目标 EPSG:%s", len(items), epsg)
    try:
        ds = stackstac.stack(
            items,
            bounds_latlon=bbox,
            resolution=TARGET_RESOLUTION,
            epsg=epsg,
            dtype="float64",
            assets=["data"],
            rescale=False,
        )
    except Exception as exc:
        logger.warning("stackstac.stack 失败: %s", exc)
        return None

    logger.info("stackstac 维度: %s", dict(ds.sizes))
    arr = ds.compute().values  # (time, band, y, x)
    # 去除 band 维度
    arr = np.squeeze(arr, axis=1)  # (time, y, x)

    # mosaic：取第一个非 NaN 值
    mosaic = arr[0].copy()
    for t in range(1, arr.shape[0]):
        mask = np.isnan(mosaic)
        mosaic[mask] = arr[t][mask]

    # NaN -> 0 (nodata)
    mosaic = np.nan_to_num(mosaic, nan=0.0).astype(np.uint8)

    # 构建地理变换
    y_coords = ds["y"].values
    x_coords = ds["x"].values
    transform = from_bounds(
        float(x_coords.min()) - TARGET_RESOLUTION / 2,
        float(y_coords.min()) - TARGET_RESOLUTION / 2,
        float(x_coords.max()) + TARGET_RESOLUTION / 2,
        float(y_coords.max()) + TARGET_RESOLUTION / 2,
        mosaic.shape[1],
        mosaic.shape[0],
    )
    return mosaic, transform, CRS.from_epsg(epsg)


def _rasterio_fallback(
    local_paths: list[Path],
    bbox: tuple[float, float, float, float],
    epsg: int,
) -> tuple[np.ndarray, rasterio.Affine, CRS]:
    """stackstac 失败时的 rasterio 回退方案。

    根据目标 bbox 和 CRS 计算输出栅格尺寸，然后依次将每个本地 tile
    使用最近邻重采样到输出数组中，实现 mosaic。
    """
    logger.info("使用 rasterio.warp.reproject 回退读取 %d 个本地 tile(s)", len(local_paths))
    target_crs = CRS.from_epsg(epsg)

    # 计算输出地理变换与尺寸
    transform, width, height = calculate_default_transform(
        CRS.from_epsg(4326),
        target_crs,
        width=None,
        height=None,
        left=bbox[0],
        bottom=bbox[1],
        right=bbox[2],
        top=bbox[3],
        resolution=TARGET_RESOLUTION,
    )
    logger.info("输出栅格: width=%d height=%d transform=%s", width, height, transform)

    mosaic = np.zeros((height, width), dtype=np.uint8)
    for path in local_paths:
        with rasterio.open(path) as src:
            logger.info("重投影 %s: shape=%s crs=%s", path.name, src.shape, src.crs)
            reproject(
                source=rasterio.band(src, 1),
                destination=mosaic,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=0,
                src_nodata=0,
                resampling=Resampling.nearest,
            )

    return mosaic, transform, target_crs


def download_esri_lulc(
    region: str,
    region_file: Path,
    output_root: Path,
    log_root: Path,
    max_items: int | None = None,
) -> Path:
    """下载指定区域的 ESRI LULC 2023 并保存为 COG GeoTIFF。

    返回输出文件路径。
    """
    output_dir = output_root / region / "esri_lulc_2023"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"esri_lulc_2023_{region}.tif"
    cache_dir = output_dir / ".tiles"

    log_path = log_root / f"download_esri_lulc_{region}.log"
    _setup_file_logger(log_path)
    logger.info("===== 开始下载 ESRI LULC 2023: region=%s =====", region)
    logger.info("日志写入: %s", log_path)

    aoi = _load_aoi(region_file)
    bbox = tuple(aoi.total_bounds.tolist())
    logger.info("区域 WGS84 bbox: %s", bbox)

    catalog = pystac_client.Client.open(
        CATALOG_URL,
        modifier=planetary_computer.sign_inplace,
    )
    items = _search_items(catalog, bbox, max_items=max_items)
    # 按 item id 中的年份后缀过滤，避免搜索边界导致包含上一年 tile
    items = [item for item in items if item.id.endswith(f"-{TARGET_YEAR}")]
    logger.info("搜索完成: 找到 %d 个 %d 年 items", len(items), TARGET_YEAR)
    if not items:
        raise RuntimeError(f"未找到 {region} 区域 {TARGET_YEAR} 年的 ESRI LULC 数据")

    epsg = _select_target_epsg(items)
    logger.info("目标 CRS: EPSG:%s", epsg)

    local_paths = [_download_tile(item, cache_dir) for item in items]
    # 复制 items 并修改 asset href 指向本地缓存，供 stackstac 使用
    patched_items = [_patch_item_asset_href(item.clone(), path) for item, path in zip(items, local_paths)]

    result = _stackstac_read(patched_items, bbox, epsg)
    if result is None:
        logger.info("切换到 rasterio.warp.reproject 回退方案")
        result = _rasterio_fallback(local_paths, bbox, epsg)
    mosaic, transform, crs = result

    height, width = mosaic.shape
    logger.info("Mosaic 尺寸: %d x %d, CRS: %s", width, height, crs)

    # 写入 Cloud-Optimized GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "lzw",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    logger.info("写入 COG: %s", out_path)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic, 1)
        # 构建 overview 以符合 COG 规范
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
        dst.update_tags(ns="gdal_metadata", TILED="YES")

    logger.info("保存成功: %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="下载 ESRI 10m Annual Land Cover 2023 作为分类目标",
    )
    parser.add_argument(
        "--region",
        required=True,
        choices=["harbin", "haidian"],
        help="目标区域名称",
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
        "--log-root",
        default=DEFAULT_LOG_ROOT,
        type=Path,
        help=f"日志输出根目录（默认 {DEFAULT_LOG_ROOT}）",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="限制搜索返回的最大 item 数量（调试用）",
    )
    args = parser.parse_args(argv)

    try:
        download_esri_lulc(
            region=args.region,
            region_file=args.region_file,
            output_root=args.output_root,
            log_root=args.log_root,
            max_items=args.max_items,
        )
    except Exception as exc:
        logger.exception("下载过程发生错误: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
