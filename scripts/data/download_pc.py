"""从 Planetary Computer 下载 S1/S2/Landsat 时序影像并保存为 NetCDF。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import stackstac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
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
        "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B09", "B11", "B12", "SCL",
    ],
    "s1": ["vv", "vh"],
    "landsat": [
        "blue", "green", "red", "nir08", "swir16", "swir22",
    ],
}


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


def _load_aoi(region_file: Path) -> gpd.GeoDataFrame:
    """读取区域范围文件并转换为 WGS84。"""
    aoi = gpd.read_file(region_file)
    if aoi.crs is None:
        logger.warning("区域文件未指定 CRS，按 EPSG:4326 解析")
        aoi.set_crs("EPSG:4326", inplace=True)
    return aoi.to_crs("EPSG:4326")


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

    返回:
        保存的 NetCDF 路径；若未搜索到有效 item 则返回 None。
    """
    collection = COLLECTIONS[source]
    output_dir = output_root / region / source
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = _build_output_path(output_dir, source, start, end)
    if out_path.exists() and not overwrite:
        logger.error(
            "输出文件已存在: %s，如需覆盖请添加 --overwrite",
            out_path,
        )
        raise FileExistsError(f"{out_path} already exists")

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

    # 从首个 item 获取 EPSG 代码，避免某些 asset 缺少 CRS 导致 stack 失败
    epsg = _extract_epsg(items[0])
    if epsg is not None:
        logger.info("使用参考 CRS: EPSG:%s", epsg)

    # 使用 stackstac 堆叠为 xarray Dataset；这里用 latlon bounds 裁剪到 AOI
    assets = DEFAULT_ASSETS.get(source)
    logger.info("读取 assets: %s", assets)
    ds = stackstac.stack(
        items,
        bounds_latlon=bbox,
        resolution=res,
        epsg=epsg,
        dtype="float32",
        fill_value=np.float32(0.0),
        rescale=False,
        assets=assets,
    )

    logger.info("堆叠结果维度: %s", dict(ds.sizes))

    # 清理无法序列化为 NetCDF 的 stackstac 内部属性
    ds.attrs.pop("spec", None)

    logger.info("开始写入 NetCDF: %s", out_path)
    ds.to_netcdf(out_path)
    logger.info("保存成功: %s", out_path)
    return out_path


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
        default="/data/xuannv_embedding/raw",
        help="原始数据输出根目录（默认 /data/xuannv_embedding/raw）",
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
    args = parser.parse_args(argv)

    try:
        download_source(
            region=args.region,
            source=args.source,
            start=args.start,
            end=args.end,
            region_file=args.region_file,
            output_root=Path(args.output_root),
            resolution=args.resolution,
            overwrite=args.overwrite,
            max_items=args.max_items,
        )
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("下载过程发生错误: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
