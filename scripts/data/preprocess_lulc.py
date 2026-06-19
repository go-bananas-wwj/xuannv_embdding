"""将 ESRI 2023 LULC 单张大图切分为与低分辨率 patch 网格对齐的分类标签。

读取区域 GeoJSON 与已下载的 ESRI LULC COG，按照与 ``preprocess.py`` 相同的
patch 网格生成 ``{source}_{date}_{patch_id}.tif``（默认 source=worldcover）。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from tqdm import tqdm

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SOURCE_NAME = "worldcover"
DEFAULT_DATE_STR = "20230101"
DEFAULT_FILL_NODATA = 0
MASTER_RES = 10.0

# ESRI 2023 原始值 → 连续训练索引（0 为 ignore_index，10 Clouds 也 ignore）
ESRI2023_REMAP = {
    0: 0,   # No Data / ignore
    1: 1,   # Water
    2: 2,   # Trees
    4: 3,   # Flooded Vegetation
    5: 4,   # Crops
    7: 5,   # Built Area
    8: 6,   # Bare Ground
    9: 7,   # Snow/Ice
    10: 0,  # Clouds → ignore
    11: 8,  # Rangeland
}


def remap_esri2023(arr: np.ndarray) -> np.ndarray:
    """使用向量化查找表将 ESRI 2023 LULC 原始值重映射到训练索引。"""
    src = np.asarray(arr)
    old_keys = np.array(sorted(ESRI2023_REMAP.keys()), dtype=np.uint8)
    new_vals = np.array([ESRI2023_REMAP[k] for k in old_keys], dtype=np.uint8)

    flat = src.ravel()
    idx = np.searchsorted(old_keys, flat)
    # 对不在映射表中的异常值统一置为 0（ignore）
    valid = (idx < len(old_keys)) & (old_keys[idx] == flat)
    remapped = np.zeros_like(flat, dtype=np.uint8)
    remapped[valid] = new_vals[idx[valid]]
    return remapped.reshape(src.shape)


def load_config(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aoi_bounds_in_crs(region_file: Path, dst_crs: CRS) -> tuple[float, float, float, float]:
    """将区域 GeoJSON 转换到目标 CRS 后返回总 bounds。"""
    aoi = gpd.read_file(region_file)
    if aoi.crs is None:
        aoi.set_crs("EPSG:4326", inplace=True)
    aoi_dst = aoi.to_crs(dst_crs)
    return tuple(aoi_dst.total_bounds)  # type: ignore[return-value]


def _read_patch(
    src: rasterio.DatasetReader,
    patch_bounds: tuple[float, float, float, float],
    dst_shape: tuple[int, int],
    dst_crs: CRS,
) -> np.ndarray:
    """从源栅格中读取指定 patch 范围的数据，必要时重投影/重采样。

    返回形状为 ``(dst_shape[0], dst_shape[1])`` 的 uint8 数组。
    """
    left, bottom, right, top = patch_bounds

    # 始终通过 warp.reproject 读取并严格对齐到目标 patch 边界，避免
    # 当源数据 CRS 与目标相同时窗口读取导致的亚像素偏移。
    dst_transform = from_bounds(left, bottom, right, top, dst_shape[1], dst_shape[0])
    dst_array = np.empty(dst_shape, dtype=np.uint8)
    rasterio.warp.reproject(
        source=rasterio.band(src, 1),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=src.nodata if src.nodata is not None else DEFAULT_FILL_NODATA,
        dst_nodata=DEFAULT_FILL_NODATA,
    )
    return dst_array


def preprocess_lulc(
    region: str,
    region_file: Path,
    input_tiff: Path,
    output_root: Path,
    dst_crs: CRS,
    patch_size_m: float,
    patch_size_px: int,
    source_name: str = DEFAULT_SOURCE_NAME,
    date_str: str = DEFAULT_DATE_STR,
    overwrite: bool = False,
) -> int:
    """切分 LULC 标签图并返回写入的 patch 数量。"""
    output_dir = output_root / "labels" / source_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = _aoi_bounds_in_crs(region_file, dst_crs)

    # 将 AOI 边界向外吸附到 10 m 网格，确保与 preprocess.py 生成的 S2/S1/Landsat
    # patch 网格严格一致。
    left, bottom, right, top = bounds
    left = math.floor(left / MASTER_RES) * MASTER_RES
    bottom = math.floor(bottom / MASTER_RES) * MASTER_RES
    right = math.ceil(right / MASTER_RES) * MASTER_RES
    top = math.ceil(top / MASTER_RES) * MASTER_RES
    bounds = (left, bottom, right, top)

    patches = make_patch_grid(bounds, patch_size_m)
    logger.info("%s: 生成 %d 个 patches", region, len(patches))

    n_rows = int(np.ceil((bounds[3] - bounds[1]) / patch_size_m))

    with rasterio.open(input_tiff) as src:
        written = 0
        for idx, pbounds in enumerate(tqdm(patches, desc=f"{region} LULC")):
            col = idx // n_rows
            row = idx % n_rows
            patch_id = f"p{col:03d}_r{row:03d}"
            out_path = output_dir / f"{source_name}_{date_str}_{patch_id}.tif"
            if out_path.exists() and not overwrite:
                continue

            array = _read_patch(src, pbounds, (patch_size_px, patch_size_px), dst_crs)
            array = remap_esri2023(array)
            if np.all(array == DEFAULT_FILL_NODATA):
                continue

            transform = from_bounds(
                pbounds[0], pbounds[1], pbounds[2], pbounds[3],
                patch_size_px, patch_size_px,
            )
            with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=patch_size_px,
                width=patch_size_px,
                count=1,
                dtype=array.dtype,
                crs=dst_crs,
                transform=transform,
                nodata=DEFAULT_FILL_NODATA,
            ) as dst:
                dst.write(array, 1)
            written += 1

    logger.info("%s: 写入 %d 个非空 LULC patches", region, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切分 ESRI 2023 LULC 标签到 patch 网格")
    parser.add_argument("--config", required=True, type=Path, help="JSON 配置文件路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 patch")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    region = cfg["region"]
    region_file = Path(cfg["region_file"])
    input_tiff = Path(cfg["input_tiff"])
    output_root = Path(cfg["output_root"])
    dst_crs = CRS.from_string(cfg["crs"])
    patch_size_m = float(cfg["patch_size_m"])
    patch_size_px = int(cfg["patch_size_px"])
    source_name = cfg.get("source_name", DEFAULT_SOURCE_NAME)
    date_str = cfg.get("date_str", DEFAULT_DATE_STR)

    preprocess_lulc(
        region=region,
        region_file=region_file,
        input_tiff=input_tiff,
        output_root=output_root,
        dst_crs=dst_crs,
        patch_size_m=patch_size_m,
        patch_size_px=patch_size_px,
        source_name=source_name,
        date_str=date_str,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
