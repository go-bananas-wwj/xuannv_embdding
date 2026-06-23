#!/usr/bin/env python
"""根据 sample_national_patches.csv 下载 AEF 教师 embedding 与 S2 输入影像。

- AEF: 从 Source Cooperative COG 读取 128x128 窗口，反量化为 float32 后保存为 .pt。
- S2: 通过 Earth Search STAC 按月度合成（中位数），使用 10m 分辨率输出。
  将 S2 拆分为两个模态：
  * s2    : 10m 波段 (red, green, blue, nir)
  * s2_20m: 20m 波段 (rededge1/2/3, nir08, swir16, swir22) 重采样到 10m。
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import stackstac
import torch
from pystac_client import Client
from rasterio.transform import from_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

# Earth Search 公开 STAC catalog
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

S2_ASSETS_10M = ["red", "green", "blue", "nir"]
S2_ASSETS_20M = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]

SOURCES = {
    "s2": S2_ASSETS_10M,
    "s2_20m": S2_ASSETS_20M,
}

DEFAULT_MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
CLOUD_COVER_THRESHOLD = 80.0


def dequantize_aef(data: np.ndarray) -> np.ndarray:
    """将 int8 AEF 量化值反量化为 float32 单位向量表示。"""
    data = data.astype(np.float32)
    return ((data / 127.5) ** 2) * np.sign(data)


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(tensor, tmp)
    tmp.replace(path)


def download_aef(row: pd.Series, output_root: Path) -> bool:
    """下载单个 patch 的 AEF 教师 embedding。"""
    patch_id = row["patch_id"]
    out_path = (
        output_root / "embeddings" / "national" / patch_id / "202512_embedding_map.pt"
    )
    if out_path.exists():
        return True
    try:
        with rasterio.open(row["aef_vsis_path"]) as src:
            window = rasterio.windows.Window(
                int(row["aef_col_off"]),
                int(row["aef_row_off"]),
                int(row["aef_width"]),
                int(row["aef_height"]),
            )
            data = src.read(window=window)  # (64, H, W)
        data = dequantize_aef(data)
        nodata_mask = np.isnan(data) | (np.abs(data) > 1.0)
        if nodata_mask.any():
            logger.warning(
                "AEF patch %s contains %d nodata pixels", patch_id, nodata_mask.sum()
            )
            data[nodata_mask] = 0.0
        tensor = torch.from_numpy(data)
        save_tensor(out_path, tensor)
        return True
    except Exception as exc:
        logger.error("AEF download failed for %s: %s", patch_id, exc)
        return False


def _month_range(month_str: str) -> str:
    """返回 ISO 8601 时间段字符串，覆盖整个月。"""
    year = int(month_str.split("-")[0])
    month = int(month_str.split("-")[1])
    _, last_day = calendar.monthrange(year, month)
    return f"{month_str}-01/{month_str}-{last_day}"


def _save_geotiff(path: Path, array: np.ndarray, bounds: tuple, epsg: int) -> None:
    """保存 (bands, H, W) 数组为 GeoTIFF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    bands, height, width = array.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=array.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(array)


def _prepare_items(items: list) -> list:
    """对 Earth Search STAC item 补充 proj:epsg（需要时）。"""
    out = []
    for it in items:
        d = it.to_dict()
        props = d.setdefault("properties", {})
        if props.get("proj:epsg") is None:
            code = props.get("proj:code")
            if isinstance(code, str) and code.upper().startswith("EPSG:"):
                try:
                    props["proj:epsg"] = int(code.split(":", 1)[1])
                except Exception:
                    pass
            else:
                m = re.search(r"_\d{2}[A-Z]{3}_", it.id)
                if m:
                    mgrs = m.group(0)[1:4]
                    zone = int(mgrs[:2])
                    south = mgrs[2] < "N"
                    props["proj:epsg"] = 32700 + zone if south else 32600 + zone
        out.append(d)
    return out


def _download_source_month(
    row: pd.Series,
    month: str,
    source: str,
    assets: list[str],
    catalog: Client,
    output_root: Path,
) -> bool:
    """下载单个 patch 某月某数据源的中位数合成影像。"""
    patch_id = row["patch_id"]
    epsg = int(row["utm_epsg"])
    bounds_utm = (
        row["bbox_left_utm"],
        row["bbox_bottom_utm"],
        row["bbox_right_utm"],
        row["bbox_top_utm"],
    )
    date_str = month.replace("-", "") + "01"
    out_path = (
        output_root
        / "processed"
        / "national"
        / "patches"
        / source
        / f"{source}_{date_str}_{patch_id}.tif"
    )
    if out_path.exists():
        return True

    try:
        # 用 UTM bbox 的 lat/lon 范围查询 STAC。
        transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
        lons, lats = transformer.transform(
            [bounds_utm[0], bounds_utm[2]], [bounds_utm[1], bounds_utm[3]]
        )
        bounds_latlon = (min(lons), min(lats), max(lons), max(lats))

        search = catalog.search(
            collections=[S2_COLLECTION],
            bbox=bounds_latlon,
            datetime=_month_range(month),
            max_items=50,
            query={"eo:cloud_cover": {"lt": CLOUD_COVER_THRESHOLD}},
        )
        items = list(search.items())
        if not items:
            array = np.zeros(
                (len(assets), int(row["aef_height"]), int(row["aef_width"])),
                dtype=np.float32,
            )
            _save_geotiff(out_path, array, bounds_utm, epsg)
            return True

        item_dicts = _prepare_items(items)
        stack = stackstac.stack(
            item_dicts,
            assets=assets,
            epsg=epsg,
            resolution=10,
            bounds=bounds_utm,
            dtype=np.float64,
            fill_value=0,
            rescale=True,
        )
        if stack.shape[0] == 0 or stack.shape[1] == 0:
            logger.warning(
                "%s %s %s: stackstac returned empty stack (shape=%s), using zeros",
                source,
                patch_id,
                month,
                stack.shape,
            )
            array = np.zeros(
                (len(assets), int(row["aef_height"]), int(row["aef_width"])),
                dtype=np.float32,
            )
            _save_geotiff(out_path, array, bounds_utm, epsg)
            return True

        median = stack.median(dim="time", skipna=True).compute()
        array = median.values.astype(np.float32)  # (bands, H, W)

        expected_h = int(row["aef_height"])
        expected_w = int(row["aef_width"])
        if array.shape[1] != expected_h or array.shape[2] != expected_w:
            logger.warning(
                "%s %s %s shape mismatch %s vs (%d,%d), cropping/padding",
                source,
                patch_id,
                month,
                array.shape[1:],
                expected_h,
                expected_w,
            )
            array = _crop_pad(array, expected_h, expected_w)

        _save_geotiff(out_path, array, bounds_utm, epsg)
        return True
    except Exception as exc:
        logger.exception(
            "%s download failed for %s %s: %s", source, patch_id, month, exc
        )
        return False


def _crop_pad(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """将 (C, H, W) 裁剪/填充到目标尺寸。"""
    c, h, w = array.shape
    out = np.zeros((c, target_h, target_w), dtype=array.dtype)
    crop_h = min(h, target_h)
    crop_w = min(w, target_w)
    off_h = (target_h - crop_h) // 2
    off_w = (target_w - crop_w) // 2
    out[:, off_h : off_h + crop_h, off_w : off_w + crop_w] = array[
        :, :crop_h, :crop_w
    ]
    return out


def process_patch(row: pd.Series, args: argparse.Namespace) -> bool:
    """处理单个 patch：AEF + 各月 s2/s2_20m。"""
    if isinstance(row, dict):
        row = pd.Series(row)
    patch_id = row["patch_id"]
    output_root = Path(args.output_root)

    # 每个 worker 使用独立的 STAC catalog 客户端。
    catalog = Client.open(EARTH_SEARCH_URL)

    ok = True
    if not download_aef(row, output_root):
        ok = False

    for month in args.months:
        for source, assets in SOURCES.items():
            if not _download_source_month(
                row, month, source, assets, catalog, output_root
            ):
                ok = False

    if ok:
        logger.info("Finished patch %s", patch_id)
    else:
        logger.warning("Patch %s had failures", patch_id)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--output-root", default="/data2/xuannv_embedding/national", type=Path
    )
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="仅处理前 N 个 patch（用于 smoke）"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)
    logger.info("Processing %d patches", len(df))

    if args.max_workers <= 1:
        results = [process_patch(row, args) for _, row in df.iterrows()]
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_patch, row, args): i
                for i, row in df.iterrows()
            }
            results = [False] * len(df)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(
                        "Worker failed for patch %s: %s", df.iloc[idx]["patch_id"], exc
                    )

    success = sum(results)
    logger.info("Completed %d / %d patches successfully", success, len(df))


if __name__ == "__main__":
    main()
