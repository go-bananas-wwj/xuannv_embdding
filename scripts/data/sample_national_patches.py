#!/usr/bin/env python
"""全国范围随机采样训练 patch 位置，输出 CSV 供下载脚本使用。

策略：
- 基于 Source Cooperative AEF 年度 index 找到 2025 年中国境内所有 tile；
- 按经纬度网格分层，确保全国覆盖与"有特点"的样本；
- 在每个 tile 内随机采样 patch 中心，保证 128x128 像素窗口不越界；
- 输出包含 UTM bbox、AEF tile 路径、像素窗口等信息的 CSV。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import s3fs

from affine import Affine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

# 中国大致经纬度范围
CHINA_BBOX = (73.5, 18.0, 134.8, 53.6)  # west, south, east, north

# 分层网格（经度 x 纬度），确保全国覆盖
LON_EDGES = [73.0, 85.0, 95.0, 105.0, 115.0, 125.0, 136.0]
LAT_EDGES = [18.0, 28.0, 35.0, 42.0, 54.0]


def load_index(index_url: str) -> pd.DataFrame:
    logger.info("Loading AEF index from %s", index_url)
    fs = s3fs.S3FileSystem(anon=True)
    table = pq.read_table(index_url, filesystem=fs)
    df = table.to_pandas()
    logger.info("Total index rows: %d", len(df))
    return df


def filter_china_tiles(df: pd.DataFrame, year: int, china_bbox: tuple[float, ...]) -> pd.DataFrame:
    west, south, east, north = china_bbox
    mask = (
        (df["year"] == year)
        & (df["wgs84_west"] < east)
        & (df["wgs84_east"] > west)
        & (df["wgs84_south"] < north)
        & (df["wgs84_north"] > south)
    )
    tiles = df[mask].copy()
    # 用 UTM 面积作为采样权重
    tiles["area_m2"] = (tiles["utm_east"] - tiles["utm_west"]).abs() * (
        tiles["utm_north"] - tiles["utm_south"]
    ).abs()
    logger.info("China %d tiles: %d", year, len(tiles))
    return tiles


def assign_bins(tiles: pd.DataFrame, lon_edges: list, lat_edges: list) -> pd.DataFrame:
    """为每个 tile 标记其主要落在哪个经纬度 bin。"""
    lon_centers = (tiles["wgs84_west"] + tiles["wgs84_east"]) / 2.0
    lat_centers = (tiles["wgs84_south"] + tiles["wgs84_north"]) / 2.0
    tiles["lon_bin"] = np.searchsorted(lon_edges, lon_centers) - 1
    tiles["lat_bin"] = np.searchsorted(lat_edges, lat_centers) - 1
    tiles["bin_id"] = tiles["lon_bin"] * (len(lat_edges) - 1) + tiles["lat_bin"]
    return tiles


# AEF COG 尺寸固定为 8192x8192
AEF_TILE_SIZE = 8192


def build_transform(tile: pd.Series) -> Affine:
    """根据 index 中的 UTM bounds 构建 top-down 像素坐标变换（与 VRT 一致）。"""
    width = AEF_TILE_SIZE
    height = AEF_TILE_SIZE
    res_x = (tile["utm_east"] - tile["utm_west"]) / width
    res_y = (tile["utm_south"] - tile["utm_north"]) / height  # 通常为负
    return Affine.translation(tile["utm_west"], tile["utm_north"]) * Affine.scale(res_x, res_y)


def sample_patch_in_tile(tile: pd.Series, patch_pixels: int, margin: int) -> dict | None:
    """在给定 tile 内随机采样一个满足边距要求的 patch。"""
    height = AEF_TILE_SIZE
    width = AEF_TILE_SIZE
    max_row = height - patch_pixels - margin
    max_col = width - patch_pixels - margin
    if max_row <= margin or max_col <= margin:
        return None
    row_off = int(np.random.randint(margin, max_row + 1))
    col_off = int(np.random.randint(margin, max_col + 1))

    transform = build_transform(tile)
    # patch 中心点 UTM 坐标
    center_x, center_y = transform * (col_off + patch_pixels / 2, row_off + patch_pixels / 2)
    half = patch_pixels * 10 / 2
    bbox_utm = {
        "left": center_x - half,
        "bottom": center_y - half,
        "right": center_x + half,
        "top": center_y + half,
    }

    epsg = int(tile["crs"].replace("EPSG:", ""))
    transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(center_x, center_y)

    return {
        "aef_tile_path": tile["path"],
        "aef_vsis_path": tile["path"].replace("s3://", "/vsis3/"),
        "utm_zone": tile["utm_zone"],
        "utm_epsg": epsg,
        "aef_row_off": row_off,
        "aef_col_off": col_off,
        "aef_height": patch_pixels,
        "aef_width": patch_pixels,
        "bbox_left_utm": bbox_utm["left"],
        "bbox_bottom_utm": bbox_utm["bottom"],
        "bbox_right_utm": bbox_utm["right"],
        "bbox_top_utm": bbox_utm["top"],
        "center_lon": lon,
        "center_lat": lat,
    }


def sample_patches(
    tiles: pd.DataFrame,
    num_patches: int,
    patch_pixels: int,
    margin: int,
    min_per_bin: int = 20,
    max_attempts_per_patch: int = 20,
) -> pd.DataFrame:
    tiles = assign_bins(tiles, LON_EDGES, LAT_EDGES)
    bins = tiles["bin_id"].unique()
    logger.info("Number of bins: %d", len(bins))

    # 按 bin 面积比例分配配额
    bin_areas = tiles.groupby("bin_id")["area_m2"].sum()
    total_area = bin_areas.sum()
    quotas = {}
    for bin_id in bins:
        quotas[bin_id] = max(min_per_bin, int(num_patches * bin_areas[bin_id] / total_area))

    # 如果配额总和超过 num_patches，按比例缩放
    total_quota = sum(quotas.values())
    if total_quota > num_patches:
        scale = num_patches / total_quota
        quotas = {k: max(min_per_bin, int(v * scale)) for k, v in quotas.items()}

    logger.info("Quota per bin: %s", json.dumps({int(k): v for k, v in quotas.items()}))

    records: list[dict] = []
    for bin_id, quota in quotas.items():
        bin_tiles = tiles[tiles["bin_id"] == bin_id]
        if bin_tiles.empty:
            continue
        weights = bin_tiles["area_m2"].values.astype(float)
        weights /= weights.sum()
        bin_records: list[dict] = []
        attempts = 0
        while len(bin_records) < quota and attempts < quota * max_attempts_per_patch:
            attempts += 1
            tile = bin_tiles.iloc[np.random.choice(len(bin_tiles), p=weights)]
            patch = sample_patch_in_tile(tile, patch_pixels, margin)
            if patch is not None:
                patch["bin_id"] = int(bin_id)
                bin_records.append(patch)
        records.extend(bin_records)
        logger.info("Bin %d sampled %d patches", bin_id, len(bin_records))

    df = pd.DataFrame(records)
    df = df.reset_index(drop=True)
    df["patch_id"] = [f"national_{i:06d}" for i in range(len(df))]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--num-patches", type=int, default=3000)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--patch-pixels", type=int, default=128)
    parser.add_argument("--margin", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-bin",
        type=int,
        default=20,
        help="每个经纬度 bin 最少采样的 patch 数（地理多样性）。",
    )
    parser.add_argument(
        "--index-url",
        default="us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    df_index = load_index(args.index_url)
    tiles = filter_china_tiles(df_index, args.year, CHINA_BBOX)
    sampled = sample_patches(
        tiles,
        num_patches=args.num_patches,
        patch_pixels=args.patch_pixels,
        margin=args.margin,
        min_per_bin=args.min_per_bin,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)
    logger.info("Saved %d patch locations to %s", len(sampled), args.output_csv)
    logger.info(
        "Lon range: %.3f - %.3f, Lat range: %.3f - %.3f",
        sampled["center_lon"].min(),
        sampled["center_lon"].max(),
        sampled["center_lat"].min(),
        sampled["center_lat"].max(),
    )


if __name__ == "__main__":
    main()
