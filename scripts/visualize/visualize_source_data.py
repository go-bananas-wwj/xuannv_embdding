"""对各数据源的原始/采样数据进行快速可视化。

输出目录结构：
    data/visualize/source_overview/{region}/{source}.png

用法：
    python -m scripts.visualize.visualize_source_data \
        --region haidian --source s2 --output-root data/visualize
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Esri LULC 颜色（仅用于概览）
ESRI_LULC_COLORS = {
    0: (0, 0, 0),
    1: (65, 155, 223),   # 水
    2: (129, 199, 132),  # 树
    3: (156, 156, 156),  # 被淹没的植被？
    4: (170, 114, 68),   # 作物
    5: (239, 211, 130),  # 建筑
    6: (226, 129, 141),  # 裸地
    7: (233, 236, 241),  # 雪/冰
    8: (127, 82, 179),   # 云
    9: (209, 209, 209),  # 牧场/草地
    10: (81, 90, 90),    # 灌木
    11: (232, 167, 99),  # 落叶林
    12: (138, 222, 160), # 草地
    13: (44, 117, 168),  # 湿地
    14: (220, 95, 95),   # 农田
}


def _esri_cmap(classes: set[int]) -> ListedColormap:
    max_class = max(classes | {0})
    colors = [(0, 0, 0, 1)] * (max_class + 1)
    for c in range(max_class + 1):
        rgb = ESRI_LULC_COLORS.get(c, (128, 128, 128))
        colors[c] = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)
    return ListedColormap(colors)


def _normalize_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    percentile: tuple[float, float] = (2, 98),
) -> np.ndarray:
    stack = np.stack([r, g, b], axis=-1).astype(np.float32)
    for i in range(3):
        band = stack[..., i]
        valid = np.isfinite(band) & (band > 0)
        if valid.any():
            vmin, vmax = np.percentile(band[valid], percentile)
            band = np.clip((band - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        else:
            band = np.zeros_like(band)
        stack[..., i] = band
    return np.nan_to_num(stack, nan=0.0)


def _find_band(descriptions: list[str], candidates: list[str]) -> int:
    """根据候选名称列表返回 1-based 波段索引。"""
    desc = [str(d).upper() if d else "" for d in descriptions]
    for cand in candidates:
        if cand.upper() in desc:
            return desc.index(cand.upper()) + 1
    return 1


def _read_s2_rgb(ds: xr.Dataset, time_idx: int) -> np.ndarray:
    var = _find_data_var(ds)
    data = ds[var].isel(time=time_idx).values.astype(np.float32)
    desc = [str(b) for b in ds.band.values]
    r = data[_find_band(desc, ["B04"]) - 1]
    g = data[_find_band(desc, ["B03"]) - 1]
    b = data[_find_band(desc, ["B02"]) - 1]
    return _normalize_rgb(r, g, b)


def _read_landsat_rgb(ds: xr.Dataset, time_idx: int) -> np.ndarray:
    var = _find_data_var(ds)
    data = ds[var].isel(time=time_idx).values.astype(np.float32)
    # Landsat DN -> 反射率
    valid = data != 0
    data[valid] = data[valid] * 0.0000275 - 0.2
    desc = [str(b) for b in ds.band.values]
    # 真彩色：红(B4)/绿(B3)/蓝(B2)
    r = data[_find_band(desc, ["B04", "B4"]) - 1]
    g = data[_find_band(desc, ["B03", "B3"]) - 1]
    b = data[_find_band(desc, ["B02", "B2"]) - 1]
    return _normalize_rgb(r, g, b, percentile=(2, 98))


def _read_s1_vv(ds: xr.Dataset, time_idx: int) -> np.ndarray:
    var = _find_data_var(ds)
    data = ds[var].isel(time=time_idx).values.astype(np.float32)
    desc = [str(b) for b in ds.band.values]
    vv = data[_find_band(desc, ["vv", "VV"]) - 1]
    vv = np.where(vv > 0, vv, np.nan)
    vv_db = 10.0 * np.log10(vv)
    vmin, vmax = np.nanpercentile(vv_db, (2, 98))
    return np.clip((vv_db - vmin) / max(vmax - vmin, 1e-6), 0, 1)


def _find_data_var(ds: xr.Dataset) -> str:
    target_dims = {"time", "band", "y", "x"}
    candidates = [n for n, v in ds.data_vars.items() if set(v.dims) == target_dims]
    if not candidates:
        raise ValueError(f"未找到目标变量: {list(ds.data_vars.keys())}")
    return candidates[0]


def _visualize_netcdf(source: str, nc_path: Path, out_path: Path) -> None:
    with xr.open_dataset(nc_path, chunks={"time": 1}) as ds:
        n_times = ds.sizes["time"]
        # 选中间时相，避免首尾可能缺失
        time_idx = n_times // 2
        date_str = pd.to_datetime(ds.time.values[time_idx]).strftime("%Y-%m-%d")
        if source == "s2":
            arr = _read_s2_rgb(ds, time_idx)
            title = f"S2 RGB {date_str}"
        elif source == "s1":
            arr = _read_s1_vv(ds, time_idx)
            title = f"S1 VV log {date_str}"
        elif source == "landsat":
            arr = _read_landsat_rgb(ds, time_idx)
            title = f"Landsat RGB {date_str}"
        else:
            raise ValueError(f"不支持的 source: {source}")

    fig, ax = plt.subplots(figsize=(8, 8))
    if source == "s1":
        ax.imshow(arr, cmap="gray")
    else:
        ax.imshow(arr)
    ax.set_title(title)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def _visualize_worldcover(tif_path: Path, out_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
    classes = set(int(x) for x in np.unique(arr) if x >= 0)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap=_esri_cmap(classes), interpolation="nearest")
    ax.set_title(f"WorldCover / LULC\n{tif_path.name}")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def _visualize_highres_optical(tif_path: Path, out_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        count = src.count
        if count >= 3:
            r = src.read(3).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(1).astype(np.float32)
        else:
            r = g = b = src.read(1).astype(np.float32)
    arr = _normalize_rgb(r, g, b, percentile=(1, 99))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr)
    ax.set_title(f"High-res Optical\n{tif_path.name}")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def _visualize_highres_sar(tif_path: Path, out_path: Path) -> None:
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="gray")
    ax.set_title(f"High-res SAR\n{tif_path.name}")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def _pick_netcdf(raw_dir: Path) -> Path | None:
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        return None
    # 尽量选中间文件，避免首尾缺数
    return files[len(files) // 2]


def _pick_worldcover(raw_root: Path) -> Path | None:
    candidates = [
        raw_root / "esri_lulc_2023" / f"esri_lulc_2023_{raw_root.name}.tif",
        raw_root / "worldcover" / f"worldcover_{raw_root.name}.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 兜底：任意 .tif
    files = list((raw_root / "esri_lulc_2023").rglob("*.tif")) if (raw_root / "esri_lulc_2023").exists() else []
    return files[0] if files else None


def _pick_highres_optical(raw_root: Path) -> Path | None:
    mosaic_dir = raw_root / "highres_optical" / "_mosaic"
    if mosaic_dir.exists():
        files = list(mosaic_dir.rglob("*.tif"))
        if files:
            return files[len(files) // 2]
    # 兜底：用已处理的 patch
    proc_dir = Path(f"/data/xuannv_embedding/processed/{raw_root.name}/patches/highres_optical")
    if proc_dir.exists():
        files = sorted(proc_dir.glob("*.tif"))
        if files:
            return files[len(files) // 2]
    return None


def _pick_highres_sar(raw_root: Path) -> Path | None:
    # 高分辨率 SAR 原始是 zip，直接用已处理 patch 做概览
    proc_dir = Path(f"/data/xuannv_embedding/processed/{raw_root.name}/patches/highres_sar")
    if proc_dir.exists():
        files = sorted(proc_dir.glob("*.tif"))
        if files:
            return files[len(files) // 2]
    return None


def visualize_source(region: str, source: str, output_root: Path) -> Path | None:
    raw_root = Path(f"/data/xuannv_embedding/raw/{region}")
    out_dir = output_root / "source_overview" / region
    out_path = out_dir / f"{source}.png"
    if out_path.exists():
        logger.info("已存在，跳过: %s", out_path)
        return out_path

    if source in ("s2", "s1", "landsat"):
        nc_path = _pick_netcdf(raw_root / source)
        if nc_path is None:
            logger.warning("%s %s 未找到 NetCDF", region, source)
            return None
        _visualize_netcdf(source, nc_path, out_path)
    elif source == "worldcover":
        tif_path = _pick_worldcover(raw_root)
        if tif_path is None:
            logger.warning("%s %s 未找到标签影像", region, source)
            return None
        _visualize_worldcover(tif_path, out_path)
    elif source == "highres_optical":
        tif_path = _pick_highres_optical(raw_root)
        if tif_path is None:
            logger.warning("%s %s 未找到样例影像", region, source)
            return None
        _visualize_highres_optical(tif_path, out_path)
    elif source == "highres_sar":
        tif_path = _pick_highres_sar(raw_root)
        if tif_path is None:
            logger.warning("%s %s 未找到样例影像", region, source)
            return None
        _visualize_highres_sar(tif_path, out_path)
    else:
        raise ValueError(f"未知 source: {source}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化各数据源样例")
    parser.add_argument("--region", required=True, help="区域名，如 haidian / harbin")
    parser.add_argument(
        "--source",
        required=True,
        help="数据源，可选 s2/s1/landsat/worldcover/highres_optical/highres_sar",
    )
    parser.add_argument("--output-root", default="data/visualize", type=Path)
    args = parser.parse_args()

    out = visualize_source(args.region, args.source, args.output_root)
    if out:
        print(out)


if __name__ == "__main__":
    main()
