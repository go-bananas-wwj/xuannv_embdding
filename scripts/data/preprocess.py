"""对齐多源 NetCDF 时序数据到统一网格并切分为 patches。

读取各 source 目录下的 NetCDF 文件，按配置尺寸生成 patch 网格，
将每个时间步裁剪为固定像素大小的 GeoTIFF，输出到 processed scenes 目录。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from xuannv_embedding.utils.geo import make_patch_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_FILL_MISSING = "zero"
DEFAULT_NODATA = 0.0


@dataclass
class NetCDFInfo:
    """NetCDF 数据集的空间与波段信息。"""

    path: Path
    data_var: str
    crs: CRS
    bounds: tuple[float, float, float, float]
    x_res: float
    y_res: float
    shape: tuple[int, int]  # (height, width)
    band_count: int
    band_names: list[str]
    times: list[datetime]
    src_transform: Affine
    needs_vertical_flip: bool


@dataclass
class PatchInfo:
    """单个 patch 的网格索引与地理边界。"""

    row: int
    col: int
    bounds: tuple[float, float, float, float]

    @property
    def patch_id(self) -> str:
        """行列号格式化字符串，例如 p000_r001。"""
        return f"p{self.col:03d}_r{self.row:03d}"


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


def _build_src_transform(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Affine, tuple[float, float, float, float], bool]:
    """根据 xarray 的 x/y 坐标构建 rasterio 仿射变换与 bounds。

    返回:
        (src_transform, bounds, needs_vertical_flip)
    """
    x_vals = np.asarray(x, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)

    x_res = float(np.abs(np.diff(x_vals).mean()))
    y_res = float(np.abs(np.diff(y_vals).mean()))

    left = float(x_vals.min()) - x_res / 2.0
    right = float(x_vals.max()) + x_res / 2.0
    bottom = float(y_vals.min()) - y_res / 2.0
    top = float(y_vals.max()) + y_res / 2.0

    # 标准 north-up：y 轴向下为负
    src_transform = from_origin(left, top, x_res, y_res)

    # 若 y 坐标自下而上递增，则数组第 0 行对应地理南方，需要垂直翻转
    needs_vertical_flip = bool(y_vals[1] > y_vals[0])

    return src_transform, (left, bottom, right, top), needs_vertical_flip


def read_netcdf_info(nc_path: Path) -> NetCDFInfo:
    """打开 NetCDF 并提取预处理所需的元信息。"""
    ds = xr.open_dataset(nc_path, chunks=None)
    data_var = _find_data_variable(ds)
    da: xr.DataArray = ds[data_var]

    epsg = _extract_epsg(ds)
    crs = CRS.from_epsg(epsg)

    src_transform, bounds, needs_vertical_flip = _build_src_transform(
        np.asarray(ds.x.values), np.asarray(ds.y.values)
    )

    times = [
        datetime.utcfromtimestamp(int(t.astype("datetime64[s]").astype(int)))
        for t in ds.time.values
    ]
    band_names = [str(b) for b in ds.band.values]

    return NetCDFInfo(
        path=nc_path,
        data_var=data_var,
        crs=crs,
        bounds=bounds,
        x_res=src_transform.a,
        y_res=abs(src_transform.e),
        shape=(int(da.sizes["y"]), int(da.sizes["x"])),
        band_count=int(da.sizes["band"]),
        band_names=band_names,
        times=times,
        src_transform=src_transform,
        needs_vertical_flip=needs_vertical_flip,
    )


def build_patch_grid(
    bounds: tuple[float, float, float, float],
    patch_size_m: float,
) -> list[PatchInfo]:
    """基于地理边界和 patch 尺寸生成带行列号的 patch 列表。"""
    raw_patches = make_patch_grid(bounds, patch_size_m)
    # make_patch_grid 从 left 开始，x 外层循环、y 内层循环；需要映射为 (row, col)
    left, bottom, right, top = bounds
    n_rows = int(np.ceil((top - bottom) / patch_size_m))
    patches: list[PatchInfo] = []
    for idx, pbounds in enumerate(raw_patches):
        col = idx // n_rows
        row = idx % n_rows
        # 与 col/row 视觉一致：row 从南向北递增
        patches.append(PatchInfo(row=row, col=col, bounds=pbounds))
    return patches


def _parse_time(time_value: Any) -> datetime:
    """将 xarray time 坐标值转换为 datetime。"""
    if isinstance(time_value, datetime):
        return time_value
    if isinstance(time_value, np.datetime64):
        return datetime.utcfromtimestamp(int(time_value.astype("datetime64[s]").astype(int)))
    if hasattr(time_value, "to_pydatetime"):
        return time_value.to_pydatetime()
    raise ValueError(f"无法解析时间值: {time_value} (类型 {type(time_value)})")


def extract_patch(
    ds: xr.Dataset,
    data_var: str,
    time_idx: int,
    patch: PatchInfo,
    patch_size_px: int,
    dst_crs: CRS,
    fill_value: float = DEFAULT_NODATA,
) -> tuple[np.ndarray, Affine] | None:
    """从 NetCDF 中裁剪并重采样单个 patch。

    返回:
        (array, transform)，array 形状为 (band, height, width)；
        若 patch 完全为空且配置跳过，则返回 None。
    """
    left, bottom, right, top = patch.bounds
    dst_transform = from_bounds(left, bottom, right, top, width=patch_size_px, height=patch_size_px)

    da: xr.DataArray = ds[data_var]
    src_array = da.isel(time=time_idx).values.astype(np.float32)

    # 若 y 坐标递增，需要垂直翻转使数组为 north-up
    y_coords = np.asarray(ds.y.values, dtype=np.float64)
    if y_coords[1] > y_coords[0]:
        src_array = src_array[:, ::-1, :]

    src_crs = CRS.from_epsg(int(ds.coords["epsg"].values))
    src_transform, _, _ = _build_src_transform(np.asarray(ds.x.values), np.asarray(ds.y.values))

    dst_array = np.empty((src_array.shape[0], patch_size_px, patch_size_px), dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=fill_value,
    )

    # 将 NaN 也替换为填充值，避免 GeoTIFF 中出现无效值
    if np.isnan(dst_array).any():
        dst_array = np.nan_to_num(dst_array, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return dst_array, dst_transform


def write_patch_tiff(
    array: np.ndarray,
    out_path: Path,
    transform: Affine,
    crs: CRS,
    band_names: list[str],
    nodata: float = DEFAULT_NODATA,
) -> None:
    """将 patch 数组写入多波段 GeoTIFF。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array)
        # 写入 band 描述，便于后续识别波段
        for idx, name in enumerate(band_names, start=1):
            dst.set_band_description(idx, name)


def patchify_source(
    source: str,
    raw_root: Path,
    output_root: Path,
    dst_crs: CRS,
    patch_size_m: float,
    patch_size_px: int,
    fill_missing: str,
    overwrite: bool,
) -> int:
    """处理单个数据源的所有 NetCDF 文件并输出 patches。

    返回:
        实际写入的 patch 数量。
    """
    source_dir = raw_root / source
    if not source_dir.exists():
        logger.warning("source 目录不存在，跳过: %s", source_dir)
        return 0

    nc_files = sorted(source_dir.glob("*.nc"))
    if not nc_files:
        logger.warning("未找到 NetCDF 文件: %s", source_dir)
        return 0

    # 使用第一个 NetCDF 的空间范围生成统一 patch 网格
    first_info = read_netcdf_info(nc_files[0])
    logger.info(
        "%s: 使用 %s 作为参考网格, bounds=%s, shape=%s, bands=%d",
        source,
        nc_files[0].name,
        first_info.bounds,
        first_info.shape,
        first_info.band_count,
    )
    patches = build_patch_grid(first_info.bounds, patch_size_m)
    logger.info(
        "%s: 生成 %d 个 patches (%d m × %d m)",
        source,
        len(patches),
        patch_size_m,
        patch_size_m,
    )

    written = 0
    skip_empty = fill_missing == "skip"

    for nc_path in nc_files:
        logger.info("%s: 处理 %s", source, nc_path.name)
        ds = xr.open_dataset(nc_path, chunks=None)
        data_var = _find_data_variable(ds)
        da: xr.DataArray = ds[data_var]

        for time_idx in range(int(da.sizes["time"])):
            time_val = _parse_time(ds.time.values[time_idx])
            date_str = time_val.strftime("%Y%m%d")

            for patch in tqdm(
                patches,
                desc=f"{source} {date_str}",
                leave=False,
            ):
                out_path = output_root / "patches" / source / f"{source}_{date_str}_{patch.patch_id}.tif"
                if out_path.exists() and not overwrite:
                    continue

                result = extract_patch(
                    ds=ds,
                    data_var=data_var,
                    time_idx=time_idx,
                    patch=patch,
                    patch_size_px=patch_size_px,
                    dst_crs=dst_crs,
                    fill_value=DEFAULT_NODATA,
                )
                if result is None:
                    continue
                array, transform = result

                # 判断是否为空 patch
                if skip_empty and np.all(array == DEFAULT_NODATA):
                    continue

                write_patch_tiff(
                    array=array,
                    out_path=out_path,
                    transform=transform,
                    crs=dst_crs,
                    band_names=[str(b) for b in ds.band.values],
                    nodata=DEFAULT_NODATA,
                )
                written += 1

    return written


def load_config(path: Path) -> dict[str, Any]:
    """加载 JSON 配置文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="将 NetCDF 时序数据对齐并切分为 patches",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON 配置文件路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 patch 文件",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="仅处理指定 source（调试用）",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    region = cfg["region"]
    raw_root = Path(cfg["raw_root"])
    output_root = Path(cfg["output_root"])
    dst_crs = CRS.from_string(cfg["crs"])
    patch_size_m = float(cfg["patch_size_m"])
    patch_size_px = int(cfg["patch_size_px"])
    sources = cfg["sources"]
    fill_missing = cfg.get("fill_missing", DEFAULT_FILL_MISSING)

    if args.source is not None:
        sources = [args.source]

    logger.info(
        "开始预处理: region=%s raw_root=%s output_root=%s crs=%s",
        region,
        raw_root,
        output_root,
        dst_crs,
    )

    total_written = 0
    for source in sources:
        count = patchify_source(
            source=source,
            raw_root=raw_root,
            output_root=output_root,
            dst_crs=dst_crs,
            patch_size_m=patch_size_m,
            patch_size_px=patch_size_px,
            fill_missing=fill_missing,
            overwrite=args.overwrite,
        )
        total_written += count
        logger.info("%s: 写入 %d 个 patches", source, count)

    logger.info("预处理完成: 共写入 %d 个 patches", total_written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
