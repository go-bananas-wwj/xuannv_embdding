from __future__ import annotations

# 地理空间工具模块测试：全部使用临时生成的 TIFF，不依赖真实卫星数据。
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling

from xuannv_embedding.utils.geo import (
    get_crs,
    make_patch_grid,
    read_bounds,
    reproject_array,
    reproject_raster_to_match,
)


def _write_temp_tiff(
    path: Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS,
    nodata: float | None = None,
) -> None:
    """写出一个单波段或多半段临时 GeoTIFF。"""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    count, height, width = data.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


@pytest.fixture
def sample_tiff(tmp_path: Path) -> Path:
    """生成一个 10x10、EPSG:32652、100m 分辨率的临时 TIFF。"""
    path = tmp_path / "sample.tif"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    transform = Affine.translation(500_000.0, 4_500_000.0) * Affine.scale(100.0, -100.0)
    _write_temp_tiff(path, data, transform, CRS.from_epsg(32652))
    return path


def test_read_bounds(sample_tiff: Path) -> None:
    """read_bounds 应返回正确的 (left, bottom, right, top)。"""
    bounds = read_bounds(sample_tiff)
    expected = (500_000.0, 4_499_000.0, 501_000.0, 4_500_000.0)
    assert bounds == pytest.approx(expected, rel=1e-6)


def test_get_crs(sample_tiff: Path) -> None:
    """get_crs 应读取到 EPSG:32652。"""
    crs = get_crs(sample_tiff)
    assert crs is not None
    assert crs.to_epsg() == 32652


def test_make_patch_grid_counts() -> None:
    """make_patch_grid 应生成正确数量的不重叠 patch。"""
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    patches = make_patch_grid(bounds, 256.0)
    # 1000 / 256 = 3.90625，所以 x、y 方向各 4 个
    assert len(patches) == 16

    # 验证边界
    assert patches[0] == (0.0, 0.0, 256.0, 256.0)
    # 最后一行/列会被截断到 right/top
    last = patches[-1]
    assert last[2] == pytest.approx(1000.0)
    assert last[3] == pytest.approx(1000.0)

    # 验证不重叠
    for i, p in enumerate(patches):
        left, bottom, right, top = p
        assert left < right
        assert bottom < top
        if i > 0:
            prev = patches[i - 1]
            assert not (left < prev[2] and right > prev[0] and bottom < prev[3] and top > prev[1])


def test_make_patch_grid_negative_size_raises() -> None:
    """patch 尺寸非法时应抛出 ValueError。"""
    with pytest.raises(ValueError):
        make_patch_grid((0.0, 0.0, 100.0, 100.0), 0.0)


def test_reproject_array_downsample(sample_tiff: Path) -> None:
    """reproject_array 应能将 10x10 数组重采样到 5x5。"""
    with rasterio.open(sample_tiff) as src:
        src_array = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    dst_shape = (5, 5)
    dst_transform = from_bounds(500_000.0, 4_499_000.0, 501_000.0, 4_500_000.0, 5, 5)
    dst_crs = CRS.from_epsg(32652)

    dst_array = reproject_array(
        src_array=src_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_shape=dst_shape,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    assert dst_array.shape == dst_shape
    assert dst_array.dtype == src_array.dtype


def test_reproject_raster_to_match_by_path(tmp_path: Path) -> None:
    """reproject_raster_to_match 应能按目标栅格路径完成重投影。"""
    src_path = tmp_path / "src.tif"
    dst_template_path = tmp_path / "template.tif"
    output_path = tmp_path / "aligned.tif"

    # 源栅格：20x20，100m 分辨率
    src_data = np.ones((20, 20), dtype=np.float32) * 5.0
    src_transform = Affine.translation(500_000.0, 4_500_000.0) * Affine.scale(100.0, -100.0)
    _write_temp_tiff(src_path, src_data, src_transform, CRS.from_epsg(32652))

    # 目标栅格：10x10，200m 分辨率，覆盖同一范围
    dst_data = np.zeros((10, 10), dtype=np.float32)
    dst_transform = Affine.translation(500_000.0, 4_500_000.0) * Affine.scale(200.0, -200.0)
    _write_temp_tiff(dst_template_path, dst_data, dst_transform, CRS.from_epsg(32652))

    reproject_raster_to_match(src_path, output_path, dst_template_path)

    with rasterio.open(output_path) as out:
        assert out.shape == (10, 10)
        assert out.crs.to_epsg() == 32652
        assert out.transform == dst_transform
        assert np.all(out.read(1) == pytest.approx(5.0))


def test_reproject_raster_to_match_by_transform(tmp_path: Path) -> None:
    """reproject_raster_to_match 应能按显式 transform / crs / shape 完成重投影。"""
    src_path = tmp_path / "src.tif"
    output_path = tmp_path / "aligned.tif"

    src_data = np.ones((20, 20), dtype=np.float32) * 7.0
    src_transform = Affine.translation(500_000.0, 4_500_000.0) * Affine.scale(100.0, -100.0)
    _write_temp_tiff(src_path, src_data, src_transform, CRS.from_epsg(32652))

    dst_shape = (10, 10)
    dst_transform = Affine.translation(500_000.0, 4_500_000.0) * Affine.scale(200.0, -200.0)
    dst_crs = CRS.from_epsg(32652)

    reproject_raster_to_match(
        src_path,
        output_path,
        dst_transform,
        dst_crs,
        dst_shape,
    )

    with rasterio.open(output_path) as out:
        assert out.shape == dst_shape
        assert out.crs.to_epsg() == 32652
        assert out.transform == dst_transform
        assert np.all(out.read(1) == pytest.approx(7.0))
