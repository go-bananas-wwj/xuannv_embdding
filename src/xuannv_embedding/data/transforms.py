from __future__ import annotations

# 影像读取、归一化与时间戳解析工具
import logging
import re
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# 匹配文件名中的 YYYYMMDD 日期字段
_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")


def load_tiff(path: Path) -> np.ndarray:
    """使用 rasterio 读取 TIFF 影像。

    参数:
        path: TIFF 文件路径。

    返回:
        形状为 ``[C, H, W]`` 的 numpy 数组。
    """
    with rasterio.open(path) as src:
        array: np.ndarray = src.read()
    return array


def normalize(
    array: np.ndarray,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """逐波段进行标准化。

    参数:
        array: 形状为 ``[C, H, W]`` 的原始数组。
        mean: 每个波段的均值，长度需与 ``C`` 一致。
        std: 每个波段的标准差，长度需与 ``C`` 一致。

    返回:
        标准化后的 float32 数组。
    """
    array = array.astype(np.float32, copy=False)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    # 避免标准差为 0 导致除零
    std_safe = np.where(std_arr == 0.0, 1.0, std_arr)
    return (array - mean_arr) / std_safe


def parse_timestamp_from_filename(filename: str) -> int:
    """从文件名中解析时间戳，返回 ``YYYYMM`` 格式的整数。

    参数:
        filename: 影像文件名，例如 ``s2_20250129_p000_r000.tif``。

    返回:
        形如 ``202501`` 的月度索引整数。

    异常:
        ValueError: 文件名中未找到合法的 ``YYYYMMDD`` 日期字段。
    """
    base_name = Path(filename).name
    match = _DATE_RE.search(base_name)
    if not match:
        raise ValueError(f"无法从文件名解析时间戳: {filename}")
    year = int(match.group(1))
    month = int(match.group(2))
    return year * 100 + month
