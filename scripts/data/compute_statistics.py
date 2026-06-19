"""遍历 processed patches 计算各数据源各波段的均值/标准差，输出 JSON。"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 默认输出根目录，保持与项目数据目录约定一致
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/statistics")


def _valid_mask(values: np.ndarray, nodata: float | None) -> npt.NDArray[np.bool_]:
    """生成有效像素掩码：排除 nan、inf 以及 nodata 值。"""
    mask = np.isfinite(values)
    if nodata is not None and not math.isnan(nodata):
        mask &= values != nodata
    return mask


class _WelfordAccumulator:
    """Welford 在线算法累加器，增量计算单变量流的均值与方差。"""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """接入一个新的有效值数组（已过滤 nan/inf/nodata）。"""
        if values.size == 0:
            return
        # 使用 float64 精度计算，避免大数求和误差
        values = values.astype(np.float64, copy=False)
        n = values.size
        self.count += n

        # 批量 Welford 更新：先得到当前批次均值，再合并到全局统计量
        batch_mean = float(values.mean())
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count

        # 合并平方差：M2_new = M2_old + M2_batch + delta^2 * n_old * n_batch / n_new
        m2_batch = float(((values - batch_mean) ** 2).sum())
        self.m2 += m2_batch + delta * delta * (self.count - n) * n / self.count

    def std(self) -> float:
        """返回总体标准差（除以 n）。"""
        if self.count == 0:
            return float("nan")
        return math.sqrt(self.m2 / self.count)

    def to_dict(self) -> dict[str, float | int]:
        """导出为可序列化的字典。"""
        return {
            "mean": float(self.mean),
            "std": self.std(),
            "count": self.count,
        }


def _collect_tif_files(source_dir: Path, max_patches: int | None) -> list[Path]:
    """收集 source 目录下的所有 .tif 文件（排除 *_mask.tif），支持数量上限。"""
    if not source_dir.exists():
        logger.warning("目录不存在，跳过：%s", source_dir)
        return []

    files = sorted(
        p for p in source_dir.glob("*.tif")
        if p.is_file() and not p.stem.endswith("_mask")
    )
    if not files:
        logger.warning("未找到 .tif 文件：%s", source_dir)
        return []

    if max_patches is not None and max_patches > 0:
        files = files[:max_patches]
    return files


def compute_statistics(
    processed_dir: Path,
    source: str,
    max_patches: int | None = None,
) -> dict[str, list[float] | list[int] | int | str]:
    """计算单个数据源所有波段的统计量。

    参数
    ----------
    processed_dir:
        processed 数据根目录，预期子目录结构为 processed_dir / source / *.tif。
    source:
        数据源名称，例如 "s2"、"s1"、"landsat"。
    max_patches:
        用于快速测试的最大 patch 数量，None 表示处理全部。

    返回
    -------
    dict[str, list[float] | list[int] | int | str]:
        包含以下字段的统计字典：
        - mean: 各波段有效像素均值列表。
        - std: 各波段有效像素标准差列表。
        - count: 所有波段有效像素总数。
        - band_counts: 各波段有效像素数量列表。
        - num_files: 成功处理的文件数量。
        - source: 数据源名称。
    """
    source_dir = processed_dir / source
    files = _collect_tif_files(source_dir, max_patches)
    if not files:
        return {
            "mean": [],
            "std": [],
            "count": 0,
            "num_files": 0,
            "source": source,
        }

    accumulators: list[_WelfordAccumulator] | None = None
    num_success = 0

    for idx, tif_path in enumerate(files, start=1):
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                # 读取所有波段，形状为 (bands, height, width)
                data = src.read(out_dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取文件失败，跳过 %s: %s", tif_path, exc)
            continue

        if accumulators is None:
            num_bands = data.shape[0]
            accumulators = [_WelfordAccumulator() for _ in range(num_bands)]
        else:
            num_bands = data.shape[0]
            if num_bands != len(accumulators):
                logger.warning(
                    "波段数量不一致：%s 有 %d 个波段，期望 %d，跳过该文件",
                    tif_path,
                    num_bands,
                    len(accumulators),
                )
                continue

        for band_idx in range(num_bands):
            band_data = data[band_idx]
            mask = _valid_mask(band_data, nodata)
            accumulators[band_idx].update(band_data[mask])

        num_success += 1
        if idx % 100 == 0 or idx == len(files):
            logger.info("%s: 已处理 %d/%d 个文件", source, idx, len(files))

    if accumulators is None:
        logger.warning("%s: 没有成功读取任何文件", source)
        return {
            "mean": [],
            "std": [],
            "count": 0,
            "num_files": 0,
            "source": source,
        }

    means = [acc.mean for acc in accumulators]
    stds = [acc.std() for acc in accumulators]
    band_counts = [acc.count for acc in accumulators]
    total_count = sum(band_counts)

    stats = {
        "mean": means,
        "std": stds,
        "count": total_count,
        "band_counts": band_counts,
        "num_files": num_success,
        "source": source,
    }

    # 记录每个 band 的简要统计结果
    logger.info(
        "%s: bands=%d, total_valid_pixels=%d, files=%d/%d",
        source,
        len(means),
        total_count,
        num_success,
        len(files),
    )
    for band_idx, (m, s, c) in enumerate(zip(means, stds, band_counts)):
        logger.info("  band %d: mean=%.6f, std=%.6f, count=%d", band_idx, m, s, c)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 processed patches 各数据源各波段的 mean/std 统计量。",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help="processed 数据根目录，例如 /data/xuannv_embedding/processed/harbin/scenes",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="统计量输出目录，例如 /data/xuannv_embedding/statistics/harbin",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["s2", "s1", "landsat"],
        help="需要统计的数据源名称列表",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="每个 source 最多处理的 patch 数量，用于快速测试",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        stats = compute_statistics(
            args.processed_dir,
            source,
            max_patches=args.max_patches,
        )
        out_path = args.output_dir / f"{source}_stats.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("已保存统计文件：%s", out_path)


if __name__ == "__main__":
    main()
