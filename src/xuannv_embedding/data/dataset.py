from __future__ import annotations

# 月度地理嵌入 Dataset 实现
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from xuannv_embedding.data.transforms import (
    load_tiff,
    normalize,
    parse_timestamp_from_filename,
)

logger = logging.getLogger(__name__)


class MonthlyEmbeddingDataset(Dataset):
    """按月组织的多源时序影像 Dataset。

    每个样本对应一个 patch，包含多个 source 的时序帧、availability mask 与时间戳。
    缺失的 source 会以 ``T=0`` 的空张量返回，保证下游模型可统一处理。
    """

    def __init__(
        self,
        manifest_path: Path,
        statistics_dir: Path,
        sources: list[str],
        patch_size: int = 256,
        max_patches: int | None = None,
        num_months: int = 17,
        ref_year: int = 2025,
        ref_month: int = 1,
    ) -> None:
        """初始化 Dataset。

        参数:
            manifest_path: manifest JSON 文件路径。
            statistics_dir: 存放 ``{source}_stats.json`` 的目录。
            sources: 需要加载的数据源列表。
            patch_size: 缺失 source 时用于构造空张量的默认空间尺寸。
            max_patches: 最大样本数限制，主要用于快速测试。
            num_months: 月度 bin 数量，阶段一默认为 17（2025-01 至 2026-05）。
            ref_year: 月度 bin 起始年份。
            ref_month: 月度 bin 起始月份。
        """
        self.manifest_path = Path(manifest_path)
        self.statistics_dir = Path(statistics_dir)
        self.sources = list(sources)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.num_months = num_months
        self.ref_year = ref_year
        self.ref_month = ref_month
        self.month_bins = self._generate_month_bins()

        self.root_dir = self.manifest_path.parent
        self.manifest: list[dict[str, Any]] = self._load_manifest()
        if self.max_patches is not None:
            self.manifest = self.manifest[: self.max_patches]

        self.statistics: dict[str, dict[str, Any]] = {}
        self.source_channels: dict[str, int] = {}
        self._load_statistics()

        # 记录每个 source 的空间尺寸，缺失时用于构造空张量
        self.source_hw: dict[str, tuple[int, int]] = {}
        self._infer_geometry()

    def _generate_month_bins(self) -> list[int]:
        """生成 ``num_months`` 个月度 bin 对应的 ``YYYYMM`` 列表。"""
        year, month = self.ref_year, self.ref_month
        bins: list[int] = []
        for _ in range(self.num_months):
            bins.append(year * 100 + month)
            month += 1
            if month > 12:
                month = 1
                year += 1
        return bins

    def _yyyymm_to_index(self, timestamp: int) -> int:
        """将 ``YYYYMM`` 整数时间戳映射到以 ``ref_year/ref_month`` 为 0 的月度索引。"""
        year = timestamp // 100
        month = timestamp % 100
        return (year - self.ref_year) * 12 + (month - self.ref_month)

    def _load_manifest(self) -> list[dict[str, Any]]:
        """加载 manifest 并将路径字符串转换为 ``Path``。"""
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest: list[dict[str, Any]] = json.load(f)

        for entry in manifest:
            for key, value in entry.items():
                if key == "patch_id":
                    continue
                if isinstance(value, list):
                    entry[key] = [Path(p) for p in value]
                elif value is None:
                    entry[key] = None
                else:
                    entry[key] = Path(value)
        return manifest

    def _load_statistics(self) -> None:
        """加载各 source 的 mean/std 统计量，并记录波段数。"""
        for source in self.sources:
            stat_path = self.statistics_dir / f"{source}_stats.json"
            if stat_path.exists():
                with stat_path.open("r", encoding="utf-8") as f:
                    stats: dict[str, Any] = json.load(f)
                self.statistics[source] = stats
                self.source_channels[source] = len(stats.get("mean", []))
            else:
                logger.warning("统计量文件不存在: %s，将跳过该 source 归一化", stat_path)

    def _infer_geometry(self) -> None:
        """推断每个 source 的通道数与空间尺寸。"""
        for source in self.sources:
            inferred_channels = self.source_channels.get(source, 0)
            inferred_hw: tuple[int, int] | None = None

            for entry in self.manifest:
                paths = entry.get(source)
                if not paths:
                    continue
                try:
                    array = load_tiff(self.root_dir / paths[0])
                except Exception as exc:  # pragma: no cover
                    logger.warning("读取 %s 失败: %s", paths[0], exc)
                    continue
                inferred_channels = array.shape[0]
                inferred_hw = (array.shape[1], array.shape[2])
                break

            self.source_channels[source] = inferred_channels
            self.source_hw[source] = inferred_hw or (self.patch_size, self.patch_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取单个样本。

        返回:
            包含 ``patch_id``、``source_frames``、``source_masks``、``timestamps`` 的字典。
            其中 ``timestamps`` 按 source 组织，便于处理不同 source 的时序差异。
        """
        entry = self.manifest[idx]
        patch_id: str = entry["patch_id"]

        sample: dict[str, Any] = {
            "patch_id": patch_id,
            "source_frames": {},
            "source_masks": {},
            "timestamps": {},
        }

        month_timestamps = torch.tensor(self.month_bins, dtype=torch.long)

        for source in self.sources:
            paths = entry.get(source)
            channels = self.source_channels[source]
            height, width = self.source_hw[source]

            if not paths:
                # 缺失 source：按月度 bin 构造空张量，保持维度一致
                if source.startswith("highres"):
                    sample["source_frames"][source] = torch.zeros(
                        (0, channels, height, width),
                        dtype=torch.float32,
                    )
                    sample["source_masks"][source] = torch.zeros((0,), dtype=torch.float32)
                    sample["timestamps"][source] = torch.zeros((0,), dtype=torch.long)
                else:
                    sample["source_frames"][source] = torch.zeros(
                        (self.num_months, channels, height, width),
                        dtype=torch.float32,
                    )
                    sample["source_masks"][source] = torch.zeros(
                        (self.num_months,), dtype=torch.float32
                    )
                    sample["timestamps"][source] = month_timestamps
                continue

            frame_list: list[torch.Tensor] = []
            timestamp_list: list[int] = []

            for path in paths:
                array = load_tiff(self.root_dir / path)
                array = array.astype(np.float32, copy=False)
                # 将影像中的 NaN/Inf 填充为 0；有效性由独立的 source_masks 控制。
                if np.issubdtype(array.dtype, np.floating):
                    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                if source in self.statistics:
                    stats = self.statistics[source]
                    array = normalize(array, stats["mean"], stats["std"])

                frame_list.append(torch.from_numpy(array))
                timestamp_list.append(parse_timestamp_from_filename(path.name))

            # 按时间戳排序，保证时序顺序
            order = np.argsort(timestamp_list)
            frames = torch.stack([frame_list[i] for i in order])
            timestamps = torch.tensor(
                [timestamp_list[i] for i in order],
                dtype=torch.long,
            )

            if source.startswith("highres"):
                # 高分辨率 source 不参与月度 binning，保留原始时序。
                sample["source_frames"][source] = frames
                sample["source_masks"][source] = torch.ones(
                    (frames.shape[0],), dtype=torch.float32
                )
                sample["timestamps"][source] = timestamps
                continue

            if source == "worldcover":
                # WorldCover 是静态 target-only 标签，直接在时间维度取平均后
                # 复制到所有月度 bin，便于下游 prepare_batch 构造逐月目标。
                static_frame = frames.mean(dim=0, keepdim=True)  # (1, C, H, W)
                binned_frames = static_frame.expand(self.num_months, -1, -1, -1).clone()
                sample["source_frames"][source] = binned_frames
                sample["source_masks"][source] = torch.ones(
                    (self.num_months,), dtype=torch.float32
                )
                sample["timestamps"][source] = month_timestamps
                continue

            # 普通时序 source：按 ``YYYYMM`` 归入月度 bin，同 bin 多帧取平均。
            binned_frames = torch.zeros(
                (self.num_months, channels, height, width),
                dtype=torch.float32,
            )
            binned_masks = torch.zeros((self.num_months,), dtype=torch.float32)

            month_indices = torch.tensor(
                [self._yyyymm_to_index(ts.item()) for ts in timestamps],
                dtype=torch.long,
            )
            in_range = (month_indices >= 0) & (month_indices < self.num_months)

            for m in range(self.num_months):
                matches = (month_indices == m) & in_range
                if matches.any():
                    binned_frames[m] = frames[matches].mean(dim=0)
                    binned_masks[m] = 1.0

            sample["source_frames"][source] = binned_frames
            sample["source_masks"][source] = binned_masks
            sample["timestamps"][source] = month_timestamps

        return sample
