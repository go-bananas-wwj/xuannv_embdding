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
    ) -> None:
        """初始化 Dataset。

        参数:
            manifest_path: manifest JSON 文件路径。
            statistics_dir: 存放 ``{source}_stats.json`` 的目录。
            sources: 需要加载的数据源列表。
            patch_size: 缺失 source 时用于构造空张量的默认空间尺寸。
            max_patches: 最大样本数限制，主要用于快速测试。
        """
        self.manifest_path = Path(manifest_path)
        self.statistics_dir = Path(statistics_dir)
        self.sources = list(sources)
        self.patch_size = patch_size
        self.max_patches = max_patches

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

        for source in self.sources:
            paths = entry.get(source)
            if not paths:
                # 缺失 source：构造空张量，保持维度一致
                channels = self.source_channels[source]
                height, width = self.source_hw[source]
                sample["source_frames"][source] = torch.zeros(
                    (0, channels, height, width),
                    dtype=torch.float32,
                )
                sample["source_masks"][source] = torch.zeros((0,), dtype=torch.float32)
                sample["timestamps"][source] = torch.zeros((0,), dtype=torch.long)
                continue

            frame_list: list[torch.Tensor] = []
            mask_list: list[float] = []
            timestamp_list: list[int] = []

            for path in paths:
                array = load_tiff(self.root_dir / path)
                array = array.astype(np.float32, copy=False)
                if source in self.statistics:
                    stats = self.statistics[source]
                    array = normalize(array, stats["mean"], stats["std"])

                frame_list.append(torch.from_numpy(array))
                mask_list.append(1.0)
                timestamp_list.append(parse_timestamp_from_filename(path.name))

            # 按时间戳排序，保证时序顺序
            order = np.argsort(timestamp_list)
            frames = torch.stack([frame_list[i] for i in order])
            masks = torch.tensor(
                [mask_list[i] for i in order],
                dtype=torch.float32,
            )
            timestamps = torch.tensor(
                [timestamp_list[i] for i in order],
                dtype=torch.long,
            )

            sample["source_frames"][source] = frames
            sample["source_masks"][source] = masks
            sample["timestamps"][source] = timestamps

        return sample
