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
        teacher_embedding_root: Path | None = None,
        region: str | None = None,
        cache_dir: Path | None = None,
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
            teacher_embedding_root: 教师 embedding 根目录，
                按 ``{region}/{patch_id}/202512_embedding_map.pt`` 组织。
            region: 当前 dataset 的默认 region；manifest 条目中的 ``region`` 字段优先级更高。
            cache_dir: 预处理后样本缓存目录；None 表示禁用缓存。
        """
        self.manifest_path = Path(manifest_path)
        self.statistics_dir = Path(statistics_dir)
        self.sources = list(sources)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.num_months = num_months
        self.ref_year = ref_year
        self.ref_month = ref_month
        self.teacher_embedding_root = (
            Path(teacher_embedding_root) if teacher_embedding_root else None
        )
        self.region = region
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.month_bins = self._generate_month_bins()

        self.root_dir = self.manifest_path.parent
        self.manifest: list[dict[str, Any]] = self._load_manifest()
        if self.max_patches is not None:
            self.manifest = self.manifest[: self.max_patches]

        self.statistics: dict[tuple[str, str], dict[str, Any]] = {}
        self.source_channels: dict[str, int] = {}
        self._load_statistics()
        self._ensure_cache_valid()

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
        """加载各 (region, source) 的 mean/std 统计量，并记录波段数。

        优先读取 ``statistics_dir/{region}/{source}_stats.json``，
        不存在时回退到 ``statistics_dir/{source}_stats.json``。
        """
        regions: set[str] = set()
        for entry in self.manifest:
            regions.add(entry.get("region", self.region or ""))
        if "" in regions and self.region:
            regions.discard("")
            regions.add(self.region)

        for region in regions:
            for source in self.sources:
                candidates = [self.statistics_dir / f"{source}_stats.json"]
                if region:
                    candidates.insert(0, self.statistics_dir / region / f"{source}_stats.json")
                for stat_path in candidates:
                    if stat_path.exists():
                        with stat_path.open("r", encoding="utf-8") as f:
                            stats: dict[str, Any] = json.load(f)
                        self.statistics[(region, source)] = stats
                        self.source_channels[source] = len(stats.get("mean", []))
                        break
                else:
                    logger.warning(
                        "统计量文件不存在: %s，将跳过 region=%s source=%s 归一化",
                        candidates,
                        region,
                        source,
                    )

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

    def _cache_dir_for_patch(self, entry: dict[str, Any]) -> Path:
        assert self.cache_dir is not None
        region = entry.get("region", self.region or "")
        return self.cache_dir / f"preprocessed_{self.patch_size}" / str(region)

    def _cache_file_for(self, entry: dict[str, Any]) -> Path:
        return self._cache_dir_for_patch(entry) / f"{entry['patch_id']}.pt"

    def _compute_stats_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for key in sorted(self.statistics.keys()):
            stats = self.statistics[key]
            h.update(json.dumps(stats, sort_keys=True).encode("utf-8"))
        return h.hexdigest()[:16]

    def _cache_meta(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "num_months": self.num_months,
            "sources": sorted(self.sources),
            "statistics_hash": self._compute_stats_hash(),
            "teacher_embedding_root": (
                str(self.teacher_embedding_root) if self.teacher_embedding_root else None
            ),
            "version": "1.0",
        }

    def _ensure_cache_valid(self) -> None:
        if self.cache_dir is None:
            return

        preproc_dir = self.cache_dir / f"preprocessed_{self.patch_size}"
        meta_path = preproc_dir / "cache_meta.json"
        current_meta = self._cache_meta()

        if meta_path.exists():
            try:
                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if old_meta == current_meta:
                    return
                logger.info("Cache meta changed, clearing %s", preproc_dir)
            except Exception as exc:
                logger.warning("Failed to read cache meta %s: %s", meta_path, exc)

        if preproc_dir.exists():
            import shutil

            shutil.rmtree(preproc_dir)
        preproc_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(current_meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        """从磁盘加载并预处理一个样本（不含缓存）。"""
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
                region = entry.get("region", self.region or "")
                stats_key = (region, source)
                if stats_key in self.statistics:
                    stats = self.statistics[stats_key]
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

        # 加载教师 embedding；缺失时用零占位，保证 collate 可拼接。
        region = entry.get("region", self.region or "")
        temporal_sources = [s for s in self.sources if not s.startswith("highres")]
        if temporal_sources:
            ref_hw = self.source_hw.get(temporal_sources[0], (self.patch_size, self.patch_size))
        else:
            ref_hw = (self.patch_size, self.patch_size)

        if self.teacher_embedding_root is not None and region:
            teacher_path = (
                self.teacher_embedding_root / region / patch_id / "202512_embedding_map.pt"
            )
            if teacher_path.exists():
                sample["teacher_embedding_map"] = torch.load(teacher_path, map_location="cpu")
            else:
                logger.warning("Teacher embedding missing: %s", teacher_path)
                sample["teacher_embedding_map"] = torch.zeros(64, *ref_hw, dtype=torch.float32)
        else:
            sample["teacher_embedding_map"] = torch.zeros(64, *ref_hw, dtype=torch.float32)

        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """获取单个样本；优先返回磁盘缓存。"""
        entry = self.manifest[idx]

        if self.cache_dir is not None:
            cache_file = self._cache_file_for(entry)
            if cache_file.exists():
                try:
                    return torch.load(cache_file, map_location="cpu", weights_only=False)
                except Exception as exc:
                    logger.warning("Failed to load cache %s: %s", cache_file, exc)
                    cache_file.unlink(missing_ok=True)

        sample = self._load_sample(idx)

        if self.cache_dir is not None:
            cache_file = self._cache_file_for(entry)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            try:
                torch.save(sample, tmp)
                tmp.replace(cache_file)
            except Exception as exc:
                logger.warning("Failed to save cache %s: %s", cache_file, exc)
                tmp.unlink(missing_ok=True)

        return sample
