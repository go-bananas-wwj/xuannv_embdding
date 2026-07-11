from __future__ import annotations

# 月度地理嵌入 Dataset 实现
import json
import logging
import math
import re
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
_MONTH_SUFFIX_RE = re.compile(r"^(?P<patch>.+)_(?P<month>\d{6})$")
_PATCH_ID_RE = re.compile(r"patch_\d+")


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
        statistics_dirs_by_region: dict[str, Path] | None = None,
        supervised_label_roots: dict[str, Path] | None = None,
        region_filter: str | None = None,
        context_margin: int = 0,
        patch_grid_path: Path | None = None,
        teacher_feature_root: Path | None = None,
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
        self.teacher_feature_root = (
            Path(teacher_feature_root) if teacher_feature_root else None
        )
        self.statistics_dirs_by_region = {
            region: Path(path)
            for region, path in (statistics_dirs_by_region or {}).items()
        }
        self.supervised_label_roots = {
            task: Path(path) for task, path in (supervised_label_roots or {}).items()
        }
        self.region_filter = region_filter
        self.context_margin = max(0, int(context_margin))
        self.patch_grid_path = Path(patch_grid_path) if patch_grid_path else None

        self.root_dir = self.manifest_path.parent
        self.manifest: list[dict[str, Any]] = self._load_manifest()
        if self.region_filter is not None:
            self.manifest = [
                entry for entry in self.manifest if entry.get("region") == self.region_filter
            ]
        if self.max_patches is not None:
            self.manifest = self.manifest[: self.max_patches]

        self.statistics: dict[str, dict[str, Any]] = {}
        self.statistics_by_region: dict[str, dict[str, dict[str, Any]]] = {}
        self.source_channels: dict[str, int] = {}
        self._load_statistics()

        self.patch_grid: dict[str, tuple[int, int]] = {}
        self.grid_to_patch: dict[tuple[int, int], str] = {}
        self._load_patch_grid()

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

        metadata_keys = {"patch_id", "region", "source_patch_id"}
        for entry in manifest:
            for key, value in entry.items():
                if key in metadata_keys:
                    continue
                if isinstance(value, list):
                    entry[key] = [Path(p) for p in value]
                elif value is None:
                    entry[key] = None
                else:
                    entry[key] = Path(value)
        return manifest

    def _load_patch_grid(self) -> None:
        """Load patch adjacency metadata for context-window stitching."""
        if self.context_margin <= 0:
            return
        grid_path = self.patch_grid_path
        if grid_path is None:
            candidate = Path("configs/regions") / f"{self.region_filter or 'haidian'}_patches.json"
            if candidate.exists():
                grid_path = candidate
        if grid_path is None or not grid_path.exists():
            logger.warning(
                "context_margin=%d 但 patch_grid_path 不存在，退化为边缘 padding",
                self.context_margin,
            )
            return
        with grid_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        patches = raw if isinstance(raw, list) else raw.get("patches", [])
        if not patches:
            logger.warning("patch_grid 为空: %s", grid_path)
            return
        lefts = sorted({round(float(p["bounds"][0]), 3) for p in patches if "bounds" in p})
        bottoms = sorted({round(float(p["bounds"][1]), 3) for p in patches if "bounds" in p})
        for patch in patches:
            if "bounds" not in patch or "patch_id" not in patch:
                continue
            col = lefts.index(round(float(patch["bounds"][0]), 3))
            row = bottoms.index(round(float(patch["bounds"][1]), 3))
            patch_id = str(patch["patch_id"])
            self.patch_grid[patch_id] = (col, row)
            self.grid_to_patch[(col, row)] = patch_id

    def _load_with_context(self, image_path: Path, nodata: float = 0.0) -> np.ndarray:
        """Read a patch plus a small halo from neighboring patch files.

        Existing processed data is stored as 128x128 per-patch GeoTIFFs. P12 keeps
        the original patch id and geographic footprint, then borrows neighboring
        pixels only for model context. Missing AOI-edge neighbors are padded.
        """
        center = load_tiff(image_path)
        margin = self.context_margin
        if margin <= 0:
            return center

        patch_match = _PATCH_ID_RE.search(image_path.stem)
        if patch_match is None:
            return np.pad(
                center,
                ((0, 0), (margin, margin), (margin, margin)),
                mode="constant",
                constant_values=nodata,
            )
        patch_id = patch_match.group(0)
        if patch_id not in self.patch_grid:
            return np.pad(
                center,
                ((0, 0), (margin, margin), (margin, margin)),
                mode="constant",
                constant_values=nodata,
            )

        channels, height, width = center.shape
        canvas = np.full(
            (channels, height + 2 * margin, width + 2 * margin),
            nodata,
            dtype=center.dtype,
        )
        center_col, center_row = self.patch_grid[patch_id]
        col_radius = max(1, math.ceil(margin / max(width, 1)))
        row_radius = max(1, math.ceil(margin / max(height, 1)))
        for dcol in range(-col_radius, col_radius + 1):
            for drow in range(-row_radius, row_radius + 1):
                neighbor_id = self.grid_to_patch.get((center_col + dcol, center_row + drow))
                if neighbor_id is None:
                    continue
                neighbor_path = image_path.with_name(image_path.name.replace(patch_id, neighbor_id))
                if not neighbor_path.exists():
                    continue
                neighbor = load_tiff(neighbor_path)
                if neighbor.shape[0] != channels:
                    continue
                src_h, src_w = neighbor.shape[-2:]
                row0 = margin - drow * height
                col0 = margin + dcol * width
                dst_y0 = max(row0, 0)
                dst_x0 = max(col0, 0)
                dst_y1 = min(row0 + src_h, canvas.shape[-2])
                dst_x1 = min(col0 + src_w, canvas.shape[-1])
                if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                    continue
                src_y0 = dst_y0 - row0
                src_x0 = dst_x0 - col0
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                canvas[:, dst_y0:dst_y1, dst_x0:dst_x1] = neighbor[
                    :, src_y0:src_y1, src_x0:src_x1
                ]
        return canvas

    @staticmethod
    def _statistics_source_candidates(source: str) -> list[str]:
        candidates = [source]
        for suffix in ("_haidian", "_harbin"):
            if source.endswith(suffix):
                candidates.append(source[: -len(suffix)])
        return list(dict.fromkeys(candidates))

    def _load_stats_file(self, directory: Path, source: str) -> dict[str, Any] | None:
        for candidate in self._statistics_source_candidates(source):
            stat_path = directory / f"{candidate}_stats.json"
            if stat_path.exists():
                with stat_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    def _load_statistics(self) -> None:
        """加载各 source 的 mean/std 统计量，并记录波段数。"""
        for source in self.sources:
            stats = self._load_stats_file(self.statistics_dir, source)
            if stats is not None:
                self.statistics[source] = stats
                self.source_channels[source] = len(stats.get("mean", []))
            elif not self.statistics_dirs_by_region:
                logger.warning(
                    "统计量文件不存在: %s/{%s}_stats.json，将跳过该 source 全局归一化",
                    self.statistics_dir,
                    ",".join(self._statistics_source_candidates(source)),
                )

            for region, directory in self.statistics_dirs_by_region.items():
                region_stats = self._load_stats_file(directory, source)
                if region_stats is None:
                    continue
                self.statistics_by_region.setdefault(region, {})[source] = region_stats
                self.source_channels.setdefault(source, len(region_stats.get("mean", [])))

    def _stats_for(self, region: str | None, source: str) -> dict[str, Any] | None:
        if region is not None:
            region_stats = self.statistics_by_region.get(region, {}).get(source)
            if region_stats is not None:
                return region_stats
        return self.statistics.get(source)

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
                    array = self._load_with_context(self.root_dir / paths[0])
                except Exception as exc:  # pragma: no cover
                    logger.warning("读取 %s 失败: %s", paths[0], exc)
                    continue
                inferred_channels = array.shape[0]
                inferred_hw = (array.shape[1], array.shape[2])
                break

            self.source_channels[source] = inferred_channels
            self.source_hw[source] = inferred_hw or (self.patch_size, self.patch_size)

    @staticmethod
    def _valid_yyyymm(value: str) -> bool:
        try:
            year = int(value[:4])
            month = int(value[4:])
        except ValueError:
            return False
        return 1900 <= year <= 2100 and 1 <= month <= 12

    @classmethod
    def _resolve_label_mask_path(cls, label_root: Path, patch_ids: list[str]) -> Path | None:
        mask_dir = label_root / "masks"
        for patch_id in patch_ids:
            exact = mask_dir / f"{patch_id}.tif"
            if exact.exists():
                return exact
            candidates = sorted(mask_dir.glob(f"{patch_id}_*.tif"))
            if not candidates:
                continue

            def _month(path: Path) -> int:
                match = _MONTH_SUFFIX_RE.match(path.stem)
                if match is None or not cls._valid_yyyymm(match.group("month")):
                    return -1
                return int(match.group("month"))

            return max(candidates, key=_month)
        return None

    def compute_supervised_sampling_weights(
        self,
        positive_boost: float = 4.0,
        max_weight: float = 8.0,
        task_weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Compute per-sample weights from sparse supervised label availability.

        A patch receives extra weight when any configured supervised task has
        positive pixels. The score is task-level rather than pixel-proportional,
        because these labels are extremely sparse and thin masks should still be
        sampled reliably.
        """
        task_weights = task_weights or {}
        weights: list[float] = []
        stats: list[dict[str, Any]] = []
        for entry in self.manifest:
            patch_id: str = entry["patch_id"]
            region = entry.get("region")
            positive_score = 0.0
            task_positive_pixels: dict[str, int] = {}
            for task, label_root in self.supervised_label_roots.items():
                source_patch_id = entry.get("source_patch_id", patch_id)
                label_patch_ids = [patch_id]
                if region is not None and str(region) in label_root.parts:
                    label_patch_ids.append(source_patch_id)
                label_patch_ids = list(dict.fromkeys(label_patch_ids))
                mask_path = self._resolve_label_mask_path(label_root, label_patch_ids)
                if mask_path is None:
                    task_positive_pixels[task] = 0
                    continue
                mask = load_tiff(mask_path)[0]
                mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
                positives = int((mask > 0).sum())
                task_positive_pixels[task] = positives
                if positives > 0:
                    positive_score += float(task_weights.get(task, 1.0))
            weight = 1.0 + float(positive_boost) * positive_score
            weight = min(max(weight, 1.0), float(max_weight))
            weights.append(weight)
            stats.append(
                {
                    "patch_id": patch_id,
                    "region": region,
                    "weight": weight,
                    "positive_score": positive_score,
                    "task_positive_pixels": task_positive_pixels,
                }
            )
        self.supervised_sampling_stats = stats
        return torch.tensor(weights, dtype=torch.double)

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
        region = entry.get("region")

        sample: dict[str, Any] = {
            "patch_id": patch_id,
            "source_frames": {},
            "source_masks": {},
            "source_pixel_masks": {},
            "timestamps": {},
            "supervised_labels": {},
            "supervised_label_masks": {},
        }

        if self.teacher_feature_root is not None:
            teacher_path = (
                self.teacher_feature_root / (region or "haidian") / f"{patch_id}_teacher.pt"
            )
            if teacher_path.exists():
                payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
                sample["teacher_feature"] = payload["feature"].to(torch.float32)
                sample["teacher_valid"] = torch.tensor(True)
            else:
                sample["teacher_feature"] = torch.zeros((32, 32, 1024), dtype=torch.float32)
                sample["teacher_valid"] = torch.tensor(False)

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
                    sample["source_pixel_masks"][source] = torch.zeros(
                        (0, height, width),
                        dtype=torch.float32,
                    )
                    sample["timestamps"][source] = torch.zeros((0,), dtype=torch.long)
                else:
                    sample["source_frames"][source] = torch.zeros(
                        (self.num_months, channels, height, width),
                        dtype=torch.float32,
                    )
                    sample["source_masks"][source] = torch.zeros(
                        (self.num_months,), dtype=torch.float32
                    )
                    sample["source_pixel_masks"][source] = torch.zeros(
                        (self.num_months, height, width),
                        dtype=torch.float32,
                    )
                    sample["timestamps"][source] = month_timestamps
                continue

            frame_list: list[torch.Tensor] = []
            pixel_mask_list: list[torch.Tensor] = []
            timestamp_list: list[int] = []

            for path in paths:
                full_path = self.root_dir / path
                array = self._load_with_context(full_path)
                array = array.astype(np.float32, copy=False)
                mask_path = full_path.with_name(f"{path.stem}_mask.tif")
                if mask_path.exists():
                    pixel_mask = self._load_with_context(mask_path)[0]
                    pixel_mask = np.nan_to_num(
                        pixel_mask,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    pixel_mask = (pixel_mask > 0).astype(np.float32, copy=False)
                    if pixel_mask.shape != array.shape[-2:]:
                        logger.warning(
                            "像素 mask 尺寸不匹配: %s mask=%s image=%s，退化为主波段有效性",
                            mask_path,
                            pixel_mask.shape,
                            array.shape[-2:],
                        )
                        pixel_mask = np.isfinite(array[0]).astype(np.float32)
                else:
                    pixel_mask = np.isfinite(array[0]).astype(np.float32)
                # 将影像中的 NaN/Inf 填充为 0；有效性由独立的 source_masks 控制。
                if np.issubdtype(array.dtype, np.floating):
                    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                stats = self._stats_for(region, source)
                if stats is not None:
                    array = normalize(array, stats["mean"], stats["std"])

                frame_list.append(torch.from_numpy(array))
                pixel_mask_list.append(torch.from_numpy(pixel_mask))
                timestamp_list.append(parse_timestamp_from_filename(path.name))

            # 按时间戳排序，保证时序顺序
            order = np.argsort(timestamp_list)
            frames = torch.stack([frame_list[i] for i in order])
            pixel_masks = torch.stack([pixel_mask_list[i] for i in order])
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
                sample["source_pixel_masks"][source] = pixel_masks
                sample["timestamps"][source] = timestamps
                continue

            if source == "worldcover":
                # WorldCover 是静态 target-only 标签，直接在时间维度取平均后
                # 复制到所有月度 bin，便于下游 prepare_batch 构造逐月目标。
                static_frame = frames.mean(dim=0, keepdim=True)  # (1, C, H, W)
                static_pixel_mask = (pixel_masks.sum(dim=0, keepdim=True) > 0).float()
                binned_frames = static_frame.expand(self.num_months, -1, -1, -1).clone()
                binned_pixel_masks = static_pixel_mask.expand(
                    self.num_months, -1, -1
                ).clone()
                sample["source_frames"][source] = binned_frames
                sample["source_masks"][source] = torch.ones(
                    (self.num_months,), dtype=torch.float32
                )
                sample["source_pixel_masks"][source] = binned_pixel_masks
                sample["timestamps"][source] = month_timestamps
                continue

            # 普通时序 source：按 ``YYYYMM`` 归入月度 bin，同 bin 多帧取平均。
            binned_frames = torch.zeros(
                (self.num_months, channels, height, width),
                dtype=torch.float32,
            )
            binned_masks = torch.zeros((self.num_months,), dtype=torch.float32)
            binned_pixel_masks = torch.zeros(
                (self.num_months, height, width),
                dtype=torch.float32,
            )

            month_indices = torch.tensor(
                [self._yyyymm_to_index(ts.item()) for ts in timestamps],
                dtype=torch.long,
            )
            in_range = (month_indices >= 0) & (month_indices < self.num_months)

            for m in range(self.num_months):
                matches = (month_indices == m) & in_range
                if matches.any():
                    selected_masks = pixel_masks[matches]
                    valid_count = selected_masks.sum(dim=0)
                    binned_frames[m] = (
                        frames[matches] * selected_masks[:, None]
                    ).sum(dim=0) / valid_count.clamp(min=1.0)[None]
                    binned_pixel_masks[m] = (valid_count > 0).float()
                    binned_masks[m] = float(binned_pixel_masks[m].any().item())

            sample["source_frames"][source] = binned_frames
            sample["source_masks"][source] = binned_masks
            sample["source_pixel_masks"][source] = binned_pixel_masks
            sample["timestamps"][source] = month_timestamps

        for task, label_root in self.supervised_label_roots.items():
            source_patch_id = entry.get("source_patch_id", patch_id)
            label_patch_ids = [patch_id]
            if region is not None and str(region) in label_root.parts:
                label_patch_ids.append(source_patch_id)
            label_patch_ids = list(dict.fromkeys(label_patch_ids))
            mask_path = self._resolve_label_mask_path(label_root, label_patch_ids)
            if mask_path is None:
                sample["supervised_labels"][task] = torch.zeros(
                    self.patch_size, self.patch_size, dtype=torch.float32
                )
                sample["supervised_label_masks"][task] = torch.tensor(0.0, dtype=torch.float32)
                continue
            mask = load_tiff(mask_path)
            mask = mask[0].astype(np.float32, copy=False)
            mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))
            sample["supervised_labels"][task] = mask_tensor
            sample["supervised_label_masks"][task] = torch.tensor(1.0, dtype=torch.float32)

        return sample
