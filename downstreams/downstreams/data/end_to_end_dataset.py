from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from downstreams.data.multi_task_dataset import apply_augment


_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")


def _normalize_months(months: str | int | list[str | int]) -> list[str]:
    if isinstance(months, (str, int)):
        return [str(months)]
    return [str(m) for m in months]


def _parse_month_from_filename(filename: str) -> str:
    match = _DATE_RE.search(filename)
    if not match:
        raise ValueError(f"无法从文件名解析时间戳: {filename}")
    return f"{match.group(1)}{match.group(2)}"


def _resolve_mask_path(mask_dir: Path, patch_id: str, months: list[str]) -> Path:
    """优先按月份后缀、再按无后缀查找 mask 路径。"""
    for month in months:
        path = mask_dir / f"{patch_id}_{month}.tif"
        if path.exists():
            return path
    path = mask_dir / f"{patch_id}.tif"
    if not path.exists():
        raise FileNotFoundError(f"mask 不存在: {path}")
    return path


class EndToEndSegmentationDataset(Dataset):
    """端到端分割数据集：直接从高分光学影像加载并做同步增强。"""

    def __init__(
        self,
        image_root: Path,
        label_root: Path,
        patch_ids: list[str],
        task_name: str = "construction",
        months: str | int | list[str | int] = "202605",
        augment: bool = False,
        bitemporal: bool = False,
        include_diff: bool = True,
        crop_size: int | None = None,
        highres_source_name: str | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.task_name = task_name
        self.months = _normalize_months(months)
        self.augment = augment
        self.bitemporal = bitemporal
        self.include_diff = include_diff
        self.crop_size = crop_size
        self.highres_source_name = highres_source_name
        self.mask_dir = self.label_root / "masks"

        if self.bitemporal and len(self.months) != 2:
            raise ValueError("bitemporal=True 需要指定两个月份")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        images = [self._load_image(patch_id, m) for m in self.months]
        mask = self._load_mask(patch_id)

        # 各时相影像空间尺寸必须一致
        for i in range(1, len(images)):
            if images[i].shape[-2:] != images[0].shape[-2:]:
                raise ValueError(
                    f"{patch_id} 时相 {self.months[i]} 影像尺寸 "
                    f"{tuple(images[i].shape[-2:])} 与时相 {self.months[0]} "
                    f"{tuple(images[0].shape[-2:])} 不一致"
                )

        if self.bitemporal:
            img_t1, img_t2 = images
            image = torch.cat([img_t1, img_t2], dim=0)
            if self.include_diff:
                diff = torch.abs(img_t1 - img_t2)
                image = torch.cat([image, diff], dim=0)
        else:
            image = images[0]

        if self.crop_size is not None:
            image, mask = self._crop_pair(image, mask)
        if self.augment:
            image, mask = apply_augment(image, mask, None)

        return {
            "image": image,
            "mask": mask,
            "patch_id": patch_id,
        }

    def _load_image(self, patch_id: str, month: str) -> torch.Tensor:
        pattern = f"highres_optical_{month}*_{patch_id}.tif"
        matches = sorted(self.image_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"未找到 {patch_id} 在月份 {month} 的高分影像: {self.image_root / pattern}"
            )
        path = matches[0]
        with rasterio.open(path) as src:
            array = src.read().astype(np.float32, copy=False)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(array)

    def _load_mask(self, patch_id: str) -> torch.Tensor:
        path = _resolve_mask_path(self.mask_dir, patch_id, self.months)
        with rasterio.open(path) as src:
            return torch.from_numpy(src.read(1)).long()

    def _crop_pair(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """在 mask 坐标系下计算裁剪窗口，再按比例映射到影像坐标系。

        训练时采用前景感知随机裁剪；验证/测试时采用中心裁剪，保证可复现。
        """
        _, img_h, img_w = image.shape
        mask_h, mask_w = mask.shape
        if mask_h <= self.crop_size or mask_w <= self.crop_size:
            return image, mask

        if self.augment:
            fg_indices = torch.nonzero(mask > 0, as_tuple=False)
            use_fg = fg_indices.numel() > 0 and torch.rand(1).item() < 0.7
            if use_fg:
                idx = torch.randint(0, fg_indices.shape[0], (1,)).item()
                cy, cx = fg_indices[idx].tolist()
            else:
                cy = torch.randint(0, mask_h, (1,)).item()
                cx = torch.randint(0, mask_w, (1,)).item()
        else:
            cy = mask_h // 2
            cx = mask_w // 2

        y1 = min(max(cy - self.crop_size // 2, 0), mask_h - self.crop_size)
        x1 = min(max(cx - self.crop_size // 2, 0), mask_w - self.crop_size)
        y2 = y1 + self.crop_size
        x2 = x1 + self.crop_size

        scale_y = img_h / mask_h
        scale_x = img_w / mask_w
        img_y1 = int(y1 * scale_y)
        img_x1 = int(x1 * scale_x)
        img_y2 = int(y2 * scale_y)
        img_x2 = int(x2 * scale_x)

        return image[:, img_y1:img_y2, img_x1:img_x2], mask[y1:y2, x1:x2]


class JointEndToEndSegmentationDataset(Dataset):
    """跨 region 端到端数据集封装，将带 region 前缀的 patch_id 路由到对应子数据集。"""

    def __init__(
        self,
        image_root: Path,
        mask_dirs: dict[str, Path],
        patch_ids: list[str],
        region_of: dict[str, str],
        task_name: str = "construction",
        months: list[str] | None = None,
        augment: bool = False,
        bitemporal: bool = False,
        include_diff: bool = True,
        crop_size: int | None = None,
        highres_source_names: dict[str, str] | None = None,
    ) -> None:
        self.patch_ids = patch_ids
        self.region_of = region_of
        self.task_name = task_name
        self.months = months if months is not None else ["202605"]
        self.augment = augment
        self.bitemporal = bitemporal
        self.include_diff = include_diff
        self.crop_size = crop_size
        self.highres_source_names = highres_source_names or {}

        self._region_datasets: dict[str, EndToEndSegmentationDataset] = {}
        self._region_patch_ids: dict[str, list[str]] = {}

        for region in sorted(set(region_of.values())):
            region_patch_ids = [pid for pid in patch_ids if region_of.get(pid) == region]
            if not region_patch_ids:
                continue
            original_ids = [self._strip_region_prefix(pid, region) for pid in region_patch_ids]
            self._region_datasets[region] = EndToEndSegmentationDataset(
                image_root=image_root / region / "patches" / "highres_optical",
                label_root=mask_dirs[region].parent,
                patch_ids=original_ids,
                task_name=task_name,
                months=self.months,
                augment=augment,
                bitemporal=bitemporal,
                include_diff=include_diff,
                crop_size=crop_size,
                highres_source_name=self.highres_source_names.get(region),
            )
            self._region_patch_ids[region] = region_patch_ids

    @staticmethod
    def _strip_region_prefix(patch_id: str, region: str) -> str:
        prefix = f"{region}_"
        if patch_id.startswith(prefix):
            return patch_id[len(prefix) :]
        raise ValueError(f"patch_id {patch_id} 不包含 region 前缀 {prefix}")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefixed_id = self.patch_ids[idx]
        region = self.region_of[prefixed_id]
        ds = self._region_datasets[region]
        local_id = self._strip_region_prefix(prefixed_id, region)
        local_idx = ds.patch_ids.index(local_id)
        sample = ds[local_idx]
        sample["patch_id"] = prefixed_id
        return sample


def collate_end_to_end(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_ids": [b["patch_id"] for b in batch],
    }
