from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
import torch
from torch.utils.data import Dataset


class MultiTaskEmbeddingDataset(Dataset):
    """多任务下游数据集，支持单时相/双时相 embedding 与差分拼接。"""

    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        task_name: str = "construction",
        months: str | int | list[str | int] = "202605",
        augment: bool = False,
        bitemporal: bool = False,
        include_diff: bool = True,
        crop_size: int | None = None,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.task_name = task_name
        self.months = self._normalize_months(months)
        self.augment = augment
        self.bitemporal = bitemporal
        self.include_diff = include_diff
        self._crop_size = crop_size
        self.mask_dir = self.label_root / "masks"

        if self.bitemporal and len(self.months) != 2:
            raise ValueError("bitemporal=True 需要指定两个月份")

    @staticmethod
    def _normalize_months(months: str | int | list[str | int]) -> list[str]:
        if isinstance(months, (str, int)):
            return [str(months)]
        return [str(m) for m in months]

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        embeddings = [self._load_embedding(patch_id, m) for m in self.months]
        mask = self._load_mask(patch_id)

        for emb in embeddings:
            if emb.shape[-2:] != mask.shape:
                raise ValueError(
                    f"{patch_id} embedding {tuple(emb.shape[-2:])} 与 mask {tuple(mask.shape)} 尺寸不一致"
                )

        if self.bitemporal:
            emb_t1, emb_t2 = embeddings
            emb = torch.cat([emb_t1, emb_t2], dim=0)
            if self.include_diff:
                diff = torch.abs(emb_t1 - emb_t2)
                emb = torch.cat([emb, diff], dim=0)
        else:
            emb = embeddings[0]

        if self.augment:
            emb, mask = self._apply_augment(emb, mask)

        return {
            "embedding_map": emb,
            "mask": mask,
            "patch_id": patch_id,
        }

    def _load_embedding(self, patch_id: str, month: str) -> torch.Tensor:
        emb_path = self.embedding_root / patch_id / f"{month}_embedding_map.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"embedding 不存在: {emb_path}")
        return torch.load(emb_path, map_location="cpu", weights_only=True)  # (D, H, W)

    def _load_mask(self, patch_id: str) -> torch.Tensor:
        for month in self.months:
            mask_path = self.mask_dir / f"{patch_id}_{month}.tif"
            if mask_path.exists():
                with rasterio.open(mask_path) as src:
                    return torch.from_numpy(src.read(1)).long()

        mask_path = self.mask_dir / f"{patch_id}.tif"
        if not mask_path.exists():
            raise FileNotFoundError(f"mask 不存在: {mask_path}")
        with rasterio.open(mask_path) as src:
            return torch.from_numpy(src.read(1)).long()

    def _apply_augment(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 同步水平翻转
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        # 同步垂直翻转
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        # 随机 90 度旋转
        if torch.rand(1).item() > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            emb = torch.rot90(emb, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        # 前景感知随机裁剪，提升低比例正样本 patch 的有效前景密度
        if self.augment and hasattr(self, "_crop_size") and self._crop_size is not None:
            emb, mask = self._random_crop(emb, mask, self._crop_size)
        return emb, mask

    @staticmethod
    def _random_crop(
        emb: torch.Tensor, mask: torch.Tensor, crop_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 emb (C, H, W) 和 mask (H, W) 做同步随机裁剪。

        若 mask 含前景，则以 70% 概率将前景像素作为裁剪中心，否则随机选中心。
        """
        _, h, w = emb.shape
        if h <= crop_size or w <= crop_size:
            return emb, mask
        fg_indices = torch.nonzero(mask > 0, as_tuple=False)
        use_fg = fg_indices.numel() > 0 and torch.rand(1).item() < 0.7
        if use_fg:
            idx = torch.randint(0, fg_indices.shape[0], (1,)).item()
            cy, cx = fg_indices[idx].tolist()
        else:
            cy = torch.randint(0, h, (1,)).item()
            cx = torch.randint(0, w, (1,)).item()

        y1 = min(max(cy - crop_size // 2, 0), h - crop_size)
        x1 = min(max(cx - crop_size // 2, 0), w - crop_size)
        y2 = y1 + crop_size
        x2 = x1 + crop_size
        return emb[:, y1:y2, x1:x2], mask[y1:y2, x1:x2]


def collate_embeddings(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embedding_map": torch.stack([b["embedding_map"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_ids": [b["patch_id"] for b in batch],
    }
