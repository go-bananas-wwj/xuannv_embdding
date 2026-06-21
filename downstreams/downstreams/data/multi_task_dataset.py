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
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.task_name = task_name
        self.months = self._normalize_months(months)
        self.augment = augment
        self.bitemporal = bitemporal
        self.include_diff = include_diff
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
        return emb, mask


def collate_embeddings(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embedding_map": torch.stack([b["embedding_map"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_ids": [b["patch_id"] for b in batch],
    }
