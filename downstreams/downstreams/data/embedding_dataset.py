from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import rasterio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """从预生成 embedding 与 mask 加载下游任务数据。"""

    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        month: int | str = 202604,
        augment: bool = False,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.month = str(month)
        self.augment = augment
        self.mask_dir = self.label_root / "masks"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        emb_path = self.embedding_root / patch_id / f"{self.month}_embedding_map.pt"
        mask_path = self.mask_dir / f"{patch_id}.tif"

        if not emb_path.exists():
            raise FileNotFoundError(f"embedding 不存在: {emb_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"mask 不存在: {mask_path}")

        emb = torch.load(emb_path, map_location="cpu", weights_only=True)  # (D, H, W)
        with rasterio.open(mask_path) as src:
            mask = torch.from_numpy(src.read(1)).long()  # (H, W)

        if emb.shape[-2:] != mask.shape:
            raise ValueError(
                f"{patch_id} embedding {emb.shape[-2:]} 与 mask {mask.shape} 尺寸不一致"
            )

        if self.augment:
            emb, mask = self._apply_augment(emb, mask)

        return {
            "embedding_map": emb,
            "mask": mask,
            "patch_id": patch_id,
        }

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
