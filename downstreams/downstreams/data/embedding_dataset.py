from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """从预生成 embedding 与 mask 加载下游任务数据。"""

    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        month: int | str = 202604,
        months: list[int | str] | None = None,
        temporal_mode: str = "single",
        augment: bool = False,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.months = [str(m) for m in (months if months is not None else [month])]
        self.temporal_mode = temporal_mode
        self.augment = augment
        self.mask_dir = self.label_root / "masks"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        mask_path = self.mask_dir / f"{patch_id}.tif"

        if not mask_path.exists():
            raise FileNotFoundError(f"mask 不存在: {mask_path}")

        emb = self._load_embedding(patch_id)
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

    def _load_month_embedding(self, patch_id: str, month: str) -> torch.Tensor:
        emb_path = self.embedding_root / patch_id / f"{month}_embedding_map.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"embedding 不存在: {emb_path}")
        return torch.load(emb_path, map_location="cpu", weights_only=True)  # (D, H, W)

    def _load_embedding(self, patch_id: str) -> torch.Tensor:
        if len(self.months) == 1 or self.temporal_mode == "single":
            return self._load_month_embedding(patch_id, self.months[0])
        if len(self.months) != 2:
            raise ValueError("非 single temporal_mode 目前仅支持两个 months")

        emb_t1 = self._load_month_embedding(patch_id, self.months[0])
        emb_t2 = self._load_month_embedding(patch_id, self.months[1])
        if emb_t1.shape != emb_t2.shape:
            raise ValueError(
                f"{patch_id} 双时相 embedding shape 不一致: {emb_t1.shape} vs {emb_t2.shape}"
            )
        if self.temporal_mode == "concat":
            return torch.cat([emb_t1, emb_t2], dim=0)
        if self.temporal_mode == "diff":
            return torch.abs(emb_t2 - emb_t1)
        if self.temporal_mode == "concat_diff":
            return torch.cat([emb_t1, emb_t2, torch.abs(emb_t2 - emb_t1)], dim=0)
        raise ValueError(f"未知 temporal_mode: {self.temporal_mode!r}")

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
