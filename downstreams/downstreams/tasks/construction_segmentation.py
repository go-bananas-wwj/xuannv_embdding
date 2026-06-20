from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from downstreams.heads.segmentation_head import build_segmentation_head
from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.tasks.base import BaseTask


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target.float())
        probs = torch.sigmoid(logits)
        pt = probs * target + (1 - probs) * (1 - target)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() + _dice_loss(logits, target)


class ConstructionSegmentationTask(BaseTask):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._loss: nn.Module | None = None

    def build_head(self) -> nn.Module:
        head_type = self.config["training"]["head_type"]
        embed_dim = self.config["data"]["embed_dim"]
        num_classes = self.config["data"]["num_classes"]
        return build_segmentation_head(
            head_type,
            embed_dim,
            num_classes,
        )

    def build_loss(self) -> nn.Module:
        if self._loss is None:
            loss_name = self.config["training"].get("loss", "focal_dice").lower()
            if loss_name == "focal_dice":
                self._loss = FocalDiceLoss()
            elif loss_name == "bce":
                pos_weight = torch.tensor(self.config["training"].get("pos_weight", 1.0))
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                raise ValueError(f"未知 loss: {loss_name}")
        return self._loss

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        model.train()
        loss_fn = self.build_loss().to(device)
        total_loss = 0.0
        for batch in loader:
            emb = batch["embedding_map"].to(device)
            mask = batch["mask"].to(device)  # (B, H, W)
            optimizer.zero_grad()
            logits = model(emb)[:, 1]  # 二分类只取前景通道
            loss = loss_fn(logits, mask.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                emb = batch["embedding_map"].to(device)
                mask = batch["mask"].to(device)
                logits = model(emb)[:, 1]
                all_logits.append(logits.cpu())
                all_masks.append(mask.cpu())
        logits = torch.cat([x.flatten() for x in all_logits])
        targets = torch.cat([x.flatten() for x in all_masks])
        return compute_segmentation_metrics(logits, targets)
