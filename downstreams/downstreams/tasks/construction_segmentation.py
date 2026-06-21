from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from downstreams.heads.segmentation_head import build_segmentation_head
from downstreams.losses.segmentation_losses import (
    BceDiceTverskyLoss,
    FocalTverskyLoss,
)
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
        if "training" not in self.config:
            raise KeyError("配置缺少 training 段")
        if "data" not in self.config:
            raise KeyError("配置缺少 data 段")
        training = self.config["training"]
        data = self.config["data"]
        if "head_type" not in training:
            raise KeyError("配置缺少 training.head_type")
        if "embed_dim" not in data:
            raise KeyError("配置缺少 data.embed_dim")
        if "num_classes" not in data:
            raise KeyError("配置缺少 data.num_classes")
        head_type = training["head_type"]
        embed_dim = data["embed_dim"]
        num_classes = data["num_classes"]
        pos_prior = training.get("pos_prior")
        months = training.get("months", ["202605"])
        if len(months) not in (1, 2):
            raise ValueError(
                f"training.months 长度必须为 1 或 2，实际为 {len(months)}: {months}"
            )
        bitemporal = len(months) == 2
        in_channels = embed_dim * (3 if bitemporal else 1)
        return build_segmentation_head(
            head_type,
            in_channels,
            num_classes,
            pos_prior=pos_prior,
        )

    def build_loss(self) -> nn.Module:
        if self._loss is None:
            if "training" not in self.config:
                raise KeyError("配置缺少 training 段")
            training = self.config["training"]
            loss_name = training.get("loss", "focal_dice").lower()
            if loss_name == "focal_dice":
                self._loss = FocalDiceLoss()
            elif loss_name == "focal_tversky":
                self._loss = FocalTverskyLoss(
                    alpha=training.get("tversky_alpha", 0.3),
                    beta=training.get("tversky_beta", 0.7),
                    gamma=training.get("focal_gamma", 1.33),
                )
            elif loss_name == "bce":
                pos_weight = torch.tensor(training.get("pos_weight", 1.0))
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif loss_name == "bce_dice_tversky":
                self._loss = BceDiceTverskyLoss(
                    pos_weight=training.get("pos_weight", 1.0),
                    tversky_alpha=training.get("tversky_alpha", 0.3),
                    tversky_beta=training.get("tversky_beta", 0.7),
                    tversky_weight=training.get("tversky_weight", 1.5),
                    bce_weight=training.get("bce_weight", 1.0),
                    dice_weight=training.get("dice_weight", 1.0),
                )
            else:
                raise ValueError(f"未知 loss 类型: {loss_name}")
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
        threshold: float | None = None,
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
        return compute_segmentation_metrics(
            logits, targets, threshold=threshold if threshold is not None else 0.5
        )
