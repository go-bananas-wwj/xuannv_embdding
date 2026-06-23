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
    target = target.float()
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def _tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    target = target.float()
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    true_pos = (pred_flat * target_flat).sum(dim=1)
    false_pos = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    false_neg = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    alpha = 1.0 - beta
    score = (true_pos + smooth) / (
        true_pos + alpha * false_pos + beta * false_neg + smooth
    )
    return (1.0 - score).mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * target + (1 - probs) * (1 - target)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() + _dice_loss(logits, target)


class BCEDiceTverskyLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        tversky_weight: float = 1.0,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.tversky_beta = float(tversky_beta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
        )
        dice = _dice_loss(logits, target)
        tversky = _tversky_loss(logits, target, beta=self.tversky_beta)
        return bce + self.dice_weight * dice + self.tversky_weight * tversky


def _final_classifier(head: nn.Module) -> nn.Conv2d | None:
    for module in reversed(list(head.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


def _initialize_foreground_bias(
    head: nn.Module,
    pos_prior: float | None,
    num_classes: int,
) -> None:
    if pos_prior is None or num_classes != 2:
        return
    if not 0.0 < pos_prior < 1.0:
        raise ValueError(f"training.pos_prior 必须在 (0, 1) 内，收到: {pos_prior}")

    classifier = _final_classifier(head)
    if classifier is None or classifier.bias is None or classifier.out_channels < 2:
        return

    bias = torch.logit(torch.tensor(float(pos_prior))).item()
    with torch.no_grad():
        classifier.bias.zero_()
        classifier.bias[1] = bias


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
        head = build_segmentation_head(
            head_type,
            embed_dim,
            num_classes,
        )
        _initialize_foreground_bias(
            head,
            training.get("pos_prior"),
            num_classes,
        )
        return head

    def build_loss(self) -> nn.Module:
        if self._loss is None:
            if "training" not in self.config:
                raise KeyError("配置缺少 training 段")
            training = self.config["training"]
            loss_name = training.get("loss", "focal_dice").lower()
            if loss_name == "focal_dice":
                self._loss = FocalDiceLoss(
                    alpha=training.get("focal_alpha", 0.8),
                    gamma=training.get("focal_gamma", 2.0),
                    pos_weight=training.get("pos_weight", 1.0),
                )
            elif loss_name == "bce_dice_tversky":
                self._loss = BCEDiceTverskyLoss(
                    pos_weight=training.get("pos_weight", 1.0),
                    dice_weight=training.get("dice_weight", 1.0),
                    tversky_weight=training.get("tversky_weight", 1.0),
                    tversky_beta=training.get("tversky_beta", 0.7),
                )
            elif loss_name == "bce":
                pos_weight = torch.tensor(training.get("pos_weight", 1.0))
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
