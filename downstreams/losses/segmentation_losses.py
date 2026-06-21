from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for imbalanced binary segmentation.

    Parameters
    ----------
    alpha
        Weight for false positives.
    beta
        Weight for false negatives.
    gamma
        Focal exponent that down-weights easy examples.
    smooth
        Smoothing constant to avoid division by zero.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky) ** self.gamma


class TverskyLoss(nn.Module):
    """Tversky loss for imbalanced binary segmentation.

    Parameters
    ----------
    alpha
        Weight for false positives.
    beta
        Weight for false negatives.
    smooth
        Smoothing constant to avoid division by zero.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class BceDiceTverskyLoss(nn.Module):
    """Combined BCE + Dice + Tversky loss for binary segmentation.

    The three terms are weighted by ``bce_weight``, ``dice_weight`` and
    ``tversky_weight``. Default weights follow the benchmark recipe:
    BCE=1.0, Dice=1.0, Tversky=1.5.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_weight: float = 1.5,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.bce = nn.BCEWithLogitsLoss()
        self._pos_weight_value = pos_weight
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = TverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, smooth=smooth
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = self.bce(logits, target)
        if self._pos_weight_value != 1.0:
            # 在 forward 中动态应用正样本权重，避免 pos_weight tensor 设备不匹配
            pos_weight = torch.tensor(
                self._pos_weight_value, device=logits.device, dtype=logits.dtype
            )
            # log(1 + exp(-logits)) = -log_sigmoid(logits)
            bce = bce + ((pos_weight - 1.0) * target * (-F.logsigmoid(logits))).mean()
        dice = self.dice(logits, target)
        tversky = self.tversky(logits, target)
        return (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.tversky_weight * tversky
        )
