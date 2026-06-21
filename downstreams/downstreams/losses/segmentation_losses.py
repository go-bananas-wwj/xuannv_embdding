from __future__ import annotations

import torch
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
