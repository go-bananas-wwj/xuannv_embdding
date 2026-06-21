from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead
from downstreams.heads.segmentation_head import _init_foreground_bias


class MLPHead(TaskHead):
    """轻量 MLP 分割头：1x1 卷积堆叠，带 BatchNorm、ReLU 与 Dropout。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        pos_prior: float | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        _init_foreground_bias(self.net[-1], pos_prior)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)
