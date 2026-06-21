from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class LinearProbeHead(TaskHead):
    """严格线性探测：单 1x1 conv，无激活/无 BN。"""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conv(embedding_map)
