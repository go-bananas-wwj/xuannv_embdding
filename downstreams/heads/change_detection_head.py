from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class ChangeDetectionHead(TaskHead):
    """双时相变化检测头（预留接口，当前仅实现 concat + 1x1 conv）。"""

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(embed_dim * 2, hidden_dim, kernel_size=1)
        self.classifier = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward_two(self, emb_t1: torch.Tensor, emb_t2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_t1, emb_t2], dim=1)
        x = torch.relu(self.fusion(x))
        return self.classifier(x)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("ChangeDetectionHead 请使用 forward_two(t1, t2)")
