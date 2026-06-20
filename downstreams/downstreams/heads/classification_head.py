from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class ClassificationHead(TaskHead):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embedding_map: torch.Tensor, scene_emb: torch.Tensor) -> torch.Tensor:
        if scene_emb is None:
            raise ValueError("ClassificationHead 需要 scene_emb")
        return self.mlp(scene_emb)
