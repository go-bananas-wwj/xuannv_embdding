from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class TaskHead(nn.Module):
    @abstractmethod
    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_emb: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
