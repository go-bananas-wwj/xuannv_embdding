from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class ChangeDetectionHead(TaskHead):
    """双时相变化检测头（预留接口，当前仅实现 concat + 1x1 conv）。

    输入约定：
    - embedding_map 为两个时相 embedding map 在通道维度拼接后的张量，
      形状为 (B, 2*D, H, W)。
    - scene_emb 为两个时相 scene embedding 在通道维度拼接后的张量，
      形状为 (B, 2*D)，可选。
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim * 2, num_classes, kernel_size=1)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conv(embedding_map)


class ChangeDetectionDiffHead(TaskHead):
    """基于差分的变化检测头：先对两个时相做差，再用 1x1 conv 分类。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(
        self,
        embedding_map: torch.Tensor,
        scene_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = embedding_map.shape
        if c % 2 != 0:
            raise ValueError("ChangeDetectionDiffHead 期望 embedding_map 的通道数为 2*embed_dim")
        mid = c // 2
        diff = embedding_map[:, :mid, :, :] - embedding_map[:, mid:, :, :]
        return self.conv(diff)
