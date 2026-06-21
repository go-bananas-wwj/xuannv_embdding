from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from downstreams.heads.base import TaskHead
from downstreams.heads.segmentation_head import _init_foreground_bias


class DiffUNetHead(TaskHead):
    """轻量 2-level UNet 分割头，支持任意输入通道数，可选 diff 分支。

    当 ``use_diff=True`` 且输入通道数可被 3 整除时，将输入按通道均分为
    ``[t1, t2, diff]`` 三部分，diff 经独立卷积分支处理后与主特征相加。
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        pos_prior: float | None = None,
        use_diff: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_diff = use_diff

        # 输入投影在第一次 forward 时根据 x.shape[1] 延迟创建。
        self.in_proj: nn.Conv2d | None = None
        self.diff_branch: nn.Module | None = None

        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(hidden_dim // 4, num_classes, kernel_size=1)
        _init_foreground_bias(self.final, pos_prior)

    def _build_in_proj(self, in_channels: int) -> None:
        """根据实际输入通道数构建输入投影与可选 diff 分支。"""
        if self.in_proj is not None:
            return
        self.in_proj = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1)
        if self.use_diff and in_channels % 3 == 0:
            diff_dim = in_channels // 3
            self.diff_branch = nn.Sequential(
                nn.Conv2d(diff_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True),
            )

    def _upsample_skip(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        self._build_in_proj(x.shape[1])
        assert self.in_proj is not None

        feat = self.in_proj(x)
        feat = self.encoder(feat)

        if self.diff_branch is not None and x.shape[1] % 3 == 0:
            diff_start = 2 * (x.shape[1] // 3)
            diff = x[:, diff_start:, :, :]
            feat = feat + self.diff_branch(diff)

        x1 = self.up1(feat)  # (B, H/2, 2H, 2W)
        x1 = self.conv1(torch.cat([x1, self._upsample_skip(feat, 2)], dim=1))
        x2 = self.up2(x1)  # (B, H/4, 4H, 4W)
        x2 = self.conv2(torch.cat([x2, self._upsample_skip(feat, 4)], dim=1))
        out = self.final(x2)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
