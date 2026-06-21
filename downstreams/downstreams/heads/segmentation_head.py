from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from downstreams.heads.base import TaskHead
from downstreams.heads.linear_probe import LinearProbeHead


def _init_foreground_bias(module: nn.Module, pos_prior: float | None, fg_index: int = 1) -> None:
    """根据前景先验初始化二分类输出层的 foreground bias。"""
    if pos_prior is None or pos_prior <= 0 or pos_prior >= 1:
        return
    if hasattr(module, "bias") and module.bias is not None:
        with torch.no_grad():
            module.bias[fg_index].fill_(float(np.log(pos_prior / (1.0 - pos_prior))))


class FCNHead(TaskHead):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        pos_prior: float | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        _init_foreground_bias(self.conv2, pos_prior)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


class UNetHead(TaskHead):
    """轻量 UNet decoder，只有两层，skip 来自输入本身。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pos_prior: float | None = None,
    ) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        _init_foreground_bias(self.final, pos_prior)

    def _upsample_skip(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, D, H, W)
        x1 = self.up1(x)  # (B, D/2, 2H, 2W)
        x1 = self.conv1(torch.cat([x1, self._upsample_skip(x, 2)], dim=1))
        x2 = self.up2(x1)  # (B, D/4, 4H, 4W)
        x2 = self.conv2(torch.cat([x2, self._upsample_skip(x, 4)], dim=1))
        out = self.final(x2)  # (B, C, 4H, 4W)
        # 下采样回原始尺寸
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class UperNetHead(TaskHead):
    """简化版 UperNet：PSP module + fusion conv。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        pos_prior: float | None = None,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True),
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        _init_foreground_bias(self.classifier, pos_prior)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        feats = [x]
        for module in self.psp_modules:
            pooled = module(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(pooled)
        fused = torch.cat(feats, dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


def build_segmentation_head(
    head_type: str,
    in_channels: int,
    num_classes: int,
    pos_prior: float | None = None,
) -> TaskHead:
    head_type = head_type.lower()
    if head_type == "linear" or head_type == "linear_probe":
        return LinearProbeHead(in_channels, num_classes)
    if head_type == "fcn":
        return FCNHead(in_channels, num_classes, pos_prior=pos_prior)
    if head_type == "unet":
        return UNetHead(in_channels, num_classes, pos_prior=pos_prior)
    if head_type == "upernet":
        return UperNetHead(in_channels, num_classes, pos_prior=pos_prior)
    if head_type == "mlp":
        from downstreams.heads.mlp_head import MLPHead

        return MLPHead(in_channels, num_classes, pos_prior=pos_prior)
    if head_type == "diff_unet":
        from downstreams.heads.diff_head import DiffUNetHead

        return DiffUNetHead(in_channels, num_classes, pos_prior=pos_prior)
    raise ValueError(f"未知 head 类型: {head_type}")
