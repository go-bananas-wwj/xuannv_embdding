from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# vMF（von Mises-Fisher）瓶颈：将特征投影到单位球面，训练时注入切空间高斯噪声。


class VMFBottleneck(nn.Module):
    """vMF 分布瓶颈层。

    使用 1x1 卷积将输入通道投影到目标维度，并通过 L2 归一化把输出约束到
    单位球面上，从而得到适合 AEF（Angular Embedding Field）的角嵌入。
    训练时会在切空间加入高斯扰动再投影回球面，起到近似 vMF 采样的正则
    效果；推理时仅做归一化，不添加噪声。
    """

    def __init__(self, in_dim: int, out_dim: int, kappa: float = 100.0) -> None:
        """初始化 VMFBottleneck。

        Args:
            in_dim: 输入通道数。
            out_dim: 输出通道数，即嵌入维度。
            kappa: vMF 集中度，越大则训练时噪声越小。
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kappa = kappa
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, in_dim, H, W)。

        Returns:
            输出张量，形状为 (B, out_dim, H, W)，每个空间位置的 L2 范数为 1。
        """
        z = self.proj(x)
        z = F.normalize(z, dim=1)

        if self.training:
            # 在切空间加入各向同性高斯噪声，再重新投影回单位球面。
            noise = torch.randn_like(z) / self.kappa
            z = F.normalize(z + noise, dim=1)

        return z
