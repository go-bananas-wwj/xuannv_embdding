from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# vMF（von Mises-Fisher）瓶颈：将特征投影到单位球面，训练时注入切空间高斯噪声。


class _KappaDefault:
    """用于区分 "未传入 kappa" 与 "显式传入 kappa" 的占位符。"""

    pass


_KAPPA_DEFAULT = _KappaDefault()


class VMFBottleneck(nn.Module):
    """vMF 分布瓶颈层。

    使用 1x1 卷积将输入通道投影到目标维度，并通过 L2 归一化把输出约束到
    单位球面上，从而得到适合 AEF（Angular Embedding Field）的角嵌入。
    训练时会在切空间加入高斯扰动再投影回球面，起到近似 vMF 采样的正则
    效果；推理时仅做归一化，不添加噪声。

    ``log_kappa`` 是可学习标量，训练时控制切空间噪声的标准差
    ``1 / exp(log_kappa)``。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kappa: float | _KappaDefault = _KAPPA_DEFAULT,
    ) -> None:
        """初始化 VMFBottleneck。

        Args:
            in_dim: 输入通道数。
            out_dim: 输出通道数，即嵌入维度。
            kappa: 可选的固定 vMF 集中度。若显式传入，则 ``log_kappa``
                初始化为 ``log(kappa)``；否则默认初始化为 ``log(10.0)``。
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        init_kappa = 10.0 if isinstance(kappa, _KappaDefault) else float(kappa)
        self.log_kappa = nn.Parameter(torch.log(torch.tensor(init_kappa)))
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
            # 在切空间加入各向同性高斯噪声，标准差为 1 / exp(log_kappa)。
            noise = torch.randn_like(z) / torch.exp(self.log_kappa)
            z = F.normalize(z + noise, dim=1)

        return z
