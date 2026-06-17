from __future__ import annotations

import torch
from torch import nn

# 空间/时间 Transformer 算子：在 patch 序列或时间序列上做自注意力。


class _BaseSelfAttentionBlock(nn.Module):
    """自注意力 + LayerNorm + MLP 的基础 Transformer 块。

    子类通过 ``forward`` 中的输入形状来区分空间或时间维度。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化基础自注意力块。

        Args:
            dim: 输入与输出通道数。
            num_heads: 多头注意力头数，必须能整除 ``dim``。

        Raises:
            ValueError: 当 ``dim`` 不能被 ``num_heads`` 整除时抛出。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除，"
                f"当前余数为 {dim % num_heads}。"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次标准的 Transformer 编码层前向传播。

        Args:
            x: 输入张量，形状为 (B, N, C)。

        Returns:
            输出张量，形状为 (B, N, C)。

        Raises:
            ValueError: 当输入维度或通道数不符合预期时抛出。
        """
        if x.dim() != 3:
            raise ValueError(f"自注意力输入必须是 3 维张量 (B, N, C)，当前维度为 {x.dim()}")
        batch_size, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"输入通道数 ({channels}) 与初始化维度 ({self.dim}) 不一致")
        if seq_len == 0:
            # 空序列无 token 可处理，直接返回，避免进入后端触发未定义行为。
            return x

        # MultiheadAttention 的 CPU backend 在输入非连续内存时可能触发段错误，
        # 因此在进入注意力前强制 contiguous。
        x = x.contiguous()
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


class SpaceOperator(_BaseSelfAttentionBlock):
    """空间自注意力算子。

    对展平后的空间 patch 序列做自注意力，用于捕获同一时刻图像内的
    长距离空间依赖。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 SpaceOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 多头注意力头数，必须能整除 ``dim``。
        """
        super().__init__(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, H*W, C)。

        Returns:
            输出张量，形状为 (B, H*W, C)。
        """
        return self._transform(x)


class TimeOperator(_BaseSelfAttentionBlock):
    """时间自注意力算子。

    对时间序列做自注意力，用于捕获同一空间位置在不同时刻之间的
    时间依赖。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 TimeOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 多头注意力头数，必须能整除 ``dim``。
        """
        super().__init__(dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, T, C)。

        Returns:
            输出张量，形状为 (B, T, C)。
        """
        return self._transform(x)
