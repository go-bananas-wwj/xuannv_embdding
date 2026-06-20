from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

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


class SinusoidalTimeEncoding(nn.Module):
    """正弦/余弦时间编码。

    将标量时间戳（毫秒、天或任意连续单位）编码为 ``dim`` 维正弦嵌入，
    用于 ``STPTimeOperator`` 与 ``SummaryPeriodEncoder`` 提供时间先验。
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        """初始化 SinusoidalTimeEncoding。

        Args:
            dim: 输出编码维度。
            max_period: 正弦周期上限，默认 10000。
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """对时间戳进行正弦编码。

        Args:
            timestamps: 形状为 ``(B, T)`` 或 ``(B,)`` 的标量时间戳。

        Returns:
            编码结果，形状为 ``(B, T, dim)``；当输入为 ``(B,)`` 且内部 T=1 时，
            返回 ``(B, dim)``。
        """
        half_dim = self.dim // 2
        # 频率序列：(half_dim,)
        freq = torch.exp(
            torch.arange(half_dim, device=timestamps.device, dtype=torch.float32)
            * (math.log(self.max_period) / max(half_dim - 1, 1))
        )

        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(1)  # (B, 1)

        t = timestamps.unsqueeze(-1).float()  # (B, T, 1)
        f = freq.view(1, 1, -1)  # (1, 1, half_dim)
        sin_emb = torch.sin(t * f)
        cos_emb = torch.cos(t * f)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, T, dim)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        if timestamps.shape[1] == 1:
            emb = emb.squeeze(1)  # (B, dim)

        return emb


class SummaryPeriodEncoder(nn.Module):
    """时序摘要查询构造器。

    将有效时间区间 ``[t_s, t_e)`` 及其长度编码为一个查询向量，
    供 ``TimePooling`` 在全局范围内聚合时间序列。
    """

    def __init__(self, dim: int) -> None:
        """初始化 SummaryPeriodEncoder。

        Args:
            dim: 输出查询维度，必须为偶数。
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) 必须为偶数")
        self.dim = dim
        self.time_enc = SinusoidalTimeEncoding(self.dim // 2)
        self.fuse = nn.Sequential(
            nn.Linear(3 * (self.dim // 2), dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.q_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, valid_period: torch.Tensor) -> torch.Tensor:
        """构造时间摘要查询。

        Args:
            valid_period: 形状为 ``(B, 2)``，每行为 ``[t_s, t_e)``。

        Returns:
            查询向量，形状为 ``(B, dim)``。
        """
        if valid_period.dim() != 2 or valid_period.size(1) != 2:
            raise ValueError("valid_period 必须是形状为 (B, 2) 的张量")
        t_s = valid_period[:, 0]
        t_e = valid_period[:, 1]
        dur = t_e - t_s
        enc_s = self.time_enc(t_s)  # (B, dim//2)
        enc_e = self.time_enc(t_e)  # (B, dim//2)
        enc_d = self.time_enc(dur)  # (B, dim//2)
        q = torch.cat([enc_s, enc_e, enc_d], dim=-1)
        q = self.fuse(q) + self.q_bias
        return q


class TimePooling(nn.Module):
    """单查询多头时间注意力池化。

    对每个空间位置，使用单个查询在有效时间步上做缩放点积注意力，
    将 ``(B, T, H, W, C)`` 压缩为 ``(B, H, W, C)``。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 TimePooling。

        Args:
            dim: 输入与输出通道数。
            num_heads: 注意力头数，必须能整除 ``dim``。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.kv = nn.Linear(dim, 2 * dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        feats: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """时间注意力池化。

        Args:
            feats: 时序特征，形状 ``(B, T, H, W, C)``。
            q: 查询向量，形状 ``(B, C)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``；0 表示被掩码。

        Returns:
            池化后的特征，形状 ``(B, H, W, C)``。
        """
        B, T, H, W, C = feats.shape
        BHW = B * H * W

        x = feats.view(BHW, T, C)
        kv = self.kv(x).view(BHW, T, 2, self.num_heads, self.head_dim)
        K = kv[:, :, 0].permute(0, 2, 1, 3)  # (BHW, heads, T, d)
        V = kv[:, :, 1].permute(0, 2, 1, 3)  # (BHW, heads, T, d)

        qh = self.q_proj(q).view(B, self.num_heads, self.head_dim)
        qh = qh.unsqueeze(1).expand(B, H * W, self.num_heads, self.head_dim).reshape(
            BHW, self.num_heads, 1, self.head_dim
        )

        logits = (qh * K).sum(dim=-1) / (self.head_dim ** 0.5)  # (BHW, heads, T)

        if mask is not None:
            mask_flat = mask.view(B, 1, 1, T).expand(B, H * W, self.num_heads, T).reshape(
                BHW, self.num_heads, T
            )
            logits = logits.masked_fill(mask_flat == 0, float("-inf"))
            # 对于全被掩码的样本，softmax 会得到 NaN；预先将这些行置 0，
            # 使 softmax 输出均匀分布，再在外部乘 0 得到真正的零输出。
            row_sum = mask.sum(dim=-1)  # (B,)
            empty_row = (row_sum == 0).view(B, 1, 1).expand(B, H * W, self.num_heads)
            empty_row = empty_row.reshape(BHW, self.num_heads, 1)
            logits = torch.where(empty_row, torch.zeros_like(logits), logits)

        attn = F.softmax(logits, dim=-1)

        if mask is not None:
            row_valid = mask.sum(dim=-1) > 0  # (B,)
            row_valid = row_valid.view(B, 1).expand(B, H * W).reshape(BHW, 1, 1)
            attn = attn * row_valid

        # 用 matmul 替换 einsum，降低在 NPU 等后端上的兼容性风险。
        out = torch.matmul(attn.unsqueeze(-2), V).squeeze(-2)  # (BHW, heads, d)
        out = out.reshape(BHW, C)
        z = self.out(out).view(B, H, W, C)
        return z


class STPSpaceOperator(nn.Module):
    """STP 空间算子（1/16L 分辨率）。

    将每帧的 ``H*W`` 视为序列长度，做预归一化多头自注意力 + MLP，
    用于捕获全局空间依赖。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 STPSpaceOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 注意力头数，必须能整除 ``dim``。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        x_flat = x.reshape(B * T, H * W, C)
        residual = x_flat

        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).view(B * T, H * W, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (BT, heads, HW, d)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # (BT, heads, HW, d)
        out = out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.view(B, T, H, W, C)


class STPTimeOperator(nn.Module):
    """STP 时间算子（1/8L 分辨率）。

    为每个空间位置独立做时间自注意力，并加入正弦时间编码。
    支持通过 ``mask`` 屏蔽无效时间步。
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """初始化 STPTimeOperator。

        Args:
            dim: 输入与输出通道数。
            num_heads: 注意力头数，必须能整除 ``dim``。
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.time_encoding = SinusoidalTimeEncoding(dim)

    def _pad_or_trim(self, tensor: torch.Tensor, target_len: int, dim: int = 1) -> torch.Tensor:
        """将张量沿指定维度补齐或截断到目标长度（复制最后一个值）。"""
        cur_len = tensor.shape[dim]
        if cur_len == target_len:
            return tensor
        if cur_len > target_len:
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(None, target_len)
            return tensor[tuple(slices)]
        repeat_times = target_len - cur_len
        last = tensor.index_select(dim, torch.tensor([cur_len - 1], device=tensor.device))
        repeats = [1] * tensor.dim()
        repeats[dim] = repeat_times
        padding = last.repeat(*repeats)
        return torch.cat([tensor, padding], dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。
            timestamps: 时间戳，形状 ``(B, T)`` 或 ``(B,)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        if timestamps.dim() == 1:
            timestamps = timestamps.view(B, T)
        timestamps = self._pad_or_trim(timestamps, T, dim=1)

        time_enc = self.time_encoding(timestamps)  # (B, T, C) 或 (B, C)
        if time_enc.dim() == 2:
            time_enc = time_enc.unsqueeze(1).expand(-1, T, -1)
        else:
            time_enc = self._pad_or_trim(time_enc, T, dim=1)
        time_enc = time_enc.unsqueeze(2).unsqueeze(3)
        x = x + time_enc

        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        residual = x_flat

        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).view(B * H * W, T, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (BHW, heads, T, d)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            # logits 形状为 (BHW, heads, T_query, T_key)，掩码作用在 key 维度上。
            mask_flat = mask.view(B, 1, T).expand(-1, H * W, -1)
            mask_flat = mask_flat.reshape(B * H * W, 1, T).unsqueeze(2)  # (BHW, 1, 1, T)
            logits = logits.masked_fill(mask_flat == 0, float("-inf"))
            # 防止全被掩码的样本产生 NaN：将这些行全部置 0 后再 softmax。
            row_sum = mask.sum(dim=-1)  # (B,)
            empty_row = (row_sum == 0).view(B, 1, 1, 1).expand(B, H * W, self.num_heads, T)
            empty_row = empty_row.reshape(B * H * W, self.num_heads, 1, T)
            logits = torch.where(empty_row, torch.zeros_like(logits), logits)
        attn = F.softmax(logits, dim=-1)

        if mask is not None:
            row_valid = mask.sum(dim=-1) > 0  # (B,)
            row_valid = row_valid.view(B, 1).expand(B, H * W).reshape(B * H * W, 1, 1)
            attn = attn * row_valid.unsqueeze(-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        x_flat = residual + self.proj(out)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.view(B, H, W, T, C).permute(0, 3, 1, 2, 4)


class STPPrecisionOperator(nn.Module):
    """STP 精度算子（1/2L 分辨率）。

    使用两组 3x3 卷积 + GroupNorm + GELU 在局部空间上做精化，
    保留高分辨率细节。
    """

    def __init__(self, dim: int) -> None:
        """初始化 STPPrecisionOperator。

        Args:
            dim: 输入与输出通道数。
        """
        super().__init__()
        self.dim = dim
        num_groups1 = 8 if dim % 8 == 0 else dim
        num_groups2 = 8 if (dim * 4) % 8 == 0 else dim * 4
        self.norm1 = nn.GroupNorm(num_groups1, dim)
        self.norm2 = nn.GroupNorm(num_groups2, dim * 4)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, C)``。

        Returns:
            输出张量，形状 ``(B, T, H, W, C)``。
        """
        B, T, H, W, C = x.shape
        x_conv = x.reshape(B * T, C, H, W)
        residual = x_conv

        x_conv = self.conv1(self.norm1(x_conv))
        x_conv = F.gelu(x_conv)
        x_conv = self.conv2(self.norm2(x_conv))
        x_conv = residual + x_conv

        return x_conv.view(B, T, H, W, C)


class LearnedSpatialResampling(nn.Module):
    """可学习的空间重采样层。

    仅对 2× 上采样是真正可学习的 ``ConvTranspose2d``。当 ``scale_factor > 1``
    且不是 2 的整数倍时，卷积输出尺寸与目标不一致，调用方会通过
    ``F.interpolate`` 兜底；下采样与 1× 采样使用普通 ``Conv2d`` 实现。
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float) -> None:
        """初始化 LearnedSpatialResampling。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            scale_factor: 空间缩放因子；大于 1 为上采样，小于 1 为下采样。
        """
        super().__init__()
        self.scale_factor = scale_factor
        if scale_factor > 1:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        elif scale_factor < 1:
            stride = int(1.0 / scale_factor)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=stride * 2 - 1,
                stride=stride,
                padding=stride - 1,
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        """可学习重采样。

        Args:
            x: 输入张量，形状 ``(N, C, H, W)``。
            target_size: 可选的目标空间尺寸 ``(H_target, W_target)``；
                当卷积输出尺寸不一致时，使用双线性插值兜底。

        Returns:
            输出张量，形状 ``(N, out_channels, H_target, W_target)``。
        """
        out = self.conv(x)
        if target_size is not None and out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class MultiResolutionSTPBlock(nn.Module):
    """多分辨率 STP 块。

    在 1/16L 空间路径、1/8L 时间路径、1/2L 精度路径上分别执行对应算子，
    并通过六个跨尺度交换分支实现信息融合。
    """

    def __init__(
        self,
        space_dim: int,
        time_dim: int,
        precision_dim: int,
        num_heads: int = 8,
    ) -> None:
        """初始化 MultiResolutionSTPBlock。

        Args:
            space_dim: 空间路径通道数。
            time_dim: 时间路径通道数。
            precision_dim: 精度路径通道数。
            num_heads: 注意力算子头数。
        """
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim

        self.space_op = STPSpaceOperator(space_dim, num_heads)
        self.time_op = STPTimeOperator(time_dim, num_heads)
        self.precision_op = STPPrecisionOperator(precision_dim)

        self.space_to_time = LearnedSpatialResampling(space_dim, time_dim, 2.0)
        self.space_to_precision = LearnedSpatialResampling(space_dim, precision_dim, 8.0)
        self.time_to_space = LearnedSpatialResampling(time_dim, space_dim, 0.5)
        self.time_to_precision = LearnedSpatialResampling(time_dim, precision_dim, 4.0)
        self.precision_to_space = LearnedSpatialResampling(precision_dim, space_dim, 0.125)
        self.precision_to_time = LearnedSpatialResampling(precision_dim, time_dim, 0.25)

    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            space_x: 空间路径输入，形状 ``(B, T, Hs, Ws, space_dim)``。
            time_x: 时间路径输入，形状 ``(B, T, Ht, Wt, time_dim)``。
            precision_x: 精度路径输入，形状 ``(B, T, Hp, Wp, precision_dim)``。
            timestamps: 时间戳，形状 ``(B, T)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            三个路径的输出，形状分别与输入一致。
        """
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps, mask=mask)
        precision_out = self.precision_op(precision_x)

        B, T = space_out.shape[:2]
        space_H, space_W = space_out.shape[2:4]
        time_H, time_W = time_out.shape[2:4]
        precision_H, precision_W = precision_out.shape[2:4]

        space_2d = space_out.reshape(B * T, self.space_dim, space_H, space_W)
        time_2d = time_out.reshape(B * T, self.time_dim, time_H, time_W)
        precision_2d = precision_out.reshape(B * T, self.precision_dim, precision_H, precision_W)

        time_to_space = self.time_to_space(time_2d, target_size=(space_H, space_W))
        precision_to_space = self.precision_to_space(precision_2d, target_size=(space_H, space_W))
        space_exchange = space_2d + time_to_space + precision_to_space

        space_to_time = self.space_to_time(space_2d, target_size=(time_H, time_W))
        precision_to_time = self.precision_to_time(precision_2d, target_size=(time_H, time_W))
        time_exchange = time_2d + space_to_time + precision_to_time

        space_to_precision = self.space_to_precision(
            space_2d, target_size=(precision_H, precision_W)
        )
        time_to_precision = self.time_to_precision(
            time_2d, target_size=(precision_H, precision_W)
        )
        precision_exchange = precision_2d + space_to_precision + time_to_precision

        space_out = space_exchange.view(B, T, self.space_dim, space_H, space_W).permute(
            0, 1, 3, 4, 2
        )
        time_out = time_exchange.view(B, T, self.time_dim, time_H, time_W).permute(
            0, 1, 3, 4, 2
        )
        precision_out = precision_exchange.view(
            B, T, self.precision_dim, precision_H, precision_W
        ).permute(0, 1, 3, 4, 2)

        return space_out, time_out, precision_out


class STPEncoder(nn.Module):
    """Space-Time-Precision 多分辨率编码器。

    将时序多源特征投影到三个分辨率路径，依次通过若干 ``MultiResolutionSTPBlock``，
    最终将所有路径对齐到 1/2L 精度分辨率并相加，返回特征与原始输入空间尺寸。
    """

    # 各路径相对于输入的空间缩放倍数。
    SPACE_SCALE = 16
    TIME_SCALE = 8
    PRECISION_SCALE = 2

    def __init__(
        self,
        input_channels: int,
        space_dim: int = 512,
        time_dim: int = 256,
        precision_dim: int = 128,
        num_blocks: int = 6,
        num_heads: int = 8,
        gradient_checkpointing: bool = False,
    ) -> None:
        """初始化 STPEncoder。

        Args:
            input_channels: 拼接后的输入通道数。
            space_dim: 空间路径通道数。
            time_dim: 时间路径通道数。
            precision_dim: 精度路径通道数。
            num_blocks: STP 块数量。
            num_heads: 注意力头数。
            gradient_checkpointing: 是否启用梯度检查点以节省显存。
        """
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.input_projection = nn.Linear(input_channels, precision_dim)
        self.space_projection = nn.Linear(precision_dim, space_dim)
        self.time_projection = nn.Linear(precision_dim, time_dim)

        self.blocks = nn.ModuleList(
            [
                MultiResolutionSTPBlock(space_dim, time_dim, precision_dim, num_heads)
                for _ in range(num_blocks)
            ]
        )

        # 最终重采样到精度路径分辨率；仅 2× 上采样是可学习的，其余靠插值兜底。
        self.final_space_resample = LearnedSpatialResampling(
            space_dim, precision_dim, float(self.SPACE_SCALE // self.PRECISION_SCALE)
        )
        self.final_time_resample = LearnedSpatialResampling(
            time_dim, precision_dim, float(self.TIME_SCALE // self.PRECISION_SCALE)
        )
        self.norm = nn.LayerNorm(precision_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """前向传播。

        要求输入空间尺寸至少为 ``SPACE_SCALE``（默认 16），否则无法产生
        1/16L 空间路径特征。

        Args:
            x: 输入张量，形状 ``(B, T, H, W, input_channels)``。
            timestamps: 时间戳，形状 ``(B, T)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            (features, input_size) 元组：
            - features: ``(B, T, H//2, W//2, precision_dim)``
            - input_size: ``(H, W)``，用于后续上采样对齐。
        """
        B, T, H, W, C = x.shape
        if H < self.SPACE_SCALE or W < self.SPACE_SCALE:
            raise ValueError(
                f"STPEncoder requires input height/width >= {self.SPACE_SCALE}, got {H}x{W}"
            )
        input_size = (H, W)

        x_proj = self.input_projection(x)

        # 空间路径：投影到 space_dim 并下采样到 1/16L
        space_features = self.space_projection(x_proj)
        space_features = space_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.space_dim, H, W
        )
        space_features = F.adaptive_avg_pool2d(
            space_features, (H // self.SPACE_SCALE, W // self.SPACE_SCALE)
        )
        space_features = space_features.view(
            B, T, self.space_dim, H // self.SPACE_SCALE, W // self.SPACE_SCALE
        ).permute(0, 1, 3, 4, 2)

        # 时间路径：投影到 time_dim 并下采样到 1/8L
        time_features = self.time_projection(x_proj)
        time_features = time_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.time_dim, H, W
        )
        time_features = F.adaptive_avg_pool2d(
            time_features, (H // self.TIME_SCALE, W // self.TIME_SCALE)
        )
        time_features = time_features.view(
            B, T, self.time_dim, H // self.TIME_SCALE, W // self.TIME_SCALE
        ).permute(0, 1, 3, 4, 2)

        # 精度路径：保持 precision_dim，下采样到 1/2L
        precision_features = x_proj.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.precision_dim, H, W
        )
        precision_features = F.adaptive_avg_pool2d(
            precision_features, (H // self.PRECISION_SCALE, W // self.PRECISION_SCALE)
        )
        precision_features = precision_features.view(
            B, T, self.precision_dim, H // self.PRECISION_SCALE, W // self.PRECISION_SCALE
        ).permute(0, 1, 3, 4, 2)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                try:
                    space_features, time_features, precision_features = checkpoint.checkpoint(
                        block,
                        space_features,
                        time_features,
                        precision_features,
                        timestamps,
                        mask,
                        use_reentrant=False,
                    )
                except TypeError:
                    # 旧版 PyTorch 不支持 use_reentrant 参数，回退默认行为。
                    space_features, time_features, precision_features = checkpoint.checkpoint(
                        block,
                        space_features,
                        time_features,
                        precision_features,
                        timestamps,
                        mask,
                    )
            else:
                space_features, time_features, precision_features = block(
                    space_features, time_features, precision_features, timestamps, mask=mask
                )

        # 将各路径 reshape 为 (BT, C, H, W) 以便重采样
        space_2d = space_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.space_dim, H // self.SPACE_SCALE, W // self.SPACE_SCALE
        )
        time_2d = time_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.time_dim, H // self.TIME_SCALE, W // self.TIME_SCALE
        )
        precision_2d = precision_features.permute(0, 1, 4, 2, 3).reshape(
            B * T, self.precision_dim, H // self.PRECISION_SCALE, W // self.PRECISION_SCALE
        )

        target_size = (H // self.PRECISION_SCALE, W // self.PRECISION_SCALE)
        space_resampled = self.final_space_resample(space_2d, target_size=target_size)
        time_resampled = self.final_time_resample(time_2d, target_size=target_size)

        final_features = space_resampled + time_resampled + precision_2d
        final_features = final_features.view(
            B, T, self.precision_dim, H // self.PRECISION_SCALE, W // self.PRECISION_SCALE
        ).permute(0, 1, 3, 4, 2)

        return self.norm(final_features), input_size


class TemporalSummarizer(nn.Module):
    """时序摘要器。

    根据有效时间区间构造查询，通过 ``TimePooling`` 将 ``(B, T, H, W, C)``
    压缩为 ``(B, H, W, embed_dim)`` 的单位向量。
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, num_heads: int = 8) -> None:
        """初始化 TemporalSummarizer。

        Args:
            feature_dim: 输入特征通道数。
            embed_dim: 输出嵌入维度。
            num_heads: 时间注意力池化头数。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.summarizer_q = SummaryPeriodEncoder(feature_dim)
        self.time_pool = TimePooling(feature_dim, num_heads=num_heads)
        self.proj = nn.Linear(feature_dim, embed_dim, bias=False)

    def forward(
        self,
        feats: torch.Tensor,
        timestamps: torch.Tensor,
        valid_periods: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            feats: 输入特征，形状 ``(B, T, H, W, C)``。
            timestamps: 时间戳，形状 ``(B, T)``。
            valid_periods: 有效时间区间，形状 ``(B, 2)``。
            mask: 可选时间有效掩码，形状 ``(B, T)``。

        Returns:
            单位向量嵌入，形状 ``(B, H, W, embed_dim)``。
        """
        q = self.summarizer_q(valid_periods)
        z = self.time_pool(feats, q, mask=mask)
        mu = self.proj(z)
        return F.normalize(mu, p=2, dim=-1)


class EmbeddingUpsampleHead(nn.Module):
    """嵌入上采样头。

    将 1/2L 分辨率的嵌入特征通过转置卷积上采样回原始输入分辨率，
    并使用 1x1 卷积做最后的精化。
    """

    def __init__(self, in_dim: int, out_dim: int | None = None) -> None:
        """初始化 EmbeddingUpsampleHead。

        Args:
            in_dim: 输入通道数。
            out_dim: 输出通道数，默认等于 ``in_dim``。
        """
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        self.out_dim = out_dim
        num_groups = 8 if out_dim % 8 == 0 else out_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, H, W, C)``。
            target_size: 可选的目标空间尺寸 ``(H_target, W_target)``。当输入的
                高度或宽度为奇数时，转置卷积输出可能与原始尺寸不一致，此时通过
                ``F.interpolate`` 强制对齐。若未提供，调用方需自行保证输入为偶数。

        Returns:
            输出张量，形状 ``(B, H_target, W_target, out_dim)``（未提供 target_size
            时默认为 ``(2H, 2W)``）。
        """
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.net(x)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)


class MonthlyEmbeddingModule(nn.Module):
    """月度嵌入模块。

    将时序特征按 ``YYYYMM`` 时间戳分配到固定月度 bin，对每个 bin 内的有效
    观测做空间位置级别的加权平均；无观测的月份/位置使用可学习 ``missing_token``
    填充，保证输出形状固定且不会出现 NaN。
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_months: int,
        ref_year: int = 2025,
        ref_month: int = 1,
        missing_token_init: float = 0.02,
    ) -> None:
        """初始化 MonthlyEmbeddingModule。

        Args:
            in_channels: 输入特征通道数（STP 精度路径维度）。
            embed_dim: 输出月度嵌入维度。
            num_months: 固定月度 bin 数量。
            ref_year: 月度 bin 起始年份。
            ref_month: 月度 bin 起始月份。
            missing_token_init: ``missing_token`` 的初始值。
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_months = num_months
        self.ref_year = ref_year
        self.ref_month = ref_month

        self.proj = nn.Linear(in_channels, embed_dim)
        self.missing_token = nn.Parameter(
            torch.full((1, 1, 1, 1, embed_dim), missing_token_init)
        )

    def _yyyymm_to_index(self, timestamps: torch.Tensor) -> torch.Tensor:
        """将 ``YYYYMM`` 整数时间戳映射到以 ``ref_year/ref_month`` 为 0 的月度索引。

        Args:
            timestamps: 形状 ``(B, T)`` 的整数时间戳。

        Returns:
            形状 ``(B, T)`` 的月度索引，越界值为负数或大于等于 ``num_months``。
        """
        years = timestamps // 100
        months = timestamps % 100
        return (years - self.ref_year) * 12 + (months - self.ref_month)

    def forward(
        self,
        feats: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            feats: 输入特征，形状 ``(B, T_obs, H, W, C)``。
            timestamps: 时间戳，形状 ``(B, T_obs)``，应为 ``YYYYMM`` 整数格式。
            mask: 可选时间有效掩码，形状 ``(B, T_obs)``；为 None 时视为全 1。

        Returns:
            ``(monthly_feats, monthly_mask)`` 元组：
            - monthly_feats: ``(B, num_months, H, W, embed_dim)``。
            - monthly_mask: ``(B, num_months)``，1 表示该月份至少有一个有效观测。
        """
        B, T, H, W, C = feats.shape
        M = self.num_months
        device = feats.device

        z = self.proj(feats)  # (B, T, H, W, embed_dim)

        month_index = self._yyyymm_to_index(timestamps)  # (B, T)
        in_range = (month_index >= 0) & (month_index < M)

        if mask is None:
            valid = in_range
        else:
            valid = mask.bool() & in_range

        # 构建 scatter_add 所需的线性索引。
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, T, H, W).reshape(-1)
        m_idx = month_index.view(B, T, 1, 1).expand(B, T, H, W).reshape(-1)
        h_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, T, H, W).reshape(-1)
        w_idx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, T, H, W).reshape(-1)

        valid_flat = valid.view(B, T, 1, 1).expand(B, T, H, W).reshape(-1)
        flat_index = ((b_idx * M + m_idx) * H + h_idx) * W + w_idx

        z_flat = z.reshape(B * T * H * W, self.embed_dim)

        acc = torch.zeros(B * M * H * W, self.embed_dim, device=device, dtype=z.dtype)
        acc.scatter_add_(
            0,
            flat_index[valid_flat].unsqueeze(-1).expand(-1, self.embed_dim),
            z_flat[valid_flat],
        )

        count = torch.zeros(B * M * H * W, device=device, dtype=z.dtype)
        count.scatter_add_(
            0,
            flat_index[valid_flat],
            torch.ones(valid_flat.sum(), device=device, dtype=z.dtype),
        )

        acc = acc.view(B, M, H, W, self.embed_dim)
        count = count.view(B, M, H, W)

        count_safe = count.clamp(min=1.0).unsqueeze(-1)
        avg = acc / count_safe
        has_obs = (count > 0).float().unsqueeze(-1)
        monthly_feats = has_obs * avg + (1 - has_obs) * self.missing_token

        monthly_mask = (count.view(B, M, H * W).sum(-1) > 0).float()
        return monthly_feats, monthly_mask
