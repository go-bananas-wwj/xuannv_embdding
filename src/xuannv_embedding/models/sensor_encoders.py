from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# 多源遥感数据 stem encoder：为每个数据源提供独立的浅层卷积编码器。


class SensorEncoder(nn.Module):
    """单个数据源的浅层卷积 stem。

    结构为可选的下采样卷积 + Conv2d -> GroupNorm -> ReLU，用于把多波段输入
    投影到统一的中间通道数，并在需要时降低空间分辨率，便于后续空间/时间
    融合模块处理。
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int = 1) -> None:
        """初始化 SensorEncoder。

        Args:
            in_channels: 输入数据的波段数，例如 S2=13、S1=2。
            out_channels: 输出通道数，即投影后的统一通道数。
            spatial_stride: 输入空间分辨率下采样倍数，默认 1 保持尺寸。
        """
        super().__init__()
        self.spatial_stride = spatial_stride
        # GroupNorm 要求 num_channels 能被 num_groups 整除；
        # 当 out_channels 小于 8 时，直接使用 out_channels 作为 group 数。
        num_groups = 8 if out_channels % 8 == 0 else out_channels
        # 通过 strided conv 同时完成投影与可选下采样。
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=spatial_stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, in_channels, H, W)。

        Returns:
            输出张量，形状为 (B, out_channels, H/S, W/S)，其中 S 为
            ``spatial_stride``。
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SensorEncoderBank(nn.Module):
    """多源独立 encoder bank。

    使用 ``nn.ModuleDict`` 为每个数据源维护一个独立的 ``SensorEncoder``，
    不同传感器之间不共享参数，避免波段语义不一致带来的干扰。
    """

    def __init__(
        self,
        sensor_configs: dict[str, int],
        out_channels: int,
        spatial_stride: int = 1,
    ) -> None:
        """初始化 SensorEncoderBank。

        Args:
            sensor_configs: 数据源到输入通道数的映射，例如
                ``{"s2": 13, "s1": 2, "landsat": 11}``。
            out_channels: 每个编码器输出的统一通道数。
            spatial_stride: 所有 sensor encoder 共用的空间下采样倍数。
        """
        super().__init__()
        self.sensor_configs = sensor_configs
        self.out_channels = out_channels
        self.spatial_stride = spatial_stride
        self.encoders = nn.ModuleDict(
            {
                source: SensorEncoder(in_channels, out_channels, spatial_stride)
                for source, in_channels in sensor_configs.items()
            }
        )

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """根据 source 选择对应编码器进行前向传播。

        Args:
            x: 输入张量，形状为 (B, in_channels, H, W)。
            source: 数据源名称，必须在 ``sensor_configs`` 中存在。

        Returns:
            输出张量，形状为 (B, out_channels, H/S, W/S)，其中 S 为
            ``spatial_stride``。

        Raises:
            KeyError: 当 ``source`` 不在已注册的编码器中时抛出。
        """
        if source not in self.encoders:
            raise KeyError(f"未知数据源: {source!r}，可用数据源: {list(self.encoders.keys())}")
        return self.encoders[source](x)


class NativeResolutionHighResEncoder(nn.Module):
    """原生分辨率高分数据源编码器（阶段二使用）。

    对来自不同传感器、不同原生分辨率与 GSD 的高分辨率影像，先通过 3 层 stride=2
    卷积（每层后接 GroupNorm + GELU）逐步下采样并扩展感受野，最后通过
    ``AdaptiveAvgPool2d`` 统一到与低分辨率基础特征相同的空间尺寸，输出通道数为
    ``out_channels``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        """初始化 NativeResolutionHighResEncoder。

        Args:
            in_channels: 高分数据源输入通道数。
            out_channels: 输出通道数，通常等于基础模型 embed_dim。
            target_size: 默认目标空间分辨率，默认与 128×128 patch 一致；
                调用 ``forward`` 时可覆盖。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size

        # 3 层 stride=2 下采样：2^3 = 8 倍下采样。
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)

        # GroupNorm 要求 num_channels 能被 num_groups 整除；
        # 当通道数小于 8 时，回退到逐通道 group。
        self.norm1 = nn.GroupNorm(min(32, 32) if 32 % 32 == 0 else 32, 32)
        self.norm2 = nn.GroupNorm(min(32, 64) if 64 % 32 == 0 else 64, 64)
        num_groups = 8 if out_channels % 8 == 0 else out_channels
        self.norm3 = nn.GroupNorm(num_groups, out_channels)

    def forward(
        self, x: torch.Tensor, target_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 ``(B, in_channels, H_native, W_native)``。
            target_size: 可选覆盖目标空间分辨率；为 ``None`` 时使用构造默认值。

        Returns:
            输出张量，形状 ``(B, out_channels, H_target, W_target)``。
        """
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        target_size = target_size if target_size is not None else self.target_size
        x = F.adaptive_avg_pool2d(x, target_size)
        return x
