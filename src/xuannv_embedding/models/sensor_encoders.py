from __future__ import annotations

# 多源遥感数据 stem encoder：为每个数据源提供独立的浅层卷积编码器。

import torch
from torch import nn


class SensorEncoder(nn.Module):
    """单个数据源的浅层卷积 stem。

    结构为 Conv2d -> GroupNorm -> ReLU，用于把多波段输入投影到统一的
    中间通道数，便于后续空间/时间融合模块处理。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """初始化 SensorEncoder。

        Args:
            in_channels: 输入数据的波段数，例如 S2=13、S1=2。
            out_channels: 输出通道数，即投影后的统一通道数。
        """
        super().__init__()
        # GroupNorm 要求 num_channels 能被 num_groups 整除；
        # 当 out_channels 小于 8 时，直接使用 out_channels 作为 group 数。
        num_groups = 8 if out_channels % 8 == 0 else out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状为 (B, in_channels, H, W)。

        Returns:
            输出张量，形状为 (B, out_channels, H, W)。
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

    def __init__(self, sensor_configs: dict[str, int], out_channels: int) -> None:
        """初始化 SensorEncoderBank。

        Args:
            sensor_configs: 数据源到输入通道数的映射，例如
                ``{"s2": 13, "s1": 2, "landsat": 11}``。
            out_channels: 每个编码器输出的统一通道数。
        """
        super().__init__()
        self.sensor_configs = sensor_configs
        self.out_channels = out_channels
        self.encoders = nn.ModuleDict(
            {
                source: SensorEncoder(in_channels, out_channels)
                for source, in_channels in sensor_configs.items()
            }
        )

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """根据 source 选择对应编码器进行前向传播。

        Args:
            x: 输入张量，形状为 (B, in_channels, H, W)。
            source: 数据源名称，必须在 ``sensor_configs`` 中存在。

        Returns:
            输出张量，形状为 (B, out_channels, H, W)。

        Raises:
            KeyError: 当 ``source`` 不在已注册的编码器中时抛出。
        """
        if source not in self.encoders:
            raise KeyError(
                f"未知数据源: {source!r}，可用数据源: {list(self.encoders.keys())}"
            )
        return self.encoders[source](x)
