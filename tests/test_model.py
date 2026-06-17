from __future__ import annotations

# 模型模块单元测试。
import pytest
import torch

from xuannv_embedding.models.sensor_encoders import SensorEncoder, SensorEncoderBank


def test_sensor_encoder_shape() -> None:
    """单个 SensorEncoder 输出形状应与输入空间尺寸一致、通道数为 out_channels。"""
    batch_size = 2
    in_channels = 13
    out_channels = 64
    height, width = 32, 32

    encoder = SensorEncoder(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    y = encoder(x)

    assert y.shape == (batch_size, out_channels, height, width)


def test_sensor_encoder_bank() -> None:
    """SensorEncoderBank 应能为不同数据源独立编码并输出统一通道数。"""
    sensor_configs = {"s2": 13, "s1": 2}
    out_channels = 64
    batch_size = 2
    height, width = 32, 32

    bank = SensorEncoderBank(sensor_configs, out_channels)

    # S2 输入
    x_s2 = torch.randn(batch_size, sensor_configs["s2"], height, width)
    y_s2 = bank(x_s2, source="s2")
    assert y_s2.shape == (batch_size, out_channels, height, width)

    # S1 输入
    x_s1 = torch.randn(batch_size, sensor_configs["s1"], height, width)
    y_s1 = bank(x_s1, source="s1")
    assert y_s1.shape == (batch_size, out_channels, height, width)


def test_sensor_encoder_bank_unknown_source() -> None:
    """传入未注册的数据源名称时应抛出 KeyError。"""
    bank = SensorEncoderBank({"s2": 13}, out_channels=64)
    x = torch.randn(1, 13, 16, 16)

    with pytest.raises(KeyError, match="未知数据源"):
        bank(x, source="landsat")
