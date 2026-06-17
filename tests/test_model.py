from __future__ import annotations

# 模型模块单元测试。
import pytest
import torch

from xuannv_embedding.models.blocks import SpaceOperator, TimeOperator
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.sensor_encoders import SensorEncoder, SensorEncoderBank


def test_space_operator() -> None:
    """SpaceOperator 应保持 (B, H*W, C) 的输入输出形状一致。"""
    batch_size = 2
    height, width = 8, 8
    dim = 64

    operator = SpaceOperator(dim, num_heads=8)
    x = torch.randn(batch_size, height * width, dim)
    y = operator(x)

    assert y.shape == (batch_size, height * width, dim)


def test_time_operator() -> None:
    """TimeOperator 应保持 (B, T, C) 的输入输出形状一致。"""
    batch_size = 2
    time_steps = 12
    dim = 64

    operator = TimeOperator(dim, num_heads=8)
    x = torch.randn(batch_size, time_steps, dim)
    y = operator(x)

    assert y.shape == (batch_size, time_steps, dim)


def test_operator_invalid_heads() -> None:
    """当 dim 不能被 num_heads 整除时应抛出 ValueError。"""
    with pytest.raises(ValueError, match="必须能被"):
        SpaceOperator(dim=63, num_heads=8)


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


def test_vmf_bottleneck() -> None:
    """VMFBottleneck 应保持 (B, C, H, W) 形状并将输出约束到单位球面。"""
    batch_size = 2
    in_dim = 64
    out_dim = 32
    height, width = 8, 8

    bottleneck = VMFBottleneck(in_dim, out_dim, kappa=100.0)
    x = torch.randn(batch_size, in_dim, height, width)
    z = bottleneck(x)

    assert z.shape == (batch_size, out_dim, height, width)

    # 每个空间位置的 L2 范数应接近 1。
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_continuous_decoder() -> None:
    """ContinuousDecoder 应保持空间尺寸并将通道映射到 out_channels。"""
    batch_size = 2
    embed_dim = 64
    out_channels = 13
    height, width = 8, 8

    decoder = ContinuousDecoder(embed_dim, out_channels)
    emb = torch.randn(batch_size, embed_dim, height, width)
    y = decoder(emb)

    assert y.shape == (batch_size, out_channels, height, width)


def test_categorical_decoder() -> None:
    """CategoricalDecoder 应保持空间尺寸并输出 num_classes 个 logits。"""
    batch_size = 2
    embed_dim = 64
    num_classes = 11
    height, width = 8, 8

    decoder = CategoricalDecoder(embed_dim, num_classes)
    emb = torch.randn(batch_size, embed_dim, height, width)
    y = decoder(emb)

    assert y.shape == (batch_size, num_classes, height, width)


def test_availability_aware_fusion() -> None:
    """AvailabilityAwareFusion 应保持形状，并且可用性掩码应影响输出。"""
    batch_size = 2
    dim = 64
    height, width = 8, 8

    fusion = AvailabilityAwareFusion(dim)
    base_feat = torch.randn(batch_size, dim, height, width)
    highres_feat = torch.randn(batch_size, dim, height, width)

    mask_zero = torch.zeros(batch_size, 1, height, width)
    mask_one = torch.ones(batch_size, 1, height, width)

    out_zero = fusion(base_feat, highres_feat, mask_zero)
    out_one = fusion(base_feat, highres_feat, mask_one)

    assert out_zero.shape == (batch_size, dim, height, width)
    assert out_one.shape == (batch_size, dim, height, width)
    assert not torch.allclose(out_zero, out_one)
