from __future__ import annotations

# 模型模块单元测试。
import pytest
import torch

from xuannv_embedding.models.blocks import SpaceOperator, TimeOperator
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.model import AEFModel, AEFOutput
from xuannv_embedding.models.sensor_encoders import SensorEncoder, SensorEncoderBank
from xuannv_embedding.models.time_encoding import TimeEncoding, WindowCode


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


def test_time_encoding() -> None:
    """TimeEncoding 应将时间戳编码为 (B, T, embed_dim)。"""
    batch_size, time_steps, embed_dim = 2, 6, 16

    encoding = TimeEncoding(embed_dim, max_periods=64)
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)
    out = encoding(timestamps)

    assert out.shape == (batch_size, time_steps, embed_dim)


def test_window_code() -> None:
    """WindowCode 应将窗口 ID 映射为 embedding。"""
    num_windows, embed_dim = 12, 16
    batch_size = 2

    window_code = WindowCode(num_windows, embed_dim)
    window_ids = torch.randint(0, num_windows, (batch_size,))
    out = window_code(window_ids)

    assert out.shape == (batch_size, embed_dim)


def test_aef_model_forward() -> None:
    """AEFModel 应能前向传播并返回正确形状的输出（含高分辨率数据）。"""
    sensor_channels = {"s2": 10, "s1": 2, "landsat": 6, "highres": 3, "highres_sar": 1}
    embed_dim = 16
    target_heads = {
        "s2_recon": ("continuous", 10),
        "s1_recon": ("continuous", 2),
        "worldcover": ("categorical", 11),
    }
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        num_time_heads=2,
        num_space_heads=2,
    )

    batch_size, time_steps, height, width = 2, 4, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
        "landsat": torch.randn(batch_size, time_steps, 6, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)
    highres_frames = {
        "highres": torch.randn(batch_size, 3, height, width),
        "highres_sar": torch.randn(batch_size, 1, height, width),
    }
    highres_masks = {
        "highres": torch.ones(batch_size, 1, height, width),
        "highres_sar": torch.ones(batch_size, 1, height, width),
    }

    out = model(
        source_frames,
        source_masks,
        timestamps,
        highres_frames,
        highres_masks,
    )

    assert isinstance(out, AEFOutput)
    assert out.embedding.shape == (batch_size, embed_dim)
    assert out.embedding_map.shape == (batch_size, embed_dim, height, width)
    assert set(out.reconstructions.keys()) == set(target_heads.keys())
    assert out.reconstructions["s2_recon"].shape == (batch_size, 10, height, width)
    assert out.reconstructions["s1_recon"].shape == (batch_size, 2, height, width)
    assert out.reconstructions["worldcover"].shape == (batch_size, 11, height, width)

    # embedding_map 应位于单位球面上。
    norms = out.embedding_map.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_aef_model_without_highres() -> None:
    """AEFModel 在没有高分辨率数据时也应能正常前向传播。"""
    sensor_channels = {"s2": 10, "s1": 2, "landsat": 6}
    embed_dim = 16
    target_heads = {"s2_recon": ("continuous", 10)}
    model = AEFModel(
        sensor_channels,
        embed_dim,
        target_heads,
        num_time_heads=2,
        num_space_heads=2,
    )

    batch_size, time_steps, height, width = 2, 3, 16, 16
    source_frames = {
        "s2": torch.randn(batch_size, time_steps, 10, height, width),
        "s1": torch.randn(batch_size, time_steps, 2, height, width),
        "landsat": torch.randn(batch_size, time_steps, 6, height, width),
    }
    source_masks = {k: torch.ones(batch_size, time_steps) for k in source_frames}
    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)

    out = model(source_frames, source_masks, timestamps)

    assert isinstance(out, AEFOutput)
    assert out.embedding.shape == (batch_size, embed_dim)
    assert out.embedding_map.shape == (batch_size, embed_dim, height, width)
    assert "s2_recon" in out.reconstructions
    assert out.reconstructions["s2_recon"].shape == (batch_size, 10, height, width)
