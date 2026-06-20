from __future__ import annotations

import pytest
import torch
from downstreams.heads import (
    ChangeDetectionHead,
    ClassificationHead,
    FCNHead,
    LinearProbeHead,
    UNetHead,
    UperNetHead,
    build_segmentation_head,
)


@pytest.fixture
def seg_input():
    """语义分割头常用的 embedding map。"""
    return torch.randn(2, 64, 16, 16)


@pytest.fixture
def scene_emb():
    """场景级 embedding。"""
    return torch.randn(2, 64)


def test_linear_probe_head(seg_input: torch.Tensor) -> None:
    head = LinearProbeHead(embed_dim=64, num_classes=5)
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


def test_fcn_head(seg_input: torch.Tensor) -> None:
    head = FCNHead(embed_dim=64, num_classes=5, hidden_dim=128)
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


def test_unet_head(seg_input: torch.Tensor) -> None:
    head = UNetHead(embed_dim=64, num_classes=5)
    out = head(seg_input)
    # UNetHead 内部上采样后再插值回原始尺寸
    assert out.shape == (2, 5, 16, 16)


def test_upernet_head(seg_input: torch.Tensor) -> None:
    head = UperNetHead(embed_dim=64, num_classes=5, pool_scales=(1, 2, 3, 6))
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


@pytest.mark.parametrize(
    "head_type, expected_cls",
    [
        ("linear", LinearProbeHead),
        ("linear_probe", LinearProbeHead),
        ("fcn", FCNHead),
        ("unet", UNetHead),
        ("upernet", UperNetHead),
    ],
)
def test_build_segmentation_head(head_type: str, expected_cls: type) -> None:
    head = build_segmentation_head(head_type, embed_dim=64, num_classes=5)
    assert isinstance(head, expected_cls)


def test_build_segmentation_head_unknown() -> None:
    with pytest.raises(ValueError, match="未知 head 类型"):
        build_segmentation_head("unknown", embed_dim=64, num_classes=5)


def test_classification_head(scene_emb: torch.Tensor) -> None:
    head = ClassificationHead(embed_dim=64, num_classes=10, hidden_dim=128)
    dummy_map = torch.randn(2, 64, 16, 16)
    out = head(dummy_map, scene_emb)
    assert out.shape == (2, 10)


def test_classification_head_requires_scene_emb(scene_emb: torch.Tensor) -> None:
    head = ClassificationHead(embed_dim=64, num_classes=10)
    dummy_map = torch.randn(2, 64, 16, 16)
    with pytest.raises(ValueError, match="需要 scene_emb"):
        head(dummy_map, None)


def test_change_detection_head_forward_two() -> None:
    emb_t1 = torch.randn(2, 64, 16, 16)
    emb_t2 = torch.randn(2, 64, 16, 16)
    head = ChangeDetectionHead(embed_dim=64, hidden_dim=256)
    out = head.forward_two(emb_t1, emb_t2)
    assert out.shape == (2, 1, 16, 16)


def test_change_detection_head_forward_raises() -> None:
    head = ChangeDetectionHead(embed_dim=64, hidden_dim=256)
    x = torch.randn(2, 128, 16, 16)
    with pytest.raises(NotImplementedError):
        head(x)
