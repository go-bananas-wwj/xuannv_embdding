from __future__ import annotations

import pytest
import torch
from downstreams.heads import (
    ChangeDetectionHead,
    ClassificationHead,
    DiffUNetHead,
    FCNHead,
    LinearProbeHead,
    MLPHead,
    UNetHead,
    UperNetHead,
    build_segmentation_head,
)
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask


@pytest.fixture
def seg_input():
    """语义分割头常用的 embedding map。"""
    return torch.randn(2, 64, 16, 16)


@pytest.fixture
def scene_emb():
    """场景级 embedding。"""
    return torch.randn(2, 64)


def test_linear_probe_head(seg_input: torch.Tensor) -> None:
    head = LinearProbeHead(in_channels=64, num_classes=5)
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


def test_fcn_head(seg_input: torch.Tensor) -> None:
    head = FCNHead(in_channels=64, num_classes=5, hidden_dim=128)
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


def test_unet_head(seg_input: torch.Tensor) -> None:
    head = UNetHead(in_channels=64, num_classes=5)
    out = head(seg_input)
    # UNetHead 内部上采样后再插值回原始尺寸
    assert out.shape == (2, 5, 16, 16)


def test_upernet_head(seg_input: torch.Tensor) -> None:
    head = UperNetHead(in_channels=64, num_classes=5, pool_scales=(1, 2, 3, 6))
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
        ("mlp", MLPHead),
        ("diff_unet", DiffUNetHead),
    ],
)
def test_build_segmentation_head(head_type: str, expected_cls: type) -> None:
    head = build_segmentation_head(head_type, in_channels=64, num_classes=5)
    assert isinstance(head, expected_cls)


def test_build_segmentation_head_unknown() -> None:
    with pytest.raises(ValueError, match="未知 head 类型"):
        build_segmentation_head("unknown", in_channels=64, num_classes=5)


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


def test_mlp_head(seg_input: torch.Tensor) -> None:
    head = MLPHead(in_channels=64, num_classes=5, hidden_dim=128)
    out = head(seg_input)
    assert out.shape == (2, 5, 16, 16)


@pytest.mark.parametrize(
    "head_type",
    ["linear", "fcn", "unet", "upernet", "mlp", "diff_unet"],
)
def test_head_bitemporal_192(head_type: str) -> None:
    """所有 head 都应支持 192 通道双时相拼接输入。"""
    x = torch.randn(2, 192, 32, 32)
    head = build_segmentation_head(head_type, in_channels=192, num_classes=5)
    out = head(x)
    assert out.shape == (2, 5, 32, 32)


def test_diff_unet_head_with_diff() -> None:
    # 3 * 64 = 192 channels: [emb_t1, emb_t2, |diff|]
    x = torch.randn(2, 192, 16, 16)
    head = DiffUNetHead(in_channels=64, num_classes=5, hidden_dim=128)
    out = head(x)
    assert out.shape == (2, 5, 16, 16)


def test_diff_unet_head_without_diff() -> None:
    # 64 channels: single temporal embedding
    x = torch.randn(2, 64, 16, 16)
    head = DiffUNetHead(in_channels=64, num_classes=5, hidden_dim=128, use_diff=False)
    out = head(x)
    assert out.shape == (2, 5, 16, 16)


def _first_conv_in_channels(model: torch.nn.Module) -> int | None:
    """返回模型中第一个 Conv2d 的输入通道数。"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return module.in_channels
    return None


@pytest.mark.parametrize(
    "months, expected_in_channels",
    [
        (["202605"], 64),
        (["202512", "202605"], 192),
    ],
)
def test_construction_task_build_head_channels(
    months: list[str], expected_in_channels: int
) -> None:
    cfg = {
        "training": {"head_type": "fcn", "months": months},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    head = task.build_head()
    assert _first_conv_in_channels(head) == expected_in_channels


@pytest.mark.parametrize("months", [[], ["202512", "202601", "202605"]])
def test_construction_task_build_head_bad_months(months: list[str]) -> None:
    cfg = {
        "training": {"head_type": "fcn", "months": months},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    with pytest.raises(ValueError, match="training.months 长度必须为 1 或 2"):
        task.build_head()
