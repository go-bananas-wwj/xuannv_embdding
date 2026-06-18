from __future__ import annotations

# 训练入口与 batch 预处理单元测试。
import pytest
import torch

from xuannv_embedding.training.batch_preparation import _head_source_name, prepare_batch


def test_prepare_batch_global_timestamps() -> None:
    """应从第一个有效 source 提取全局 timestamps。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, 0, 3, 4, 4),
            "s1": torch.zeros(2, 3, 2, 4, 4),
        },
        "source_masks": {
            "s2": torch.zeros(2, 0),
            "s1": torch.ones(2, 3),
        },
        "timestamps": {
            "s2": torch.zeros(2, 0, dtype=torch.long),
            "s1": torch.tensor([[202501, 202502, 202503], [202501, 202502, 202503]]),
        },
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert "timestamps" in out
    assert out["timestamps"].shape == (2, 3)
    assert torch.equal(
        out["timestamps"],
        torch.tensor([[202501, 202502, 202503], [202501, 202502, 202503]]),
    )


def test_prepare_batch_continuous_target() -> None:
    """continuous head 应生成时间加权平均 target 与全 1 target_mask。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": torch.tensor(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ]
            )
            .float()
            .unsqueeze(0),
        },
        "source_masks": {
            "s2": torch.tensor([[1.0, 1.0]]),
        },
        "timestamps": {
            "s2": torch.tensor([[202501, 202502]], dtype=torch.long),
        },
    }
    # 修正 batch 维度为 [B=1, T=2, C=1, H=2, W=2]
    batch["source_frames"]["s2"] = batch["source_frames"]["s2"].reshape(1, 2, 1, 2, 2)

    target_heads = {
        "s2_recon": {
            "loss_type": "continuous",
            "channels": 1,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert "targets" in out
    assert "target_masks" in out
    expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]]).float().unsqueeze(1)
    assert torch.allclose(out["targets"]["s2_recon"], expected)
    assert torch.equal(
        out["target_masks"]["s2_recon"],
        torch.ones(1, 2, 2),
    )


def test_prepare_batch_categorical_target() -> None:
    """categorical head 应通过 argmax 生成 [B, H, W] 的 target。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "worldcover": torch.tensor(
                [
                    [
                        [[0.1, 0.9], [0.8, 0.2]],
                        [[0.7, 0.3], [0.4, 0.6]],
                        [[0.2, 0.1], [0.1, 0.7]],
                    ]
                ]
            ).float(),
        },
        "source_masks": {
            "worldcover": torch.ones(1, 1),
        },
        "timestamps": {
            "worldcover": torch.tensor([[202501]], dtype=torch.long),
        },
    }
    # 维度修正为 [B=1, T=1, C=3, H=2, W=2]
    batch["source_frames"]["worldcover"] = batch["source_frames"]["worldcover"].reshape(
        1, 1, 3, 2, 2
    )

    target_heads = {
        "worldcover": {
            "loss_type": "categorical",
            "channels": 3,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert "targets" in out
    assert out["targets"]["worldcover"].shape == (1, 2, 2)
    expected = torch.tensor([[[1, 0], [0, 2]]]).long()
    assert torch.equal(out["targets"]["worldcover"], expected)


def test_prepare_batch_highres_separation() -> None:
    """highres source 应被聚合为单帧，并生成正确形状的可用性掩码。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, 1, 3, 4, 4),
            "highres": torch.ones(2, 2, 3, 4, 4),
        },
        "source_masks": {
            "s2": torch.ones(2, 1),
            "highres": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        },
        "timestamps": {
            "s2": torch.tensor([[202501], [202501]], dtype=torch.long),
            "highres": torch.tensor([[202501, 202502], [202501, 202502]], dtype=torch.long),
        },
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert "highres" not in out["source_frames"]
    assert "highres" in out["highres_frames"]
    assert out["highres_frames"]["highres"].shape == (2, 3, 4, 4)
    assert "highres" in out["highres_masks"]
    assert out["highres_masks"]["highres"].shape == (2, 1, 4, 4)
    # 第二个样本 highres 第二帧缺失，但仍有一帧有效，掩码应为 1。
    assert out["highres_masks"]["highres"][0].sum().item() == 16.0
    assert out["highres_masks"]["highres"][1].sum().item() == 16.0


def test_prepare_batch_multiple_highres_sources() -> None:
    """多个以 highres 开头的 source 应被分别聚合并保留各自 source 名。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, 1, 3, 4, 4),
            "highres": torch.ones(2, 2, 3, 4, 4),
            "highres_sar": torch.ones(2, 1, 1, 4, 4) * 2.0,
        },
        "source_masks": {
            "s2": torch.ones(2, 1),
            "highres": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            "highres_sar": torch.ones(2, 1),
        },
        "timestamps": {
            "s2": torch.tensor([[202501], [202501]], dtype=torch.long),
            "highres": torch.tensor([[202501, 202502], [202501, 202502]], dtype=torch.long),
            "highres_sar": torch.tensor([[202501], [202501]], dtype=torch.long),
        },
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert "highres" not in out["source_frames"]
    assert "highres_sar" not in out["source_frames"]
    assert set(out["highres_frames"].keys()) == {"highres", "highres_sar"}
    assert out["highres_frames"]["highres"].shape == (2, 3, 4, 4)
    assert out["highres_frames"]["highres_sar"].shape == (2, 1, 4, 4)
    assert out["highres_masks"]["highres"].shape == (2, 1, 4, 4)
    assert out["highres_masks"]["highres_sar"].shape == (2, 1, 4, 4)


def test_prepare_batch_highres_mask_with_spatial_stride() -> None:
    """spatial_stride > 1 时 highres_mask 应下采样到与编码特征对齐的分辨率。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": torch.zeros(1, 1, 3, 8, 8),
            "highres": torch.ones(1, 1, 3, 8, 8),
        },
        "source_masks": {
            "s2": torch.ones(1, 1),
            "highres": torch.ones(1, 1),
        },
        "timestamps": {
            "s2": torch.tensor([[202501]], dtype=torch.long),
            "highres": torch.tensor([[202501]], dtype=torch.long),
        },
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads, spatial_stride=2)

    assert out["highres_frames"]["highres"].shape == (1, 3, 8, 8)
    assert out["highres_masks"]["highres"].shape == (1, 1, 4, 4)
    # 全部可用，下采样后掩码仍应为 1。
    assert torch.equal(out["highres_masks"]["highres"], torch.ones(1, 1, 4, 4))


def test_prepare_batch_missing_source() -> None:
    """target 对应 source 缺失时应返回零 target 与零 target_mask。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": torch.zeros(1, 0, 3, 4, 4),
        },
        "source_masks": {
            "s2": torch.zeros(1, 0),
        },
        "timestamps": {
            "s2": torch.zeros(1, 0, dtype=torch.long),
        },
    }
    target_heads = {
        "s2_recon": {
            "loss_type": "continuous",
            "channels": 3,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert torch.equal(out["targets"]["s2_recon"], torch.zeros(1, 3, 4, 4))
    assert torch.equal(out["target_masks"]["s2_recon"], torch.zeros(1, 4, 4))


def test_head_source_name_priority() -> None:
    """_head_source_name 的优先级：显式 source > _recon 后缀 > head name 本身。"""
    available_sources = {"s2", "s1", "worldcover"}

    # 1. 显式 source 字段优先。
    assert (
        _head_source_name(
            "s2_recon",
            {"loss_type": "continuous", "channels": 3, "source": "worldcover"},
            available_sources,
        )
        == "worldcover"
    )

    # 2. 无显式 source 时，_recon 后缀匹配。
    assert (
        _head_source_name(
            "s2_recon",
            {"loss_type": "continuous", "channels": 3},
            available_sources,
        )
        == "s2"
    )

    # 3. 无 _recon 后缀时，head name 本身作为 source 名匹配。
    assert (
        _head_source_name(
            "worldcover",
            {"loss_type": "categorical", "channels": 11},
            available_sources,
        )
        == "worldcover"
    )

    # 4. 未匹配到任何可用 source。
    assert (
        _head_source_name(
            "landsat_recon",
            {"loss_type": "continuous", "channels": 6},
            available_sources,
        )
        is None
    )


def test_prepare_batch_empty_source_frames() -> None:
    """source_frames 为空字典时应抛出清晰的 ValueError。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {},
        "source_masks": {},
        "timestamps": {},
    }
    target_heads: dict[str, dict] = {}

    with pytest.raises(ValueError, match="source_frames 不能为空"):
        prepare_batch(batch, target_heads)
