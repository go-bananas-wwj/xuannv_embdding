from __future__ import annotations

# 训练入口与 batch 预处理单元测试。
import pytest
import torch

from xuannv_embedding.training.batch_preparation import _head_source_name, prepare_batch

NUM_MONTHS = 17


def _month_tensor() -> torch.Tensor:
    """返回阶段一 2025-01 至 2026-05 的 YYYYMM 月度列表。"""
    year, month = 2025, 1
    values = []
    for _ in range(NUM_MONTHS):
        values.append(year * 100 + month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return torch.tensor(values, dtype=torch.long)


MONTH_TENSOR = _month_tensor()


def test_prepare_batch_global_timestamps() -> None:
    """应从所有非高分辨率 source 提取统一的全局 timestamps。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, NUM_MONTHS, 3, 4, 4),
            "s1": torch.zeros(2, NUM_MONTHS, 2, 4, 4),
        },
        "source_masks": {
            "s2": torch.ones(2, NUM_MONTHS),
            "s1": torch.ones(2, NUM_MONTHS),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0).expand(2, -1),
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert "timestamps" in out
    assert out["timestamps"].shape == (2, NUM_MONTHS)
    assert torch.equal(out["timestamps"], MONTH_TENSOR.unsqueeze(0).expand(2, -1))


def test_prepare_batch_continuous_target() -> None:
    """continuous head 应生成逐月 target 与逐月 target_mask。"""
    frames = torch.tensor(
        [
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[5.0, 6.0], [7.0, 8.0]]],
            ]
        ]
    ).float()
    # (B=1, T=2, C=1, H=2, W=2) -> 补齐到 NUM_MONTHS
    frames = torch.cat(
        [frames, torch.zeros(1, NUM_MONTHS - 2, 1, 2, 2)], dim=1
    )
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {"s2": frames},
        "source_masks": {
            "s2": torch.cat([torch.ones(1, 2), torch.zeros(1, NUM_MONTHS - 2)], dim=1),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
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
    assert out["targets"]["s2_recon"].shape == (1, NUM_MONTHS, 1, 2, 2)
    assert torch.allclose(out["targets"]["s2_recon"], frames)
    assert out["target_masks"]["s2_recon"].shape == (1, NUM_MONTHS)
    assert torch.equal(
        out["target_masks"]["s2_recon"],
        torch.cat([torch.ones(1, 2), torch.zeros(1, NUM_MONTHS - 2)], dim=1),
    )


def test_prepare_batch_source_dropout_keeps_targets() -> None:
    """source dropout 只应作用于模型输入，不能删除重建 target。"""
    frames = torch.ones(1, NUM_MONTHS, 1, 2, 2)
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {"s2": frames.clone()},
        "source_masks": {"s2": torch.ones(1, NUM_MONTHS)},
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
    target_heads = {
        "s2_recon": {
            "loss_type": "continuous",
            "channels": 1,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads, source_dropout_probs={"s2": 1.0})

    assert torch.equal(out["source_frames"]["s2"], torch.zeros_like(frames))
    assert torch.equal(out["source_masks"]["s2"], torch.zeros(1, NUM_MONTHS))
    assert torch.equal(out["targets"]["s2_recon"], frames)
    assert torch.equal(out["target_masks"]["s2_recon"], torch.ones(1, NUM_MONTHS))


def test_prepare_batch_categorical_target() -> None:
    """categorical head 应通过 argmax 生成逐月 (B, T, H, W) target 与空间掩码。"""
    # (B=1, T=NUM_MONTHS, C=3, H=2, W=2)
    worldcover = torch.zeros(1, NUM_MONTHS, 3, 2, 2)
    worldcover[0, 0, 1, 0, 1] = 1.0
    worldcover[0, 0, 0, 1, 0] = 1.0
    worldcover[0, 0, 2, 1, 1] = 1.0
    worldcover[0, 1, 1, 0, 0] = 1.0

    batch = {
        "patch_ids": ["p0"],
        "source_frames": {"worldcover": worldcover},
        "source_masks": {"worldcover": torch.ones(1, NUM_MONTHS)},
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
    target_heads = {
        "worldcover": {
            "loss_type": "categorical",
            "channels": 3,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert "targets" in out
    assert out["targets"]["worldcover"].shape == (1, NUM_MONTHS, 2, 2)
    assert out["target_masks"]["worldcover"].shape == (1, NUM_MONTHS, 2, 2)
    assert out["target_masks"]["worldcover"][0, 0, 0, 1].item() == 1.0
    assert out["target_masks"]["worldcover"][0, 0, 0, 0].item() == 0.0


def test_prepare_batch_categorical_static_replicated() -> None:
    """静态 worldcover 单帧应被复制到所有月份。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "worldcover": torch.tensor(
                [
                    [
                        [[[0, 1], [2, 0]]],
                    ]
                ]
            ).float(),
        },
        "source_masks": {
            "worldcover": torch.ones(1, 1),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
    # 维度修正为 [B=1, T=1, C=1, H=2, W=2]
    batch["source_frames"]["worldcover"] = batch["source_frames"]["worldcover"].reshape(
        1, 1, 1, 2, 2
    )

    target_heads = {
        "worldcover": {
            "loss_type": "categorical",
            "channels": 3,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert out["targets"]["worldcover"].shape == (1, NUM_MONTHS, 2, 2)
    expected_label = torch.tensor([[0, 1], [2, 0]]).long()
    for t in range(NUM_MONTHS):
        assert torch.equal(out["targets"]["worldcover"][0, t], expected_label)
    expected_mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    for t in range(NUM_MONTHS):
        assert torch.equal(out["target_masks"]["worldcover"][0, t], expected_mask)


def test_prepare_batch_highres_separation() -> None:
    """highres source 应被聚合为单帧，并生成正确形状的可用性掩码。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, NUM_MONTHS, 3, 4, 4),
            "highres": torch.ones(2, 2, 3, 4, 4),
        },
        "source_masks": {
            "s2": torch.ones(2, NUM_MONTHS),
            "highres": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0).expand(2, -1),
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


def test_prepare_batch_highres_targets_are_binned_by_month() -> None:
    """highres 重建 target 只应监督真实有高分观测的月份。"""
    month_tensor = torch.tensor(
        [202512, 202601, 202602, 202603, 202604, 202605],
        dtype=torch.long,
    )
    s2 = torch.zeros(1, 6, 3, 2, 2)
    # 两期 highres，第一期为 10，末期为 30；中间月份无高分监督。
    highres = torch.stack(
        [
            torch.full((1, 4, 4), 10.0),
            torch.full((1, 4, 4), 30.0),
        ],
        dim=0,
    ).unsqueeze(0)
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": s2,
            "highres": highres,
        },
        "source_masks": {
            "s2": torch.ones(1, 6),
            "highres": torch.ones(1, 2),
        },
        "timestamps": month_tensor.unsqueeze(0),
        "source_timestamps": {
            "s2": month_tensor.unsqueeze(0),
            "highres": torch.tensor([[20251201, 20260501]], dtype=torch.long),
        },
    }
    target_heads = {
        "highres_recon": {
            "source": "highres",
            "loss_type": "continuous",
            "channels": 1,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    target = out["targets"]["highres_recon"]
    target_mask = out["target_masks"]["highres_recon"]
    assert target.shape == (1, 6, 1, 2, 2)
    assert torch.equal(
        target_mask,
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    assert torch.equal(target[0, 0], torch.full((1, 2, 2), 10.0))
    assert torch.equal(target[0, 5], torch.full((1, 2, 2), 30.0))
    assert torch.equal(target[0, 1:5], torch.zeros(4, 1, 2, 2))


def test_prepare_batch_multiple_highres_sources() -> None:
    """多个以 highres 开头的 source 应被分别聚合并保留各自 source 名。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, NUM_MONTHS, 3, 4, 4),
            "highres": torch.ones(2, 2, 3, 4, 4),
            "highres_sar": torch.ones(2, 1, 1, 4, 4) * 2.0,
        },
        "source_masks": {
            "s2": torch.ones(2, NUM_MONTHS),
            "highres": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            "highres_sar": torch.ones(2, 1),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0).expand(2, -1),
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


def test_prepare_batch_highres_mask_full_resolution() -> None:
    """默认情况下 highres_mask 应保持与输入相同的空间分辨率。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": torch.zeros(1, NUM_MONTHS, 3, 8, 8),
            "highres": torch.ones(1, 1, 3, 8, 8),
        },
        "source_masks": {
            "s2": torch.ones(1, NUM_MONTHS),
            "highres": torch.ones(1, 1),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert out["highres_frames"]["highres"].shape == (1, 3, 8, 8)
    assert out["highres_masks"]["highres"].shape == (1, 1, 8, 8)
    # 全部可用，掩码仍应为 1。
    assert torch.equal(out["highres_masks"]["highres"], torch.ones(1, 1, 8, 8))


def test_prepare_batch_missing_source() -> None:
    """target 对应 source 缺失时应返回零 target 与零 target_mask。"""
    batch = {
        "patch_ids": ["p0"],
        "source_frames": {
            "s2": torch.zeros(1, NUM_MONTHS, 3, 4, 4),
        },
        "source_masks": {
            "s2": torch.zeros(1, NUM_MONTHS),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0),
    }
    target_heads = {
        "s1_recon": {
            "loss_type": "continuous",
            "channels": 2,
            "weight": 1.0,
        }
    }

    out = prepare_batch(batch, target_heads)

    assert torch.equal(out["targets"]["s1_recon"], torch.zeros(1, NUM_MONTHS, 2, 4, 4))
    assert torch.equal(out["target_masks"]["s1_recon"], torch.zeros(1, NUM_MONTHS))


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
        "timestamps": torch.zeros(1, 0, dtype=torch.long),
    }
    target_heads: dict[str, dict] = {}

    with pytest.raises(ValueError, match="source_frames 不能为空"):
        prepare_batch(batch, target_heads)


def test_prepare_batch_temporal_sources_aligned() -> None:
    """所有时序 source 应共享统一的月度时间长度与时间戳。"""
    batch = {
        "patch_ids": ["p0", "p1"],
        "source_frames": {
            "s2": torch.zeros(2, NUM_MONTHS, 3, 4, 4),
            "s1": torch.zeros(2, NUM_MONTHS, 2, 4, 4),
        },
        "source_masks": {
            "s2": torch.ones(2, NUM_MONTHS),
            "s1": torch.ones(2, NUM_MONTHS),
        },
        "timestamps": MONTH_TENSOR.unsqueeze(0).expand(2, -1),
    }
    target_heads: dict[str, dict] = {}

    out = prepare_batch(batch, target_heads)

    assert out["source_frames"]["s2"].shape == (2, NUM_MONTHS, 3, 4, 4)
    assert out["source_frames"]["s1"].shape == (2, NUM_MONTHS, 2, 4, 4)
    assert out["timestamps"].shape == (2, NUM_MONTHS)
    assert torch.equal(out["timestamps"], MONTH_TENSOR.unsqueeze(0).expand(2, -1))
