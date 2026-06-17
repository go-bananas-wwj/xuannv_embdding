from __future__ import annotations

# 训练入口与 batch 预处理单元测试。
import torch

from xuannv_embedding.training.batch_preparation import prepare_batch


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
    device = torch.device("cpu")

    out = prepare_batch(batch, target_heads, device)

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
    device = torch.device("cpu")

    out = prepare_batch(batch, target_heads, device)

    assert "targets" in out
    assert "target_masks" in out
    expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]]).float().unsqueeze(1)
    assert torch.allclose(out["targets"]["s2_recon"], expected)
    assert torch.equal(
        out["target_masks"]["s2_recon"],
        torch.ones(1, 2, 2),
    )


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
    device = torch.device("cpu")

    out = prepare_batch(batch, target_heads, device)

    assert "highres" not in out["source_frames"]
    assert out["highres_frame"] is not None
    assert out["highres_frame"].shape == (2, 3, 4, 4)
    assert out["highres_mask"] is not None
    assert out["highres_mask"].shape == (2, 1, 4, 4)
    # 第二个样本 highres 第二帧缺失，但仍有一帧有效，掩码应为 1。
    assert out["highres_mask"][0].sum().item() == 16.0
    assert out["highres_mask"][1].sum().item() == 16.0


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
    device = torch.device("cpu")

    out = prepare_batch(batch, target_heads, device)

    assert torch.equal(out["targets"]["s2_recon"], torch.zeros(1, 3, 4, 4))
    assert torch.equal(out["target_masks"]["s2_recon"], torch.zeros(1, 4, 4))
