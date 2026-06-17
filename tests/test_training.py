from __future__ import annotations

# 训练模块单元测试。
import torch
import torch.nn.functional as F

from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.losses import (
    TotalLoss,
    batch_uniformity_loss,
    reconstruction_loss,
)


def test_reconstruction_loss_l1() -> None:
    """验证 L1 掩码重建损失计算正确。"""
    batch_size, channels, height, width = 2, 3, 4, 4
    pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    mask = torch.ones(batch_size, height, width)
    mask[0, :2, :] = 0.0  # 屏蔽第一个样本的上半部分。

    loss = reconstruction_loss(pred, target, mask, loss_type="l1")

    # 手动计算：在通道维度平均 L1 后按 mask 加权平均。
    expected = F.l1_loss(pred, target, reduction="none").mean(dim=1)
    expected_masked_sum = (expected * mask).sum()
    expected_count = mask.sum()
    expected_loss = expected_masked_sum / expected_count

    assert torch.allclose(loss, expected_loss)


def test_reconstruction_loss_ce() -> None:
    """验证 CE 掩码重建损失计算正确。"""
    batch_size, num_classes, height, width = 2, 5, 4, 4
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    mask = torch.ones(batch_size, height, width)
    mask[1, :, :2] = 0.0  # 屏蔽第二个样本的左半部分。

    loss = reconstruction_loss(pred, target, mask, loss_type="ce")

    expected = torch.nn.functional.cross_entropy(pred, target, reduction="none")
    expected_masked_sum = (expected * mask).sum()
    expected_count = mask.sum()
    expected_loss = expected_masked_sum / expected_count

    assert torch.allclose(loss, expected_loss)


def test_batch_uniformity_loss() -> None:
    """验证 embedding 更分散时 uniformity loss 更大。"""
    # 所有向量相同，距离为 0，损失最小。
    emb_same = torch.ones(4, 8)
    loss_same = batch_uniformity_loss(emb_same)

    # 随机向量通常更分散。
    torch.manual_seed(42)
    emb_random = torch.randn(4, 8)
    loss_random = batch_uniformity_loss(emb_random)

    assert loss_same < loss_random
    # 相同向量损失应接近 0。
    assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6)


def test_total_loss() -> None:
    """验证 TotalLoss 返回正确的各字段。"""
    batch_size, embed_dim, height, width = 2, 16, 8, 8

    reconstructions = {
        "s2_recon": torch.randn(batch_size, 10, height, width),
        "worldcover": torch.randn(batch_size, 11, height, width),
    }
    output = AEFOutput(
        embedding_map=torch.randn(batch_size, embed_dim, height, width),
        embedding=torch.randn(batch_size, embed_dim),
        reconstructions=reconstructions,
    )

    targets = {
        "s2_recon": torch.randn(batch_size, 10, height, width),
        "worldcover": torch.randint(0, 11, (batch_size, height, width)),
    }
    masks = {
        "s2_recon": torch.ones(batch_size, height, width),
        "worldcover": torch.ones(batch_size, height, width),
    }

    target_cfg = {
        "s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0},
        "worldcover": {"loss_type": "ce", "channels": 11, "weight": 0.5},
    }

    criterion = TotalLoss(target_cfg)
    losses = criterion(output, targets, masks)

    assert "total" in losses
    assert "recon" in losses
    assert "uniformity" in losses
    assert "recon_s2_recon" in losses
    assert "recon_worldcover" in losses

    expected_total = losses["recon"] + losses["uniformity"]
    assert torch.allclose(losses["total"], expected_total)

    # recon 应为加权求和。
    expected_recon = 1.0 * losses["recon_s2_recon"] + 0.5 * losses["recon_worldcover"]
    assert torch.allclose(losses["recon"], expected_recon)
