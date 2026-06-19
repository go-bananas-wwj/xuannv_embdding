"""在真实 Harbin 数据子集上跑通 dataset -> model -> loss 的 mini end-to-end 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xuannv_embedding.config import DataConfig
from xuannv_embedding.data.builder import build_dataloader
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.training.losses import TotalLoss

MANIFEST_PATH = Path("/data/xuannv_embedding/processed/harbin/manifest.json")
STATISTICS_DIR = Path("/data/xuannv_embedding/statistics/harbin")


pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists() or not STATISTICS_DIR.exists(),
    reason="Harbin manifest 或统计量尚未生成，跳过 mini end-to-end 测试",
)


def _make_tiny_model() -> AEFModel:
    """构造一个极小的 AEFModel 用于快速验证，不追求精度。"""
    sensor_channels = {"s2": 12, "s1": 2, "landsat": 7}
    target_heads = {
        "s2_recon": ("continuous", 12),
        "s1_recon": ("continuous", 2),
        "landsat_recon": ("continuous", 7),
        "worldcover": ("categorical", 9),
    }
    return AEFModel(
        sensor_channels=sensor_channels,
        embed_dim=16,
        target_heads=target_heads,
        stem_dim=16,
        stp={
            "space_dim": 32,
            "time_dim": 16,
            "precision_dim": 16,
            "num_blocks": 2,
            "num_heads": 2,
        },
        num_space_heads=2,
    )


def test_mini_e2e_forward_and_loss() -> None:
    """从真实数据加载 4 个 patch，跑两个 batch 的前向与损失计算。"""
    cfg = DataConfig(
        root=MANIFEST_PATH.parent,
        region="harbin",
        manifest_path=MANIFEST_PATH,
        statistics_dir=STATISTICS_DIR,
        max_patches=4,
        batch_size=2,
        num_workers=0,
        patch_size=128,
        sources=["s2", "s1", "landsat", "worldcover"],
    )

    loader = build_dataloader(cfg, split="train")
    model = _make_tiny_model()
    model.eval()

    # prepare_batch 使用 continuous/categorical；TotalLoss 使用 l1/ce
    head_cfg_for_prepare = {
        "s2_recon": {"loss_type": "continuous", "channels": 12, "weight": 1.0},
        "s1_recon": {"loss_type": "continuous", "channels": 2, "weight": 1.0},
        "landsat_recon": {"loss_type": "continuous", "channels": 7, "weight": 1.0},
        "worldcover": {"loss_type": "categorical", "channels": 9, "weight": 0.5},
    }
    head_cfg_for_loss = {
        "s2_recon": {"loss_type": "l1", "channels": 12, "weight": 1.0},
        "s1_recon": {"loss_type": "l1", "channels": 2, "weight": 1.0},
        "landsat_recon": {"loss_type": "l1", "channels": 7, "weight": 1.0},
        "worldcover": {"loss_type": "ce", "channels": 9, "weight": 0.5},
    }
    criterion = TotalLoss(head_cfg_for_loss)

    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(batch, head_cfg_for_prepare)
            output = model(
                source_frames=prepared["source_frames"],
                source_masks=prepared["source_masks"],
                timestamps=prepared["timestamps"],
            )
            losses = criterion(output, prepared["targets"], prepared["target_masks"])

            assert torch.isfinite(losses["total"]), "总损失应为有限值"
            assert output.embedding_map.shape == (
                cfg.batch_size,
                16,
                cfg.patch_size,
                cfg.patch_size,
            )
            assert output.embedding.shape == (cfg.batch_size, 16)

            num_batches += 1
            if num_batches >= 2:
                break

    assert num_batches >= 1, "至少应成功运行一个 batch"
