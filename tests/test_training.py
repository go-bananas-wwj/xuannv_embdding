from __future__ import annotations

# 训练模块单元测试。
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.checkpoint import load_checkpoint, save_checkpoint
from xuannv_embedding.training.losses import (
    TotalLoss,
    batch_uniformity_loss,
    reconstruction_loss,
)
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.training.trainer import Trainer


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


def test_checkpoint_save_load() -> None:
    """验证 checkpoint 保存与加载能正确恢复模型、优化器与 scheduler 状态。"""
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # 给模型参数一个非零状态，便于验证恢复。
    for param in model.parameters():
        param.data.fill_(0.5)
    optimizer.step()
    scheduler.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoints" / "epoch_0.pt"
        metrics = {"loss": 0.123}
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=3, metrics=metrics)
        assert ckpt_path.exists()

        # 新实例加载 checkpoint。
        new_model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=2, gamma=0.9)

        state = load_checkpoint(ckpt_path, new_model, new_optimizer, new_scheduler)

        assert state["epoch"] == 3
        assert state["metrics"] == metrics

        for old_param, new_param in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(old_param, new_param)

        # optimizer 与 scheduler 状态应被恢复。
        old_pg = optimizer.state_dict()["param_groups"][0]
        new_pg = new_optimizer.state_dict()["param_groups"][0]
        assert new_pg["lr"] == old_pg["lr"]
        assert new_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_optimizer_and_scheduler() -> None:
    """验证 optimizer 与带 warmup 的 cosine scheduler 构建正确。"""
    model = nn.Linear(4, 2)
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.05)
    assert isinstance(optimizer, torch.optim.AdamW)

    scheduler = build_scheduler(optimizer, warmup_epochs=2, total_epochs=10)
    # warmup 阶段学习率线性增长；记录 step 之前的学习率以反映当前 epoch 的调度值。
    lrs = []
    for _ in range(10):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    # warmup 期间学习率递增（第 0 个值已是 warmup 起始）。
    assert lrs[0] < lrs[1]
    # warmup 结束后学习率开始下降。
    assert lrs[1] > lrs[-1]


class _DummyAEFModel(nn.Module):
    """用于 trainer 测试的最小化 AEF 风格模型。"""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Conv2d(3, embed_dim, kernel_size=1)
        self.s2_decoder = nn.Conv2d(embed_dim, 10, kernel_size=1)
        self.worldcover_decoder = nn.Conv2d(embed_dim, 11, kernel_size=1)

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frame: torch.Tensor | None = None,
        highres_mask: torch.Tensor | None = None,
    ) -> AEFOutput:
        # 只取 s2 时序最后一帧作为输入，输出固定空间尺寸。
        x = source_frames["s2"][:, -1, :, :, :]  # [B, 10, H, W]
        x = x[:, :3, :, :]  # 裁剪到 3 通道过 encoder
        emb_map = self.encoder(x)
        emb = emb_map.mean(dim=[2, 3])
        return AEFOutput(
            embedding_map=emb_map,
            embedding=emb,
            reconstructions={
                "s2_recon": self.s2_decoder(emb_map),
                "worldcover": self.worldcover_decoder(emb_map),
            },
        )


class _DummyDataset(Dataset):
    """产生 trainer 所需格式 batch 的模拟数据集。"""

    def __init__(self, size: int = 4, height: int = 8, width: int = 8) -> None:
        self.size = size
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source_frames": {
                "s2": torch.randn(2, 10, self.height, self.width),
                "s1": torch.randn(2, 2, self.height, self.width),
            },
            "source_masks": {
                "s2": torch.ones(2),
                "s1": torch.ones(2),
            },
            "timestamps": torch.arange(2).float(),
            "targets": {
                "s2_recon": torch.randn(10, self.height, self.width),
                "worldcover": torch.randint(0, 11, (self.height, self.width)),
            },
            "target_masks": {
                "s2_recon": torch.ones(self.height, self.width),
                "worldcover": torch.ones(self.height, self.width),
            },
        }


def _dummy_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """简单 collate：对 dict 内每个张量按 batch 维度拼接。"""
    result: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        value = batch[0][key]
        if isinstance(value, dict):
            result[key] = {k: torch.stack([b[key][k] for b in batch]) for k in value}
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


def test_trainer_single_epoch() -> None:
    """验证 Trainer 能在 CPU 上用模拟数据跑完 1 个 epoch。"""
    from dataclasses import dataclass

    @dataclass
    class _TrainingConfig:
        epochs: int = 1
        lr: float = 1e-3
        weight_decay: float = 0.0
        warmup_epochs: int = 0
        gradient_accumulation_steps: int = 1
        save_every: int = 1
        eval_every: int = 1

    @dataclass
    class _ExperimentConfig:
        name: str = "test_trainer_single_epoch"

    @dataclass
    class _Config:
        experiment: _ExperimentConfig
        training: _TrainingConfig

    cfg = _Config(
        experiment=_ExperimentConfig(),
        training=_TrainingConfig(),
    )

    model = _DummyAEFModel(embed_dim=16)
    dataset = _DummyDataset(size=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)

    target_cfg = {
        "s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0},
        "worldcover": {"loss_type": "ce", "channels": 11, "weight": 0.5},
    }
    criterion = TotalLoss(target_cfg)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        criterion=criterion,
    )

    metrics = trainer.train_epoch()
    assert "train_loss" in metrics
    assert "total" in metrics
    assert metrics["train_loss"] >= 0.0

    val_metrics = trainer.val_epoch()
    assert val_metrics is not None
    assert "val_loss" in val_metrics

    # 验证 fit 能跑完配置的 1 个 epoch 并保存 checkpoint。
    trainer.fit()
    ckpt_files = list(trainer.output_dir.glob("epoch_*.pt"))
    assert len(ckpt_files) == 1
