from __future__ import annotations

# 训练模块单元测试。
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.models.model import AEFOutput
from xuannv_embedding.training.checkpoint import load_checkpoint, save_checkpoint
from xuannv_embedding.training.losses import (
    TotalLoss,
    batch_uniformity_loss,
    reconstruction_loss,
)
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.training.trainer import Trainer


def test_config_num_months_mismatch(tmp_path: Path) -> None:
    """验证 data.num_months 与 model.num_months 不一致时抛出 ConfigError。"""
    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"

    base.write_text(
        """
experiment:
  name: mismatch_test
data:
  root: /data/xuannv_embedding/processed/base
  region: base
  manifest_path: /data/xuannv_embedding/processed/base/manifest.json
model:
  embed_dim: 64
  sensor_channels:
    s2: 12
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: 12
training:
  epochs: 1
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 0
  gradient_accumulation_steps: 1
  save_every: 1
  eval_every: 1
""",
        encoding="utf-8",
    )
    derived.write_text(
        """
_base_: base.yaml
data:
  num_months: 12
model:
  num_months: 17
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="num_months"):
        Config.from_yaml(derived)


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
    """验证 CE 掩码重建损失计算正确，且 class-0 像素被忽略。"""
    batch_size, num_classes, height, width = 2, 5, 4, 4
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    mask = torch.ones(batch_size, height, width)
    mask[1, :, :2] = 0.0  # 屏蔽第二个样本的左半部分。

    loss = reconstruction_loss(pred, target, mask, loss_type="ce")

    expected = torch.nn.functional.cross_entropy(
        pred, target, ignore_index=0, reduction="none"
    )
    effective_mask = mask * (target != 0).float()
    expected_masked_sum = (expected * effective_mask).sum()
    expected_count = effective_mask.sum()
    expected_loss = expected_masked_sum / expected_count

    assert torch.allclose(loss, expected_loss)


def test_reconstruction_loss_ce_ignores_class_zero() -> None:
    """当 mask 全为 1 时，class-0 像素仍不应影响 CE 重建损失。"""
    batch_size, num_classes, height, width = 1, 3, 4, 4
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    # 确保存在 class-0 像素。
    target[0, 0, 0] = 0
    target[0, 1, 1] = 0
    mask = torch.ones(batch_size, height, width)

    loss = reconstruction_loss(pred, target, mask, loss_type="ce")

    expected = torch.nn.functional.cross_entropy(
        pred, target, ignore_index=0, reduction="none"
    )
    expected_loss = expected.sum() / (target != 0).float().sum()

    assert torch.allclose(loss, expected_loss)


def test_batch_uniformity_loss() -> None:
    """验证 embedding 更分散时 uniformity loss 更小，适合直接最小化。"""
    # 所有向量相同，距离为 0，损失接近 0。
    emb_same = torch.ones(4, 8)
    loss_same = batch_uniformity_loss(emb_same)

    # 随机向量通常更分散，因此 Wang-Isola uniformity loss 更小。
    torch.manual_seed(42)
    emb_random = torch.randn(4, 8)
    loss_random = batch_uniformity_loss(emb_random)

    assert loss_random < loss_same
    # 相同向量损失应接近 0。
    assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6)


def test_total_loss() -> None:
    """验证 TotalLoss 返回正确的各字段（月度输出格式）。"""
    batch_size, num_months, embed_dim, height, width = 2, 3, 16, 8, 8

    reconstructions = {
        "s2_recon": torch.randn(batch_size, num_months, 10, height, width),
        "worldcover": torch.randn(batch_size, num_months, 9, height, width),
    }
    output = AEFOutput(
        embedding_map=torch.randn(batch_size, num_months, embed_dim, height, width),
        embedding=torch.randn(batch_size, num_months, embed_dim),
        reconstructions=reconstructions,
    )

    targets = {
        "s2_recon": torch.randn(batch_size, num_months, 10, height, width),
        "worldcover": torch.randint(0, 9, (batch_size, num_months, height, width)),
    }
    masks = {
        "s2_recon": torch.ones(batch_size, num_months, height, width),
        "worldcover": torch.ones(batch_size, num_months, height, width),
    }

    target_cfg = {
        "s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0},
        "worldcover": {"loss_type": "ce", "channels": 9, "weight": 0.5},
    }

    criterion = TotalLoss(target_cfg)
    losses = criterion(output, targets, masks)

    assert "total" in losses
    assert "recon" in losses
    assert "uniformity" in losses
    assert "recon_s2_recon" in losses
    assert "recon_worldcover" in losses

    expected_total = losses["recon"] + losses["uniformity_weighted"]
    assert torch.allclose(losses["total"], expected_total)

    # recon 应为加权求和。
    expected_recon = 1.0 * losses["recon_s2_recon"] + 0.5 * losses["recon_worldcover"]
    assert torch.allclose(losses["recon"], expected_recon)


def test_reconstruction_loss_temporal() -> None:
    """验证月度 (B, T, C, H, W) 输入下的掩码重建损失计算正确。"""
    batch_size, time_steps, channels, height, width = 2, 3, 4, 4, 4
    pred = torch.randn(batch_size, time_steps, channels, height, width)
    target = torch.randn(batch_size, time_steps, channels, height, width)
    mask = torch.ones(batch_size, time_steps, height, width)
    mask[0, 0, :2, :] = 0.0

    loss = reconstruction_loss(pred, target, mask, loss_type="l1")

    # 手动按时间维度展开后计算。
    pred_flat = pred.reshape(batch_size * time_steps, channels, height, width)
    target_flat = target.reshape(batch_size * time_steps, channels, height, width)
    mask_flat = mask.reshape(batch_size * time_steps, height, width)
    expected = F.l1_loss(pred_flat, target_flat, reduction="none").mean(dim=1)
    expected_loss = (expected * mask_flat).sum() / mask_flat.sum()

    assert torch.allclose(loss, expected_loss)


def test_batch_uniformity_loss_temporal() -> None:
    """验证月度 (B, T, D) embedding 的均匀性损失计算正确。"""
    emb_same = torch.ones(2, 4, 8)
    loss_same = batch_uniformity_loss(emb_same)

    torch.manual_seed(42)
    emb_random = torch.randn(2, 4, 8)
    loss_random = batch_uniformity_loss(emb_random)

    assert loss_random < loss_same
    assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6)


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
        trainer_state = {"best_val_loss": 0.456, "best_epoch": 9}
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch=3,
            metrics=metrics,
            trainer_state=trainer_state,
        )
        assert ckpt_path.exists()

        # 新实例加载 checkpoint。
        new_model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=2, gamma=0.9)

        state = load_checkpoint(ckpt_path, new_model, new_optimizer, new_scheduler)

        assert state["epoch"] == 3
        assert state["metrics"] == metrics
        assert state["trainer_state"] == trainer_state

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
        highres_frames: dict[str, torch.Tensor] | None = None,
        highres_masks: dict[str, torch.Tensor] | None = None,
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


# 模块级最小配置类，避免在函数内部定义 dataclass 时捕获外部变量失败。


@dataclass
class _TrainingConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    gradient_accumulation_steps: int = 1
    save_every: int = 1
    eval_every: int = 1
    amp: bool = False
    gradient_checkpointing: bool = False
    log_every: int = 0


@dataclass
class _ExperimentConfig:
    name: str = "test_trainer"
    use_wandb: bool = False
    wandb_project: str = "test-project"
    wandb_run_name: str | None = None
    output_dir: Path | None = None


@dataclass
class _Config:
    experiment: _ExperimentConfig = field(default_factory=_ExperimentConfig)
    training: _TrainingConfig = field(default_factory=_TrainingConfig)


def _make_trainer_cfg(
    *,
    epochs: int = 1,
    save_every: int = 1,
    eval_every: int = 1,
    amp: bool = False,
    gradient_checkpointing: bool = False,
    log_every: int = 0,
    use_wandb: bool = False,
    experiment_name: str = "test_trainer",
    output_dir: Path | None = None,
) -> Any:
    """构造 Trainer 测试用的最小配置对象。"""
    training = _TrainingConfig()
    training.epochs = epochs
    training.save_every = save_every
    training.eval_every = eval_every
    training.amp = amp
    training.gradient_checkpointing = gradient_checkpointing
    training.log_every = log_every

    experiment = _ExperimentConfig()
    experiment.name = experiment_name
    experiment.use_wandb = use_wandb
    experiment.output_dir = output_dir

    return _Config(experiment=experiment, training=training)


def test_trainer_single_epoch(tmp_path: Path) -> None:
    """验证 Trainer 能在 CPU 上用模拟数据跑完 1 个 epoch。"""
    cfg = _make_trainer_cfg(
        experiment_name="test_trainer_single_epoch",
        output_dir=tmp_path,
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

    assert trainer.amp_enabled is False
    assert trainer.scaler is not None

    metrics = trainer.train_epoch()
    assert "train_loss" in metrics
    assert "total" in metrics
    assert metrics["train_loss"] >= 0.0

    val_metrics = trainer.val_epoch()
    assert val_metrics is not None
    assert "val_loss" in val_metrics

    # 验证 fit 能跑完配置的 1 个 epoch 并保存 1-based checkpoint。
    trainer.fit()
    ckpt_files = list(trainer.output_dir.glob("epoch_*.pt"))
    assert len(ckpt_files) == 1
    assert ckpt_files[0].name == "epoch_1.pt"


def test_trainer_amp_noop_scaler(tmp_path: Path) -> None:
    """AMP 关闭时应使用 no-op scaler，训练仍能正常收敛。"""
    cfg = _make_trainer_cfg(
        epochs=1,
        amp=False,
        experiment_name="test_trainer_amp_noop",
        output_dir=tmp_path,
    )

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)

    target_cfg = {"s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0}}
    criterion = TotalLoss(target_cfg)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )

    from xuannv_embedding.training.amp_utils import _NoOpScaler

    assert isinstance(trainer.scaler, _NoOpScaler)

    before = {name: p.clone() for name, p in model.named_parameters()}
    trainer.train_epoch()
    after = {name: p.clone() for name, p in model.named_parameters()}

    changed = any(not torch.allclose(before[name], after[name]) for name in before)
    assert changed, "AMP 关闭时参数应被更新"


def test_trainer_wandb_logging(monkeypatch, tmp_path: Path) -> None:
    """验证 use_wandb=True 时会按预期调用 wandb.log。"""
    logged: list[dict[str, Any]] = []
    init_calls: list[dict[str, Any]] = []

    class _FakeRun:
        def __init__(self, **kwargs: Any) -> None:
            init_calls.append(kwargs)

        def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
            logged.append({"metrics": metrics, "step": step})

        def finish(self) -> None:
            pass

    fake_wandb = type("FakeWandb", (), {"init": _FakeRun})

    cfg = _make_trainer_cfg(
        epochs=1,
        log_every=1,
        use_wandb=True,
        experiment_name="test_trainer_wandb",
        output_dir=tmp_path,
    )

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)

    target_cfg = {"s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0}}
    criterion = TotalLoss(target_cfg)

    # 通过 monkeypatch 直接替换 sys.modules 中的 wandb。
    import sys

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )

    trainer.train_epoch()

    assert len(init_calls) == 1
    assert init_calls[0]["project"] == "test-project"
    assert any("train/loss_total" in entry["metrics"] for entry in logged)
    assert any("train/loss_recon" in entry["metrics"] for entry in logged)
    assert any("train/loss_uniformity" in entry["metrics"] for entry in logged)
    assert any("train/lr" in entry["metrics"] for entry in logged)


def test_trainer_utilization_logging(monkeypatch, tmp_path: Path) -> None:
    """验证设备利用率指标会被正确命名并记录到 WANDB。"""
    logged: list[dict[str, Any]] = []

    class _FakeRun:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
            logged.append(metrics)

        def finish(self) -> None:
            pass

    fake_wandb = type("FakeWandb", (), {"init": _FakeRun})

    cfg = _make_trainer_cfg(
        epochs=1,
        use_wandb=True,
        experiment_name="test_trainer_util",
        output_dir=tmp_path,
    )

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)
    target_cfg = {"s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0}}
    criterion = TotalLoss(target_cfg)

    import sys

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )

    # 模拟 CPU 上不可用的利用率返回 None；通过 monkeypatch 模拟 NPU/CUDA 返回值。
    monkeypatch.setattr(trainer, "_get_utilization", lambda: ("system/npu_utilization", 42.0))
    trainer.train_epoch()

    assert any("system/npu_utilization" in entry for entry in logged)
    assert any(entry.get("system/npu_utilization") == 42.0 for entry in logged)


def test_stp_encoder_gradient_checkpointing() -> None:
    """验证 STPEncoder 在 gradient_checkpointing=True 时仍能完成前向/反向传播。"""
    from xuannv_embedding.models.blocks import STPEncoder

    enc = STPEncoder(
        input_channels=16,
        space_dim=64,
        time_dim=32,
        precision_dim=16,
        num_blocks=2,
        num_heads=4,
        gradient_checkpointing=True,
    )
    enc.train()
    x = torch.randn(1, 2, 32, 32, 16, requires_grad=True)
    timestamps = torch.tensor([[202501, 202502]])
    out, _ = enc(x, timestamps)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_trainer_checkpoint_best_and_latest3(tmp_path: Path) -> None:
    """验证仅保存 save_every 倍数的 epoch，并保留 best.pt 与最近 3 个 epoch 权重。"""
    cfg = _make_trainer_cfg(
        epochs=6,
        save_every=2,
        eval_every=1,
        experiment_name="test_trainer_ckpt",
    )

    class _DummyLossWithControl(nn.Module):
        """可控制损失值的假损失，用于构造最佳 epoch；损失依赖模型输出保证可梯度。"""

        def __init__(self) -> None:
            super().__init__()
            self._val_loss = 1.0

        def set_val_loss(self, value: float) -> None:
            self._val_loss = value

        def forward(self, output: AEFOutput, targets: Any, masks: Any) -> dict[str, torch.Tensor]:
            base = output.embedding.mean() * 0.0 + self._val_loss
            return {
                "total": base,
                "recon": base * 0.8,
                "uniformity": base * 0.2,
                "recon_s2_recon": base * 0.8,
            }

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)

    criterion = _DummyLossWithControl()

    # 覆盖输出目录到 tmp_path，避免污染数据盘。
    cfg.experiment.output_dir = tmp_path

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        criterion=criterion,
    )

    for epoch in range(6):
        criterion.set_val_loss(1.0 - epoch * 0.1)  # 损失递减，最后 epoch 最佳。
        trainer.train_epoch()
        trainer.val_epoch()
        if trainer._is_main_process() and trainer._should_save_checkpoint(epoch, 6):
            one_based = epoch + 1
            epoch_path = trainer.output_dir / f"epoch_{one_based}.pt"
            save_checkpoint(
                epoch_path,
                trainer._unwrap_model(),
                trainer.optimizer,
                trainer.scheduler,
                epoch,
                {"train": {"train_loss": 0.0}, "val": {"val_loss": 1.0 - epoch * 0.1}},
            )
            current_loss = 1.0 - epoch * 0.1
            if current_loss < trainer.best_val_loss:
                trainer.best_val_loss = current_loss
                trainer.best_epoch = epoch
                shutil.copy2(epoch_path, trainer.output_dir / "best.pt")
            trainer._cleanup_old_checkpoints(keep_last=3)

    epoch_files = sorted(trainer.output_dir.glob("epoch_*.pt"))
    epoch_names = {p.name for p in epoch_files}

    # save_every=2，1-based epochs 1..6，保存 epoch_2, epoch_4, epoch_6（最后 epoch）。
    # 经过 keep_last=3 清理后全部保留。
    assert "epoch_2.pt" in epoch_names
    assert "epoch_4.pt" in epoch_names
    assert "epoch_6.pt" in epoch_names
    assert "epoch_1.pt" not in epoch_names
    assert "epoch_3.pt" not in epoch_names
    assert "epoch_5.pt" not in epoch_names
    assert len(epoch_files) == 3

    # best.pt 必须存在且指向损失最小的 epoch_6（原始 epoch index 5）。
    best_path = trainer.output_dir / "best.pt"
    assert best_path.exists()
    state = torch.load(best_path, weights_only=True)
    assert state["epoch"] == 5


def test_cleanup_old_checkpoints_ignores_malformed_names(tmp_path: Path) -> None:
    """验证 _cleanup_old_checkpoints 严格匹配 epoch_N.pt，忽略畸形文件名。"""
    cfg = _make_trainer_cfg(
        epochs=1,
        experiment_name="test_cleanup_regex",
        output_dir=tmp_path,
    )

    model = _DummyAEFModel(embed_dim=4)
    dataset = _DummyDataset(size=1)
    loader = DataLoader(dataset, batch_size=1, collate_fn=_dummy_collate)
    target_cfg = {"s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0}}
    criterion = TotalLoss(target_cfg)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )

    # 创建符合与不符合命名规则的文件。
    (tmp_path / "epoch_1.pt").touch()
    (tmp_path / "epoch_2.pt").touch()
    (tmp_path / "epoch_3.pt").touch()
    (tmp_path / "epoch_10.pt").touch()
    (tmp_path / "epoch_.pt").touch()
    (tmp_path / "epoch_not_a_number.pt").touch()
    (tmp_path / "best.pt").touch()

    trainer._cleanup_old_checkpoints(keep_last=3)

    remaining = {p.name for p in tmp_path.glob("*.pt")}
    # 严格匹配 epoch_N.pt 的文件按编号保留最近 3 个；畸形文件名被跳过（不删除也不计入）。
    assert "epoch_1.pt" not in remaining
    assert "epoch_2.pt" in remaining
    assert "epoch_3.pt" in remaining
    assert "epoch_10.pt" in remaining
    assert "best.pt" in remaining
    assert "epoch_.pt" in remaining
    assert "epoch_not_a_number.pt" in remaining


def test_trainer_load_restores_best_state(tmp_path: Path) -> None:
    """验证 Trainer.load() 能恢复 best_val_loss 与 best_epoch。"""
    cfg = _make_trainer_cfg(
        epochs=1,
        experiment_name="test_trainer_resume",
        output_dir=tmp_path,
    )

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)
    target_cfg = {"s2_recon": {"loss_type": "l1", "channels": 10, "weight": 1.0}}
    criterion = TotalLoss(target_cfg)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )

    # 手动设置并保存最佳状态与 epoch。
    trainer.epoch = 7
    trainer.best_val_loss = 0.123
    trainer.best_epoch = 7
    ckpt_path = tmp_path / "resume.pt"
    trainer.save(ckpt_path)

    # 新 Trainer 实例加载后应恢复该状态。
    new_model = _DummyAEFModel(embed_dim=8)
    new_trainer = Trainer(
        cfg=cfg,
        model=new_model,
        train_loader=loader,
        val_loader=None,
        device="cpu",
        criterion=criterion,
    )
    new_trainer.load(ckpt_path)

    assert new_trainer.best_val_loss == 0.123
    assert new_trainer.best_epoch == 7
    assert new_trainer.epoch == 8  # epoch = saved_epoch + 1


class _DecreasingDummyLoss(nn.Module):
    """每次调用都返回递减损失的假损失，用于验证 best.pt 选择逻辑。"""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def forward(self, output: AEFOutput, targets: Any, masks: Any) -> dict[str, torch.Tensor]:
        self._call_count += 1
        loss_val = max(0.01, 1.0 - self._call_count * 0.01)
        base = output.embedding.mean() * 0.0 + loss_val
        return {
            "total": base,
            "recon": base * 0.8,
            "uniformity": base * 0.2,
            "recon_s2_recon": base * 0.8,
        }


def test_trainer_best_pt_when_save_and_eval_aligned(tmp_path: Path) -> None:
    """当 save_every 与 eval_every 相同时，fit 应在保存点生成 best.pt。

    此测试用于捕获 eval 使用 0-based、save 使用 1-based 导致保存点无 val_metrics、
    从而 best.pt 从未生成的调度不对齐 bug。
    """
    cfg = _make_trainer_cfg(
        epochs=6,
        save_every=2,
        eval_every=2,
        experiment_name="test_trainer_best_aligned",
    )
    cfg.experiment.output_dir = tmp_path

    model = _DummyAEFModel(embed_dim=8)
    dataset = _DummyDataset(size=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=_dummy_collate)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        criterion=_DecreasingDummyLoss(),
    )

    trainer.fit()

    best_path = tmp_path / "best.pt"
    assert best_path.exists(), "best.pt 应在保存点存在"
    state = torch.load(best_path, weights_only=True)
    # 最后一个保存点 epoch index 5（1-based epoch 6）损失最低。
    assert state["epoch"] == 5
