from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from xuannv_embedding.training.checkpoint import load_checkpoint, save_checkpoint
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)

# DDP 训练器：支持单卡/多卡训练、checkpoint 保存与自动保留最近 3 个。


class Trainer:
    """AEF 模型训练器。

    负责训练循环、优化器/scheduler 管理、DDP 包装以及 checkpoint 保存。
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device | str | None,
        criterion: nn.Module,
    ) -> None:
        """初始化 Trainer。

        参数:
            cfg: 顶层配置对象，需包含 ``experiment.name`` 与 ``training`` 字段。
            model: 待训练模型。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器，可选。
            device: 训练设备；为 ``None`` 时自动选择。
            criterion: 损失函数模块。
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        if device is None:
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model = self.model.to(self.device)

        # DDP 包装：仅在分布式已初始化时进行。
        if dist.is_initialized():
            device_index = self.device.index if self.device.index is not None else 0
            self.model = DDP(self.model, device_ids=[device_index])

        self.optimizer = build_optimizer(
            self._unwrap_model(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            total_epochs=cfg.training.epochs,
        )

        self.epoch = 0
        self.global_step = 0
        self.gradient_accumulation_steps = getattr(cfg.training, "gradient_accumulation_steps", 1)

        # 输出目录：约定 /data/xuannv_embedding/outputs/{experiment_name}
        self.output_dir = Path("/data/xuannv_embedding/outputs") / cfg.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _unwrap_model(self) -> nn.Module:
        """获取未包装 DDP 的原始模型。"""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """将 batch 中所有 Tensor 移动到训练设备。"""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                moved[key] = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                moved[key] = value
        return moved

    def _forward(self, batch: dict[str, Any]) -> Any:
        """根据 batch 内容调用模型前向。"""
        source_frames = batch["source_frames"]
        source_masks = batch["source_masks"]
        timestamps = batch["timestamps"]
        highres_frame = batch.get("highres_frame")
        highres_mask = batch.get("highres_mask")
        return self.model(
            source_frames=source_frames,
            source_masks=source_masks,
            timestamps=timestamps,
            highres_frame=highres_frame,
            highres_mask=highres_mask,
        )

    def train_epoch(self) -> dict[str, float]:
        """执行一个训练 epoch。

        返回:
            包含 ``train_loss`` 以及各子损失平均值的字典。
        """
        self.model.train()

        total_loss = 0.0
        metric_sums: dict[str, float] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)
            output = self._forward(batch)
            losses = self.criterion(
                output,
                batch["targets"],
                batch["target_masks"],
            )
            loss = losses["total"]

            # 梯度累积：按累积步数缩放损失。
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            batch_loss = losses["total"].item()
            total_loss += batch_loss
            num_batches += 1

            for name, value in losses.items():
                metric_sums.setdefault(name, 0.0)
                metric_sums[name] += value.item()

        # 处理最后未满一个累积步数的梯度。
        if (len(self.train_loader) % self.gradient_accumulation_steps) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        self.scheduler.step()

        metrics: dict[str, float] = {
            name: value / num_batches for name, value in metric_sums.items()
        }
        metrics["train_loss"] = total_loss / num_batches
        return metrics

    @torch.no_grad()
    def val_epoch(self) -> dict[str, float] | None:
        """执行一个验证 epoch。

        返回:
            验证指标字典；若未提供 ``val_loader`` 则返回 ``None``。
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        metric_sums: dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            output = self._forward(batch)
            losses = self.criterion(
                output,
                batch["targets"],
                batch["target_masks"],
            )
            for name, value in losses.items():
                metric_sums.setdefault(name, 0.0)
                metric_sums[name] += value.item()
            num_batches += 1

        metrics: dict[str, float] = {
            name: value / num_batches for name, value in metric_sums.items()
        }
        metrics["val_loss"] = metrics.get("total", 0.0)
        return metrics

    def _cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """仅保留最近的 ``keep_last`` 个 epoch checkpoint。"""
        checkpoints = sorted(self.output_dir.glob("epoch_*.pt"))
        if len(checkpoints) <= keep_last:
            return
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()

    def _is_main_process(self) -> bool:
        """判断当前是否为分布式环境下的主进程。"""
        return not dist.is_initialized() or dist.get_rank() == 0

    def fit(self) -> None:
        """执行完整训练循环。"""
        total_epochs = self.cfg.training.epochs

        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()

            if self._is_main_process():
                log_msg = f"epoch {epoch}: train_loss={train_metrics['train_loss']:.6f}"
                if val_metrics is not None:
                    log_msg += f", val_loss={val_metrics['val_loss']:.6f}"
                logger.info(log_msg)

                # 保存 checkpoint。
                save_checkpoint(
                    self.output_dir / f"epoch_{epoch}.pt",
                    self._unwrap_model(),
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                )
                self._cleanup_old_checkpoints(keep_last=3)

    def save(self, path: str | Path) -> None:
        """手动保存 checkpoint。"""
        save_checkpoint(
            path,
            self._unwrap_model(),
            self.optimizer,
            self.scheduler,
            self.epoch,
            {},
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        """手动加载 checkpoint 并恢复训练状态。"""
        state = load_checkpoint(
            path,
            self._unwrap_model(),
            self.optimizer,
            self.scheduler,
            device=self.device,
        )
        self.epoch = state.get("epoch", 0) + 1
        return state
