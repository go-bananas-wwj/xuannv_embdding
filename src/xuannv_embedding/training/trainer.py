from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from xuannv_embedding.training.amp_utils import get_autocast, get_grad_scaler
from xuannv_embedding.training.checkpoint import load_checkpoint, save_checkpoint
from xuannv_embedding.training.optimizer import build_optimizer, build_scheduler
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)

# DDP 训练器：支持单卡/多卡训练、AMP、梯度检查点、WANDB 监控与 checkpoint 管理。


class Trainer:
    """AEF 模型训练器。

    负责训练循环、优化器/scheduler 管理、DDP 包装、AMP、梯度检查点、
    WANDB 监控以及 best/latest-3 checkpoint 保存。
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device | str | None,
        criterion: nn.Module,
        train_sampler: DistributedSampler | None = None,
    ) -> None:
        """初始化 Trainer。

        参数:
            cfg: 顶层配置对象，需包含 ``experiment`` 与 ``training`` 字段。
            model: 待训练模型。
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器，可选。
            device: 训练设备；为 ``None`` 时自动选择。
            criterion: 损失函数模块。
            train_sampler: 训练集的 DistributedSampler；提供时每个 epoch 调用
                ``set_epoch`` 以保证打乱顺序的正确性。
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.train_sampler = train_sampler

        if device is None:
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model = self.model.to(self.device)

        # DDP 包装：仅在分布式已初始化时进行；CPU 后端不传递 device_ids。
        if dist.is_initialized():
            device_index = self.device.index if self.device.index is not None else 0
            ddp_kwargs: dict[str, Any] = {}
            if self.device.type in ("cuda", "npu"):
                ddp_kwargs["device_ids"] = [device_index]
            self.model = DDP(self.model, **ddp_kwargs)

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

        # 输出目录：优先使用 experiment.output_dir，否则约定为
        # /data/xuannv_embedding/outputs/{experiment_name}
        output_dir = getattr(cfg.experiment, "output_dir", None)
        if output_dir is None:
            output_dir = Path("/data/xuannv_embedding/outputs") / cfg.experiment.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AMP 设置。
        self.amp_enabled = getattr(cfg.training, "amp", True)
        self.autocast = get_autocast(self.device, self.amp_enabled)
        self.scaler = get_grad_scaler(self.device, self.amp_enabled)

        # WANDB 设置。
        self.use_wandb = getattr(cfg.experiment, "use_wandb", True)
        self.wandb_project = getattr(
            cfg.experiment, "wandb_project", "xuannv-embedding-stage1"
        )
        self.wandb_run_name = getattr(cfg.experiment, "wandb_run_name", None)
        self._wandb_run: Any = None
        if self.use_wandb and self._is_main_process():
            self._init_wandb()

        # 最佳验证指标跟踪。
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _init_wandb(self) -> None:
        """在主进程初始化 WANDB run；未安装或未认证时不中断训练。"""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb 未安装，跳过 WANDB 初始化")
            return

        try:
            config_dict: dict[str, Any] | None = None
            if hasattr(self.cfg, "to_dict"):
                config_dict = self.cfg.to_dict()
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=config_dict,
            )
            logger.info("WANDB 已初始化: project=%s", self.wandb_project)
        except Exception as exc:
            logger.warning("WANDB 初始化失败: %s", exc)
            self._wandb_run = None

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
        highres_frames = batch.get("highres_frames")
        highres_masks = batch.get("highres_masks")
        return self.model(
            source_frames=source_frames,
            source_masks=source_masks,
            timestamps=timestamps,
            highres_frames=highres_frames,
            highres_masks=highres_masks,
        )

    def _log_to_wandb(self, metrics: dict[str, float], step: int | None = None) -> None:
        """在主进程向 WANDB 发送指标。"""
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(metrics, step=step)
        except Exception as exc:
            logger.warning("WANDB log 失败: %s", exc)

    def _get_memory_mb(self) -> float | None:
        """获取当前设备的已分配显存/NPU 内存（MB）。"""
        try:
            if self.device.type == "npu":
                return torch.npu.memory_allocated(self.device) / 1e6
            if self.device.type == "cuda":
                return torch.cuda.memory_allocated(self.device) / 1e6
        except Exception:
            pass
        return None

    def _get_utilization(self) -> tuple[str, float] | None:
        """获取当前设备的利用率，返回 (metric_name, value) 或 None。"""
        try:
            if self.device.type == "npu":
                util = torch.npu.utilization(self.device)
                return ("system/npu_utilization", float(util))
            if self.device.type == "cuda":
                util = torch.cuda.utilization(self.device)
                return ("system/gpu_utilization", float(util))
        except Exception:
            pass
        return None

    def train_epoch(self) -> dict[str, float]:
        """执行一个训练 epoch。

        返回:
            包含 ``train_loss`` 以及各子损失平均值的字典。
        """
        self.model.train()

        total_loss = 0.0
        metric_sums: dict[str, float] = {}
        num_batches = 0
        log_every = getattr(self.cfg.training, "log_every", 0)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            with self.autocast:
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

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            batch_loss = losses["total"].item()
            total_loss += batch_loss
            num_batches += 1

            for name, value in losses.items():
                metric_sums.setdefault(name, 0.0)
                metric_sums[name] += value.item()

            # 按 step 记录 WANDB 训练指标。
            if (
                log_every > 0
                and self.global_step > 0
                and self.global_step % log_every == 0
                and (batch_idx + 1) % self.gradient_accumulation_steps == 0
            ):
                step_metrics: dict[str, float] = {
                    "train/loss_total": losses["total"].item(),
                    "train/loss_recon": losses["recon"].item(),
                    "train/loss_uniformity": losses["uniformity"].item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                }
                for name, value in losses.items():
                    if name.startswith("recon_"):
                        step_metrics[f"train/{name}"] = value.item()
                memory_mb = self._get_memory_mb()
                if memory_mb is not None:
                    step_metrics["system/npu_memory_allocated_MB"] = memory_mb
                util_info = self._get_utilization()
                if util_info is not None:
                    step_metrics[util_info[0]] = util_info[1]
                self._log_to_wandb(step_metrics, step=self.global_step)

        # 处理最后未满一个累积步数的梯度。
        if (len(self.train_loader) % self.gradient_accumulation_steps) != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        self.scheduler.step()

        metrics: dict[str, float] = {
            name: value / num_batches for name, value in metric_sums.items()
        }
        metrics["train_loss"] = total_loss / num_batches

        # 每轮结束时在主进程记录 epoch 级训练指标。
        if self._is_main_process():
            epoch_metrics: dict[str, float] = {
                "train/loss_total": metrics["total"],
                "train/loss_recon": metrics["recon"],
                "train/loss_uniformity": metrics["uniformity"],
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }
            for name, value in metrics.items():
                if name.startswith("recon_"):
                    epoch_metrics[f"train/{name}"] = value
            memory_mb = self._get_memory_mb()
            if memory_mb is not None:
                epoch_metrics["system/npu_memory_allocated_MB"] = memory_mb
            util_info = self._get_utilization()
            if util_info is not None:
                epoch_metrics[util_info[0]] = util_info[1]
            self._log_to_wandb(epoch_metrics, step=self.epoch)

        return metrics

    @torch.no_grad()
    def val_epoch(self) -> dict[str, float] | None:
        """执行一个验证 epoch。

        返回:
            验证指标字典；若非评估 epoch 或 ``val_loader`` 未提供则返回 ``None``。
        """
        if self.val_loader is None:
            return None

        eval_every = getattr(self.cfg.training, "eval_every", 1)
        if eval_every <= 0 or self.epoch % eval_every != 0:
            return None

        self.model.eval()
        metric_sums: dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)

            with self.autocast:
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

        if self._is_main_process():
            val_metrics: dict[str, float] = {"val/loss_total": metrics["val_loss"]}
            for name, value in metrics.items():
                if name.startswith("recon_"):
                    val_metrics[f"val/{name}"] = value
            self._log_to_wandb(val_metrics, step=self.epoch)

        return metrics

    _EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)\.pt$")

    def _cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """仅保留 1-based epoch 编号最大的 ``keep_last`` 个 checkpoint，不删除 best.pt。"""
        checkpoints: list[tuple[int, Path]] = []
        for path in self.output_dir.glob("epoch_*.pt"):
            match = self._EPOCH_FILE_RE.match(path.name)
            if not match:
                continue
            checkpoints.append((int(match.group(1)), path))

        if len(checkpoints) <= keep_last:
            return

        checkpoints.sort(key=lambda item: item[0])
        for _, old_ckpt in checkpoints[:-keep_last]:
            if old_ckpt.name == "best.pt":
                continue
            old_ckpt.unlink()

    def _is_main_process(self) -> bool:
        """判断当前是否为分布式环境下的主进程。"""
        return not dist.is_initialized() or dist.get_rank() == 0

    def _should_save_checkpoint(self, epoch: int, total_epochs: int) -> bool:
        """判断是否应在当前 epoch 保存 checkpoint；使用 1-based epoch 编号。"""
        save_every = getattr(self.cfg.training, "save_every", 1)
        one_based_epoch = epoch + 1
        is_save_epoch = save_every > 0 and (one_based_epoch % save_every == 0)
        is_last_epoch = epoch == total_epochs - 1
        return is_save_epoch or is_last_epoch

    def fit(self) -> None:
        """执行完整训练循环。"""
        total_epochs = self.cfg.training.epochs

        try:
            for epoch in range(self.epoch, total_epochs):
                self.epoch = epoch
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                train_metrics = self.train_epoch()
                val_metrics = self.val_epoch()

                if self._is_main_process():
                    log_msg = f"epoch {epoch}: train_loss={train_metrics['train_loss']:.6f}"
                    if val_metrics is not None:
                        log_msg += f", val_loss={val_metrics['val_loss']:.6f}"
                    logger.info(log_msg)

                    # 保存 checkpoint。
                    if self._should_save_checkpoint(epoch, total_epochs):
                        one_based_epoch = epoch + 1
                        epoch_path = self.output_dir / f"epoch_{one_based_epoch}.pt"
                        save_checkpoint(
                            epoch_path,
                            self._unwrap_model(),
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            {"train": train_metrics, "val": val_metrics},
                            trainer_state={
                                "best_val_loss": self.best_val_loss,
                                "best_epoch": self.best_epoch,
                            },
                        )

                        # 更新 best.pt。
                        current_loss: float | None = None
                        if val_metrics is not None:
                            current_loss = val_metrics["val_loss"]
                        elif self.val_loader is None:
                            current_loss = train_metrics["train_loss"]

                        if current_loss is not None and current_loss < self.best_val_loss:
                            self.best_val_loss = current_loss
                            self.best_epoch = epoch
                            best_path = self.output_dir / "best.pt"
                            shutil.copy2(epoch_path, best_path)
                            logger.info(
                                "新的最佳 checkpoint: epoch=%d, loss=%.6f", epoch, current_loss
                            )

                        self._cleanup_old_checkpoints(keep_last=3)
        finally:
            self._finish_wandb()

    def _finish_wandb(self) -> None:
        """在主进程结束 WANDB run。"""
        if self._wandb_run is None:
            return
        if not self._is_main_process():
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:
            logger.warning("WANDB finish 失败: %s", exc)
        self._wandb_run = None

    def save(self, path: str | Path) -> None:
        """手动保存 checkpoint。"""
        save_checkpoint(
            path,
            self._unwrap_model(),
            self.optimizer,
            self.scheduler,
            self.epoch,
            {},
            trainer_state={
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
            },
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
        trainer_state = state.get("trainer_state", {}) or {}
        self.best_val_loss = trainer_state.get("best_val_loss", float("inf"))
        self.best_epoch = trainer_state.get("best_epoch", -1)
        return state
