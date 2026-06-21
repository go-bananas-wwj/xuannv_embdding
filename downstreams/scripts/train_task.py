#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.distributed as dist
from downstreams.data.end_to_end_dataset import (
    EndToEndSegmentationDataset,
    JointEndToEndSegmentationDataset,
    collate_end_to_end,
)
from downstreams.data.multi_task_dataset import (
    MultiTaskEmbeddingDataset,
    collate_embeddings,
)
from downstreams.data.split import create_combined_stratified_folds, create_stratified_folds
from downstreams.models.end_to_end_model import build_end_to_end_model
from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask
from downstreams.utils.config import load_config
from downstreams.utils.ddp import cleanup_ddp, setup_ddp
from downstreams.utils.device import get_downstream_device
from downstreams.utils.reproducibility import set_seed
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class JointMultiTaskEmbeddingDataset(Dataset[Any]):
    """跨 region 的多任务数据集封装。

    将 ``{region}_{patch_id}`` 形式的前缀 patch_id 路由到对应 region 的
    ``MultiTaskEmbeddingDataset`` 子实例。
    """

    def __init__(
        self,
        embedding_root: Path,
        mask_dirs: dict[str, Path],
        patch_ids: list[str],
        region_of: dict[str, str],
        task_name: str = "construction",
        months: list[str] | None = None,
        augment: bool = False,
        bitemporal: bool = False,
        include_diff: bool = True,
        crop_size: int | None = None,
    ) -> None:
        self.patch_ids = patch_ids
        self.region_of = region_of
        self.task_name = task_name
        self.months = months if months is not None else ["202605"]
        self.augment = augment
        self.bitemporal = bitemporal
        self.include_diff = include_diff
        self.crop_size = crop_size

        self._region_datasets: dict[str, MultiTaskEmbeddingDataset] = {}
        self._region_patch_ids: dict[str, list[str]] = {}

        for region in sorted(set(region_of.values())):
            region_patch_ids = [pid for pid in patch_ids if region_of.get(pid) == region]
            if not region_patch_ids:
                continue
            original_ids = [self._strip_region_prefix(pid, region) for pid in region_patch_ids]
            self._region_datasets[region] = MultiTaskEmbeddingDataset(
                embedding_root=embedding_root / region,
                label_root=mask_dirs[region].parent,
                patch_ids=original_ids,
                task_name=task_name,
                months=self.months,
                augment=augment,
                bitemporal=bitemporal,
                include_diff=include_diff,
                crop_size=crop_size,
            )
            self._region_patch_ids[region] = region_patch_ids

    @staticmethod
    def _strip_region_prefix(patch_id: str, region: str) -> str:
        prefix = f"{region}_"
        if patch_id.startswith(prefix):
            return patch_id[len(prefix) :]
        raise ValueError(f"patch_id {patch_id} 不包含 region 前缀 {prefix}")

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefixed_id = self.patch_ids[idx]
        region = self.region_of[prefixed_id]
        ds = self._region_datasets[region]
        local_id = self._strip_region_prefix(prefixed_id, region)
        local_idx = ds.patch_ids.index(local_id)
        sample = ds[local_idx]
        sample["patch_id"] = prefixed_id
        return sample


def _has_embedding(emb_root: Path, patch_id: str, months: list[str]) -> bool:
    """检查 patch_id 在 emb_root 下是否包含所有请求月份的 embedding。"""
    return all((emb_root / patch_id / f"{m}_embedding_map.pt").exists() for m in months)


def _has_image(image_root: Path, patch_id: str, months: list[str]) -> bool:
    """检查 patch_id 在 image_root 下是否包含所有请求月份的高分影像。"""
    return all(
        any(image_root.glob(f"highres_optical_{month}*_{patch_id}.tif"))
        for month in months
    )


def _has_mask(mask_dir: Path, patch_id: str, months: list[str]) -> bool:
    """检查 patch_id 在 mask_dir 下是否包含任一请求月份的 mask。"""
    return any(
        (mask_dir / f"{patch_id}_{m}.tif").exists() for m in months
    ) or (mask_dir / f"{patch_id}.tif").exists()


def _filter_patch_ids(
    patch_ids: list[str],
    emb_region_root: Path,
    months: list[str],
    mask_dir: Path | None = None,
    region_of: dict[str, str] | None = None,
    mask_dirs: dict[str, Path] | None = None,
) -> list[str]:
    """过滤掉缺失对应月份 embedding 或 mask 的 patch_id。"""
    if region_of is None:
        assert mask_dir is not None
        return [
            pid
            for pid in patch_ids
            if _has_embedding(emb_region_root, pid, months)
            and _has_mask(mask_dir, pid, months)
        ]

    def _check(pid: str) -> bool:
        region = region_of.get(pid)
        if region is None or mask_dirs is None:
            return False
        base_id = pid[len(f"{region}_") :]
        return _has_embedding(
            emb_region_root / region, base_id, months
        ) and _has_mask(mask_dirs[region], base_id, months)

    return [pid for pid in patch_ids if _check(pid)]


def _resolve_mask_path(mask_dir: Path, patch_id: str, months: list[str]) -> Path:
    """优先按月份后缀、再按无后缀查找 mask 路径。"""
    for month in months:
        path = mask_dir / f"{patch_id}_{month}.tif"
        if path.exists():
            return path
    path = mask_dir / f"{patch_id}.tif"
    if not path.exists():
        raise FileNotFoundError(f"mask 不存在: {path}")
    return path


def save_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_dir: Path,
    mask_dir: Path,
    months: list[str],
    region_of: dict[str, str] | None = None,
    mask_dirs: dict[str, Path] | None = None,
) -> None:
    """将测试集概率图保存为 GeoTIFF，便于后续可视化与溯源。"""
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            if "embedding_map" in batch:
                inp = batch["embedding_map"].to(device)
            else:
                inp = batch["image"].to(device)
            patch_ids = batch["patch_ids"]
            logits = model(inp)[:, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            for b, patch_id in enumerate(patch_ids):
                if region_of is not None and mask_dirs is not None:
                    region = region_of[patch_id]
                    base_id = patch_id[len(f"{region}_") :]
                    current_mask_dir = mask_dirs[region]
                    out_patch_id = base_id
                else:
                    current_mask_dir = mask_dir
                    out_patch_id = patch_id

                mask_path = _resolve_mask_path(current_mask_dir, out_patch_id, months)
                with rasterio.open(mask_path) as src:
                    profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress="lzw",
                    nodata=None,
                )
                out_path = pred_dir / f"{out_patch_id}_prob.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(probs[b].astype(np.float32), 1)


def _average_metrics(metrics: dict[str, float], world_size: int) -> dict[str, float]:
    """在 DDP 环境下将所有 rank 的标量指标 all-reduce 并取平均。

    返回的 dict 在所有 rank 上一致。
    """
    if world_size <= 1:
        return metrics

    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float32, device="npu")
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / world_size
    return {k: float(v) for k, v in zip(keys, values.cpu().tolist())}


def _compute_sample_weights(
    dataset: Dataset[Any], pos_weight: float, strategy: str = "ratio"
) -> torch.Tensor:
    """根据每张图的前景像素占比计算 WeightedRandomSampler 权重。

    strategy="ratio": 含前景的样本按 ``1 + pos_weight * positive_ratio`` 加权。
    strategy="binary": 含前景的样本统一按 ``pos_weight`` 加权，全背景样本权重为 1，
    适用于正像素比例极低、需要强制过采样正样本 patch 的任务。
    """
    weights = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample["mask"]
        if isinstance(mask, torch.Tensor):
            pos_ratio = float(mask.float().mean())
        else:
            pos_ratio = float(np.mean(mask))
        if strategy == "binary":
            weights.append(float(pos_weight) if pos_ratio > 0 else 1.0)
        else:
            weights.append(1.0 + pos_weight * pos_ratio)
    return torch.tensor(weights, dtype=torch.float64)


def _find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """在验证集上搜索使 F1 最大的概率阈值。"""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if "embedding_map" in batch:
                inp = batch["embedding_map"].to(device)
            else:
                inp = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(inp)[:, 1]
            all_logits.append(logits.cpu())
            all_masks.append(mask.cpu())
    if not all_logits:
        return 0.5
    logits = torch.cat([x.flatten() for x in all_logits]).numpy()
    targets = torch.cat([x.flatten() for x in all_masks]).numpy()
    metrics = compute_segmentation_metrics(logits, targets, return_curve=True)
    p_arr = metrics["precision_curve"]
    r_arr = metrics["recall_curve"]
    thresholds = metrics["thresholds"]
    if len(thresholds) == 0:
        return 0.5
    f1s = 2 * p_arr[:-1] * r_arr[:-1] / (p_arr[:-1] + r_arr[:-1] + 1e-8)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def run_fold(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    fold_info: dict[str, Any],
    *,
    emb_region_root: Path,
    mask_dir: Path,
    mask_dirs: dict[str, Path] | None,
    region: str,
    bitemporal: bool,
    fractions: dict[str, Any] | None = None,
    region_of: dict[str, str] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any] | None:
    """训练并评测单个 fold，返回测试指标字典。

    该函数可被 ``benchmark_heads.py`` 直接导入复用，从而避免启动子进程与 DDP 开销。
    """
    fold_idx = fold_info["fold"]
    if rank == 0:
        logger.info("===== Fold %d =====", fold_idx)
    out_dir = args.output_root / f"fold_{fold_idx}"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    cfg["training"]["months"] = args.months
    task = ConstructionSegmentationTask(cfg)
    device = get_downstream_device(cfg["experiment"].get("device", "auto"))

    end_to_end = cfg["training"].get("end_to_end", False)
    backbone_lr = cfg["training"].get("backbone_lr", 1.0e-5)
    head_lr = cfg["training"].get("head_lr", 1.0e-4)
    freeze_backbone_epochs = cfg["training"].get("freeze_backbone_epochs", 0)

    if end_to_end and args.regions:
        raise NotImplementedError("端到端训练暂不支持联合多区域模式")

    train_ids = fold_info["train"]
    if args.fraction is not None and fractions is not None:
        frac_str = str(args.fraction)
        train_ids = fractions.get(frac_str, {}).get(f"fold_{fold_idx}", train_ids)

    if args.regions:
        region_of = region_of if region_of is not None else {}
        train_ids = _filter_patch_ids(
            train_ids,
            emb_region_root,
            args.months,
            region_of=region_of,
            mask_dirs=mask_dirs,
        )
        val_ids = _filter_patch_ids(
            fold_info["val"],
            emb_region_root,
            args.months,
            region_of=region_of,
            mask_dirs=mask_dirs,
        )
        test_ids = _filter_patch_ids(
            fold_info["test"],
            emb_region_root,
            args.months,
            region_of=region_of,
            mask_dirs=mask_dirs,
        )
        ds_kwargs = {
            "mask_dirs": mask_dirs,
            "region_of": region_of,
            "task_name": args.task,
            "months": args.months,
            "bitemporal": bitemporal,
            "include_diff": True,
            "crop_size": cfg["training"].get("crop_size"),
        }
        train_ds = JointMultiTaskEmbeddingDataset(
            emb_region_root, patch_ids=train_ids, augment=True, **ds_kwargs
        )
        val_ds = JointMultiTaskEmbeddingDataset(
            emb_region_root, patch_ids=val_ids, augment=False, **ds_kwargs
        )
        test_ds = JointMultiTaskEmbeddingDataset(
            emb_region_root, patch_ids=test_ids, augment=False, **ds_kwargs
        )
    elif end_to_end:
        label_root = args.label_root / args.task
        image_root = label_root.parent.parent / "patches" / "highres_optical"
        train_ids = [
            pid
            for pid in train_ids
            if _has_image(image_root, pid, args.months)
            and _has_mask(mask_dir, pid, args.months)
        ]
        val_ids = [
            pid
            for pid in fold_info["val"]
            if _has_image(image_root, pid, args.months)
            and _has_mask(mask_dir, pid, args.months)
        ]
        test_ids = [
            pid
            for pid in fold_info["test"]
            if _has_image(image_root, pid, args.months)
            and _has_mask(mask_dir, pid, args.months)
        ]
        ds_kwargs = {
            "task_name": args.task,
            "months": args.months,
            "bitemporal": bitemporal,
            "include_diff": True,
            "crop_size": cfg["training"].get("crop_size"),
        }
        train_ds = EndToEndSegmentationDataset(
            image_root, label_root, train_ids, augment=True, **ds_kwargs
        )
        val_ds = EndToEndSegmentationDataset(
            image_root, label_root, val_ids, augment=False, **ds_kwargs
        )
        test_ds = EndToEndSegmentationDataset(
            image_root, label_root, test_ids, augment=False, **ds_kwargs
        )
    else:
        label_root = args.label_root / args.task
        train_ids = _filter_patch_ids(
            train_ids, emb_region_root, args.months, mask_dir=mask_dir
        )
        val_ids = _filter_patch_ids(
            fold_info["val"], emb_region_root, args.months, mask_dir=mask_dir
        )
        test_ids = _filter_patch_ids(
            fold_info["test"], emb_region_root, args.months, mask_dir=mask_dir
        )
        ds_kwargs = {
            "task_name": args.task,
            "months": args.months,
            "bitemporal": bitemporal,
            "include_diff": True,
            "crop_size": cfg["training"].get("crop_size"),
        }
        train_ds = MultiTaskEmbeddingDataset(
            emb_region_root, label_root, train_ids, augment=True, **ds_kwargs
        )
        val_ds = MultiTaskEmbeddingDataset(
            emb_region_root, label_root, val_ids, augment=False, **ds_kwargs
        )
        test_ds = MultiTaskEmbeddingDataset(
            emb_region_root, label_root, test_ids, augment=False, **ds_kwargs
        )

    use_weighted_sampler = cfg["training"].get("use_weighted_sampler", False)
    train_sampler: DistributedSampler | WeightedRandomSampler | None = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
    elif use_weighted_sampler:
        weights = _compute_sample_weights(
            train_ds,
            cfg["training"].get("pos_weight", 1.0),
            cfg["training"].get("sampler_weight_strategy", "ratio"),
        )
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True
        )

    val_sampler = (
        DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    test_sampler = (
        DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )

    collate = collate_end_to_end if end_to_end else collate_embeddings
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate,
    )

    cfg["training"]["months"] = args.months

    if end_to_end:
        if args.config_path is None or args.checkpoint is None:
            raise ValueError("端到端训练需要 --config-path 与 --checkpoint")
        output_size = cfg["training"].get("crop_size", cfg["data"].get("patch_size", 128))
        output_size = (output_size, output_size)
        head = task.build_head().to(device)
        model, _ = build_end_to_end_model(
            args.config_path,
            args.checkpoint,
            head,
            freeze_backbone_epochs=freeze_backbone_epochs,
            output_size=output_size,
            target_size=output_size,
            months=args.months,
            include_diff=True,
        )
    else:
        model = task.build_head().to(device)

    if world_size > 1:
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    def _unwrap(m: nn.Module) -> nn.Module:
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    if end_to_end and freeze_backbone_epochs > 0:
        optimizer = torch.optim.AdamW(
            _unwrap(model).head.parameters(),
            lr=head_lr,
            weight_decay=cfg["training"].get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    elif end_to_end:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": _unwrap(model).encoder.parameters(),
                    "lr": backbone_lr,
                },
                {
                    "params": _unwrap(model).head.parameters(),
                    "lr": head_lr,
                },
            ],
            lr=head_lr,
            weight_decay=cfg["training"].get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"].get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    loss_fn = task.build_loss()

    early_stop_metric = cfg["training"].get("early_stop_metric", "miou")
    best_score = -float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    use_threshold_tuning = cfg["training"].get("use_threshold_tuning", False)
    best_threshold = 0.5

    for epoch in range(cfg["training"]["epochs"]):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        # 在冻结期结束后切换为完整优化器
        if (
            end_to_end
            and freeze_backbone_epochs > 0
            and epoch == freeze_backbone_epochs
        ):
            _unwrap(model).set_backbone_frozen(False)
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": _unwrap(model).encoder.parameters(),
                        "lr": backbone_lr,
                    },
                    {
                        "params": _unwrap(model).head.parameters(),
                        "lr": head_lr,
                    },
                ],
                lr=head_lr,
                weight_decay=cfg["training"].get("weight_decay", 0.0),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["training"]["epochs"]
            )
            if rank == 0:
                logger.info("Epoch %d 解冻 backbone，使用 backbone_lr=%.2e head_lr=%.2e", epoch, backbone_lr, head_lr)

        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if end_to_end:
                inp = batch["image"].to(device)
            else:
                inp = batch["embedding_map"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(inp)[:, 1]
            loss = loss_fn(logits, mask.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = train_loss / n_batches if n_batches > 0 else 0.0

        val_metrics = task.evaluate(model, val_loader, device)
        val_metrics = _average_metrics(val_metrics, world_size)

        if rank == 0:
            logger.info(
                "Epoch %d train_loss=%.4f val_%s=%.4f",
                epoch,
                train_loss,
                early_stop_metric,
                val_metrics[early_stop_metric],
            )

        if val_metrics[early_stop_metric] > best_score:
            best_score = val_metrics[early_stop_metric]
            patience_counter = 0
            best_epoch = epoch
            best_state = (
                model.module.state_dict()
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.state_dict()
            )
            if use_threshold_tuning:
                best_threshold = _find_best_threshold(model, val_loader, device)
            if rank == 0:
                (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(best_state, out_dir / "checkpoints" / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"].get("early_stop_patience", 10):
                if rank == 0:
                    logger.info("早停于 epoch %d", epoch)
                break

    # 测试
    if best_state is None:
        # 所有 rank 都未保存过 best_state（数据极少时可能出现），直接跳过
        return None

    if rank == 0:
        logger.info("加载最佳模型并测试 fold %d", fold_idx)
    model.load_state_dict(
        best_state
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else {f"module.{k}": v for k, v in best_state.items()}
    )
    test_metrics = task.evaluate(model, test_loader, device, threshold=best_threshold)
    test_metrics = _average_metrics(test_metrics, world_size)

    if rank == 0:
        test_metrics["fold"] = fold_idx
        test_metrics["best_epoch"] = best_epoch
        test_metrics["region"] = region
        test_metrics["fraction"] = args.fraction
        test_metrics["threshold"] = best_threshold
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        # 保存测试集概率图
        if args.regions:
            save_test_predictions(
                model,
                test_loader,
                device,
                out_dir / "predictions",
                mask_dir,
                args.months,
                region_of=region_of,
                mask_dirs=mask_dirs,
            )
        else:
            save_test_predictions(
                model,
                test_loader,
                device,
                out_dir / "predictions",
                mask_dir,
                args.months,
            )

    return test_metrics


def main() -> None:
    rank, local_rank, world_size = setup_ddp()

    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="construction")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="embedding 子目录名；默认从 label-root 父目录推断",
    )
    p.add_argument("--months", type=str, nargs="+", default=["202605"])
    p.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="联合训练的区域列表，例如 haidian harbin",
    )
    p.add_argument("--head-type", type=str, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None, help="只跑单个 fold 调试")
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="AEF 编码器配置路径（端到端训练必填）",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="AEF 编码器检查点路径（端到端训练必填）",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.head_type is not None:
        cfg["training"]["head_type"] = args.head_type

    set_seed(cfg["experiment"]["seed"] + rank)

    label_root = args.label_root / args.task
    mask_dir = label_root / "masks"
    bitemporal = len(args.months) == 2

    if args.regions:
        region = "joint"
        emb_region_root = args.embedding_root
        split_path = label_root / "split_joint_5fold.json"
        mask_dirs = {
            r: args.label_root / r / "labels" / args.task / "masks"
            for r in args.regions
        }
    else:
        region = args.region if args.region else args.label_root.parent.name
        emb_region_root = args.embedding_root / region
        split_path = label_root / "split_5fold.json"
        mask_dirs = None

    if rank == 0 and not split_path.exists():
        logger.info("split 文件不存在，自动生成: %s", split_path)
        if args.regions:
            split = create_combined_stratified_folds(
                mask_dirs, seed=cfg["experiment"]["seed"]
            )
        else:
            split = create_stratified_folds(mask_dir, seed=cfg["experiment"]["seed"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    if world_size > 1:
        dist.barrier()

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    folds = [split["folds"][args.fold]] if args.fold is not None else split["folds"]

    summary: list[dict[str, Any]] = []
    fractions = split.get("fractions", {})
    split_region_of = split.get("region_of", {})
    for fold_info in folds:
        test_metrics = run_fold(
            cfg,
            args,
            fold_info,
            emb_region_root=emb_region_root,
            mask_dir=mask_dir,
            mask_dirs=mask_dirs,
            region=region,
            bitemporal=bitemporal,
            fractions=fractions,
            region_of=split_region_of,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0 and test_metrics is not None:
            summary.append(test_metrics)

    # 汇总
    if rank == 0:
        with open(args.output_root / "summary_5fold.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("5-fold 汇总：%s", args.output_root / "summary_5fold.json")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp()
