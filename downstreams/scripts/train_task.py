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
from downstreams.data.embedding_dataset import EmbeddingDataset, collate_embeddings
from downstreams.data.split import create_stratified_folds
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask
from downstreams.utils.config import load_config
from downstreams.utils.device import get_downstream_device
from downstreams.utils.reproducibility import set_seed
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def save_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_dir: Path,
    mask_dir: Path,
) -> None:
    """将测试集概率图保存为 GeoTIFF，便于后续可视化与溯源。"""
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding_map"].to(device)
            patch_ids = batch["patch_ids"]
            logits = model(emb)[:, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            for b, patch_id in enumerate(patch_ids):
                mask_path = mask_dir / f"{patch_id}.tif"
                with rasterio.open(mask_path) as src:
                    profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress="lzw",
                    nodata=None,
                )
                out_path = pred_dir / f"{patch_id}_prob.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(probs[b].astype(np.float32), 1)


def _positive_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum() / total)


def _build_train_loader(
    train_ds: EmbeddingDataset,
    train_ids: list[str],
    mask_dir: Path,
    training_cfg: dict,
) -> DataLoader:
    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 0)
    use_weighted = training_cfg.get("use_weighted_sampler", False)
    if not use_weighted:
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_embeddings,
        )
    min_weight = training_cfg.get("sampler_min_weight", 1e-4)
    oversample_factor = training_cfg.get("sampler_oversample_factor", 1.0)
    weights = []
    for pid in train_ids:
        ratio = _positive_ratio(mask_dir / f"{pid}.tif")
        weight = max(ratio, min_weight) ** oversample_factor
        weights.append(float(weight))
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()
    sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(train_ids),
        replacement=True,
    )
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_embeddings,
    )


def _build_scheduler(optimizer, training_cfg: dict):
    epochs = training_cfg["epochs"]
    lr = training_cfg["lr"]
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    scheduler_name = training_cfg.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
    elif scheduler_name == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=training_cfg.get("step_size", 30), gamma=training_cfg.get("lr_decay", 0.1)
        )
    else:
        raise ValueError(f"未知 scheduler: {scheduler_name}")
    if warmup_epochs <= 0:
        return main_scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def _gather_logits_and_masks(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding_map"].to(device)
            mask = batch["mask"].to(device)
            logits = model(emb)[:, 1]
            all_logits.append(logits.cpu())
            all_masks.append(mask.cpu())
    logits = torch.cat([x.flatten() for x in all_logits])
    masks = torch.cat([x.flatten() for x in all_masks])
    return logits, masks


def _select_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    from downstreams.metrics.segmentation import find_best_threshold

    logits, masks = _gather_logits_and_masks(model, loader, device)
    if masks.sum() == 0 or masks.sum() == masks.numel():
        return 0.5
    best_thr, _ = find_best_threshold(logits, masks, metric="f1")
    return best_thr


def _evaluate_at_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    from downstreams.metrics.segmentation import compute_segmentation_metrics_at_threshold

    logits, masks = _gather_logits_and_masks(model, loader, device)
    metrics = compute_segmentation_metrics_at_threshold(logits, masks, threshold)
    metrics["f1_0.5"] = metrics["f1"]
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="construction_segmentation")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="embedding 子目录名；默认从 label-root 父目录推断",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None, help="只跑单个 fold 调试")
    p.add_argument("--fraction", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_downstream_device(cfg["experiment"].get("device", "auto"))

    region = args.region if args.region else args.label_root.parent.name
    emb_region_root = args.embedding_root / region
    mask_dir = args.label_root / "masks"
    split_path = args.label_root / "split_5fold.json"
    if not split_path.exists():
        logger.info("split_5fold.json 不存在，自动生成")
        split = create_stratified_folds(mask_dir, seed=cfg["experiment"]["seed"])
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
    else:
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

    task = ConstructionSegmentationTask(cfg)
    folds = [split["folds"][args.fold]] if args.fold is not None else split["folds"]

    summary = []
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        logger.info("===== Fold %d =====", fold_idx)
        out_dir = args.output_root / f"fold_{fold_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_ids = fold_info["train"]
        if args.fraction is not None:
            frac_str = str(args.fraction)
            train_ids = split["fractions"][frac_str][f"fold_{fold_idx}"]

        train_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            train_ids,
            month=cfg["training"]["month"],
            augment=True,
        )
        val_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["val"],
            month=cfg["training"]["month"],
        )
        test_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["test"],
            month=cfg["training"]["month"],
        )

        train_loader = _build_train_loader(
            train_ds,
            train_ids,
            mask_dir,
            cfg["training"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_embeddings,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_embeddings,
        )

        model = task.build_head().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = _build_scheduler(optimizer, cfg["training"])
        loss_fn = task.build_loss().to(device)
        early_stop_metric = cfg["training"].get("early_stop_metric", "ap")
        grad_clip = cfg["training"].get("grad_clip", None)
        use_threshold_tuning = cfg["training"].get("use_threshold_tuning", False)

        best_score = -1.0
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None
        best_threshold = 0.5
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                emb = batch["embedding_map"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(emb)[:, 1]
                loss = loss_fn(logits, mask.float())
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            val_metrics = task.evaluate(model, val_loader, device)
            score = val_metrics.get(early_stop_metric, val_metrics["ap"])
            logger.info(
                "Epoch %d train_loss=%.4f val_%s=%.4f val_miou=%.4f",
                epoch,
                train_loss,
                early_stop_metric,
                score,
                val_metrics["miou"],
            )

            if score > best_score:
                best_score = score
                patience_counter = 0
                best_state = model.state_dict()
                (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(best_state, out_dir / "checkpoints" / "best.pt")
                if use_threshold_tuning:
                    best_threshold = _select_threshold(model, val_loader, device)
            else:
                patience_counter += 1
                if patience_counter >= cfg["training"]["early_stop_patience"]:
                    logger.info("早停于 epoch %d", epoch)
                    break

        # 测试
        assert best_state is not None
        model.load_state_dict(best_state)
        test_metrics_default = task.evaluate(model, test_loader, device)
        if use_threshold_tuning:
            test_metrics_tuned = _evaluate_at_threshold(
                model, test_loader, device, best_threshold
            )
            test_metrics = test_metrics_tuned
            test_metrics["miou_0.5"] = test_metrics_default["miou"]
            test_metrics["f1_0.5_default"] = test_metrics_default["f1_0.5"]
            test_metrics["f1_best"] = test_metrics_default["f1_best"]
            test_metrics["ap"] = test_metrics_default["ap"]
            test_metrics["auprc"] = test_metrics_default["auprc"]
            test_metrics["auc_roc"] = test_metrics_default["auc_roc"]
            test_metrics["best_threshold"] = best_threshold
        else:
            test_metrics = test_metrics_default
        test_metrics["fold"] = fold_idx
        test_metrics["best_epoch"] = epoch - patience_counter
        test_metrics["region"] = region
        test_metrics["fraction"] = args.fraction
        summary.append(test_metrics)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        # 保存测试集概率图
        save_test_predictions(model, test_loader, device, out_dir / "predictions", mask_dir)

    # 汇总
    with open(args.output_root / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("5-fold 汇总：%s", args.output_root / "summary_5fold.json")


if __name__ == "__main__":
    main()
