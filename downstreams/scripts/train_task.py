#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

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
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_MONTH_SUFFIX_RE = re.compile(r"^(?P<patch>.+)_(?P<month>\d{6})$")


def _valid_yyyymm(value: str) -> bool:
    year = int(value[:4])
    month = int(value[4:])
    return 1900 <= year <= 2100 and 1 <= month <= 12


def resolve_mask_path(mask_dir: Path, patch_id: str) -> Path:
    """Resolve exact masks or month-suffixed masks for a split patch id."""
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        return exact

    def _month(path: Path) -> int:
        match = _MONTH_SUFFIX_RE.match(path.stem)
        if match is None or not _valid_yyyymm(match.group("month")):
            return -1
        return int(match.group("month"))

    return max(candidates, key=_month)


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
                mask_path = resolve_mask_path(mask_dir, patch_id)
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
    p.add_argument(
        "--skip-predictions",
        action="store_true",
        help="跳过测试集概率 GeoTIFF 导出；用于快速诊断。",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="每 N 个 epoch 做一次验证；默认读取 training.eval_every 或 1。",
    )
    p.add_argument(
        "--allow-create-split",
        action="store_true",
        help="允许在 split_5fold.json 缺失时自动生成；正式验收默认禁止。",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_downstream_device(cfg["experiment"].get("device", "auto"))
    eval_every = args.eval_every or int(cfg["training"].get("eval_every", 1))
    if eval_every <= 0:
        raise ValueError(f"eval_every 必须为正整数，实际得到 {eval_every}")

    region = args.region if args.region else args.label_root.parent.name
    emb_region_root = args.embedding_root / region
    mask_dir = args.label_root / "masks"
    split_path = args.label_root / "split_5fold.json"
    if not split_path.exists():
        if not args.allow_create_split:
            raise FileNotFoundError(
                f"固定 split 不存在: {split_path}。正式验收禁止自动生成；"
                "调试时可显式传 --allow-create-split。"
            )
        logger.info("split_5fold.json 不存在，按显式参数自动生成")
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
        months = cfg["training"].get("months")
        if months is None:
            months = [cfg["training"]["month"]]
        temporal_mode = cfg["training"].get("temporal_mode", "single")

        train_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            train_ids,
            months=months,
            temporal_mode=temporal_mode,
            augment=True,
        )
        val_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["val"],
            months=months,
            temporal_mode=temporal_mode,
        )
        test_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["test"],
            months=months,
            temporal_mode=temporal_mode,
        )

        num_workers = int(cfg["training"].get("num_workers", 0))
        loader_kwargs = {
            "num_workers": num_workers,
            "collate_fn": collate_embeddings,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(
                cfg["training"].get("prefetch_factor", 2)
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )

        model = task.build_head().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
        loss_fn = task.build_loss()

        best_miou = -1.0
        best_threshold = 0.5
        patience_counter = 0
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
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
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            should_eval = (epoch + 1) % eval_every == 0 or epoch == 0
            if not should_eval:
                logger.info(
                    "Epoch %d train_loss=%.4f val_miou=skipped",
                    epoch,
                    train_loss,
                )
                continue

            val_metrics = task.evaluate(model, val_loader, device)
            logger.info(
                "Epoch %d train_loss=%.4f val_miou=%.4f",
                epoch,
                train_loss,
                val_metrics["miou"],
            )

            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                best_threshold = float(val_metrics.get("best_threshold", 0.5))
                patience_counter = 0
                best_epoch = epoch
                best_state = model.state_dict()
                (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(best_state, out_dir / "checkpoints" / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg["training"]["early_stop_patience"]:
                    logger.info("早停于 epoch %d", epoch)
                    break

        # 测试
        assert best_state is not None
        model.load_state_dict(best_state)
        test_metrics = task.evaluate(model, test_loader, device, threshold=best_threshold)
        test_metrics["fold"] = fold_idx
        test_metrics["best_epoch"] = best_epoch
        test_metrics["val_threshold"] = best_threshold
        test_metrics["months"] = months
        test_metrics["temporal_mode"] = temporal_mode
        test_metrics["region"] = region
        test_metrics["fraction"] = args.fraction
        summary.append(test_metrics)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        # 保存测试集概率图
        if not args.skip_predictions and cfg["training"].get("save_predictions", True):
            save_test_predictions(
                model,
                test_loader,
                device,
                out_dir / "predictions",
                mask_dir,
            )

    # 汇总
    with open(args.output_root / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("5-fold 汇总：%s", args.output_root / "summary_5fold.json")


if __name__ == "__main__":
    main()
