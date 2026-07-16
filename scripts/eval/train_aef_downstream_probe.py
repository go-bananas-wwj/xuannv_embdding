#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn

LOGGER = logging.getLogger("aef_probe")


@dataclass(frozen=True)
class PatchItem:
    patch_id: str
    embedding_path: Path
    mask_path: Path


@dataclass(frozen=True)
class LoadedPatch:
    patch_id: str
    embedding: torch.Tensor
    mask: torch.Tensor
    mask_path: Path


@dataclass(frozen=True)
class FeatureCache:
    x: torch.Tensor
    y: torch.Tensor


class PixelProbe(nn.Module):
    def __init__(self, embed_dim: int, head: str, hidden_dim: int) -> None:
        super().__init__()
        head = head.lower()
        if head in {"linear", "logistic"}:
            self.net = nn.Linear(embed_dim, 1)
        elif head == "mlp":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        elif head == "mlp_deep":
            self.net = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(p=0.10),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.10),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEF-specific downstream probe using pixel-wise Linear/MLP heads."
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202512")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Explicit fixed split JSON. Paper evaluation should use a spatial block split.",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--head", choices=["linear", "mlp", "mlp_deep"], default="mlp")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-patches", type=int, default=8)
    parser.add_argument("--max-pixels-per-patch", type=int, default=4096)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument(
        "--train-positive-patches",
        type=int,
        default=None,
        help=(
            "Use only this many positive training patches for sparse-label mapping. "
            "Validation and test splits are unchanged. Selection is deterministic per fold/seed."
        ),
    )
    parser.add_argument(
        "--train-negative-patches",
        type=int,
        default=None,
        help=(
            "Optional number of all-negative training patches to add for sparse-label mapping. "
            "Defaults to min(train-positive-patches, available negatives)."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument(
        "--no-cache-patches",
        action="store_true",
        help="Disable in-memory patch cache. By default, each fold caches AEF tensors once.",
    )
    parser.add_argument(
        "--cache-device",
        choices=["none", "cpu", "npu"],
        default="npu",
        help=(
            "Materialize sampled train pixels once. 'npu' keeps the feature matrix on the "
            "training device and avoids per-epoch CPU sampling/copies."
        ),
    )
    parser.add_argument(
        "--eval-cache-device",
        choices=["none", "cpu", "npu"],
        default="cpu",
        help="Materialize val/test pixels for faster threshold search and metrics.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import torch_npu  # noqa: F401

        torch.npu.manual_seed_all(seed)
    except Exception:
        pass


def make_device(device_name: str) -> torch.device:
    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(device_name)
    if device.type == "npu":
        torch.npu.set_device(device)
    return device


def load_split(label_root: Path, split_path: Path | None = None) -> dict[str, Any]:
    split_path = split_path or (label_root / "split_5fold.json")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing fixed split: {split_path}")
    return json.loads(split_path.read_text(encoding="utf-8"))


def resolve_mask(mask_dir: Path, patch_id: str) -> Path:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        return exact
    return candidates[-1]


def patch_has_positive(mask_dir: Path, patch_id: str) -> bool:
    mask_path = resolve_mask(mask_dir, patch_id)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask for sparse selection: {mask_path}")
    with rasterio.open(mask_path) as src:
        mask = src.read(1, masked=True)
    return bool(np.asarray(mask.filled(0) == 1).any())


def select_sparse_train_patch_ids(
    label_root: Path,
    train_patch_ids: list[str],
    fold_idx: int,
    seed: int,
    positive_budget: int | None,
    negative_budget: int | None,
) -> list[str]:
    if positive_budget is None:
        return list(train_patch_ids)
    if positive_budget <= 0:
        raise ValueError("--train-positive-patches must be positive when provided")

    mask_dir = label_root / "masks"
    positives: list[str] = []
    negatives: list[str] = []
    for patch_id in train_patch_ids:
        if patch_has_positive(mask_dir, patch_id):
            positives.append(patch_id)
        else:
            negatives.append(patch_id)

    rng = random.Random(seed + fold_idx * 1009)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected_pos = positives[: min(positive_budget, len(positives))]
    if negative_budget is None:
        negative_budget = len(selected_pos)
    selected_neg = negatives[: min(max(0, negative_budget), len(negatives))]
    selected = selected_pos + selected_neg
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError(
            f"Sparse selection for fold {fold_idx} produced no train patches: "
            f"positives={len(positives)} negatives={len(negatives)}"
        )
    return selected


def build_items(
    embedding_region_root: Path,
    label_root: Path,
    patch_ids: list[str],
    month: str,
) -> list[PatchItem]:
    mask_dir = label_root / "masks"
    items: list[PatchItem] = []
    for patch_id in patch_ids:
        emb_path = embedding_region_root / patch_id / f"{month}_embedding_map.pt"
        mask_path = resolve_mask(mask_dir, patch_id)
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding: {emb_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        items.append(PatchItem(patch_id, emb_path, mask_path))
    return items


def load_patch(
    item: PatchItem | LoadedPatch,
    l2_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(item, LoadedPatch):
        return item.embedding, item.mask

    emb = torch.load(item.embedding_path, map_location="cpu", weights_only=True).float()
    if emb.ndim != 3:
        raise ValueError(f"Expected (D,H,W) embedding, got {tuple(emb.shape)}: {item.embedding_path}")
    if l2_normalize:
        emb = emb / emb.norm(dim=0, keepdim=True).clamp_min(1e-6)
    with rasterio.open(item.mask_path) as src:
        mask_np = src.read(1)
    mask = torch.from_numpy(mask_np.astype(np.int64))
    if emb.shape[-2:] != mask.shape:
        raise ValueError(
            f"Shape mismatch for {item.patch_id}: embedding {emb.shape[-2:]}, mask {tuple(mask.shape)}"
        )
    return emb, mask


def cache_items(items: list[PatchItem], l2_normalize: bool) -> list[LoadedPatch]:
    cached: list[LoadedPatch] = []
    for item in items:
        emb, mask = load_patch(item, l2_normalize=l2_normalize)
        cached.append(
            LoadedPatch(
                patch_id=item.patch_id,
                embedding=emb,
                mask=mask,
                mask_path=item.mask_path,
            )
        )
    return cached


def sample_pixels(
    emb: torch.Tensor,
    mask: torch.Tensor,
    max_pixels: int,
    positive_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).contiguous()
    y = mask.reshape(-1).long()
    valid = y >= 0
    x = x[valid]
    y = y[valid]
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        total = min(max_pixels, len(y))
        idx = torch.randperm(len(y))[:total]
        return x[idx], y[idx].float()

    n_pos = min(len(pos), max(1, int(max_pixels * positive_fraction)))
    n_neg = min(len(neg), max_pixels - n_pos)
    if n_neg <= 0:
        n_neg = min(len(neg), n_pos)
    pos_idx = pos[torch.randperm(len(pos))[:n_pos]]
    neg_idx = neg[torch.randperm(len(neg))[:n_neg]]
    idx = torch.cat([pos_idx, neg_idx])
    idx = idx[torch.randperm(len(idx))]
    return x[idx], y[idx].float()


def iter_patch_batches(items: list[PatchItem], batch_patches: int) -> list[list[PatchItem]]:
    order = list(items)
    random.shuffle(order)
    return [order[i : i + batch_patches] for i in range(0, len(order), batch_patches)]


def make_train_batch(
    items: list[PatchItem | LoadedPatch],
    max_pixels_per_patch: int,
    positive_fraction: float,
    l2_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for item in items:
        emb, mask = load_patch(item, l2_normalize=l2_normalize)
        x, y = sample_pixels(
            emb,
            mask,
            max_pixels=max_pixels_per_patch,
            positive_fraction=positive_fraction,
        )
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def materialize_sampled_features(
    items: list[PatchItem | LoadedPatch],
    max_pixels_per_patch: int,
    positive_fraction: float,
    l2_normalize: bool,
    device: torch.device | None = None,
) -> FeatureCache:
    x_cpu, y_cpu = make_train_batch(
        items,
        max_pixels_per_patch=max_pixels_per_patch,
        positive_fraction=positive_fraction,
        l2_normalize=l2_normalize,
    )
    if device is not None:
        return FeatureCache(
            x=x_cpu.to(device, non_blocking=True),
            y=y_cpu.to(device, non_blocking=True),
        )
    return FeatureCache(x=x_cpu, y=y_cpu)


def materialize_full_features(
    items: list[PatchItem | LoadedPatch],
    l2_normalize: bool,
    device: torch.device | None = None,
) -> FeatureCache:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for item in items:
        emb, mask = load_patch(item, l2_normalize=l2_normalize)
        x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).contiguous()
        y = mask.reshape(-1).float()
        valid = y >= 0
        xs.append(x[valid])
        ys.append(y[valid])
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    if device is not None:
        return FeatureCache(
            x=x_all.to(device, non_blocking=True),
            y=y_all.to(device, non_blocking=True),
        )
    return FeatureCache(x=x_all, y=y_all)


def train_epoch_from_feature_cache(
    model: nn.Module,
    cache: FeatureCache,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_pixels: int,
) -> float:
    model.train()
    total = int(cache.y.shape[0])
    order = torch.randperm(total, device=cache.y.device)
    losses: list[float] = []
    for start in range(0, total, batch_pixels):
        idx = order[start : start + batch_pixels]
        x = cache.x.index_select(0, idx)
        y = cache.y.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def logits_for_items(
    model: nn.Module,
    items: list[PatchItem | LoadedPatch],
    device: torch.device,
    l2_normalize: bool,
    chunk_pixels: int = 262144,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for item in items:
            emb, mask = load_patch(item, l2_normalize=l2_normalize)
            x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).contiguous()
            logits_parts: list[torch.Tensor] = []
            for start in range(0, x.shape[0], chunk_pixels):
                xb = x[start : start + chunk_pixels].to(device, non_blocking=True)
                logits_parts.append(model(xb).detach().cpu())
            logits = torch.cat(logits_parts, dim=0).numpy()
            all_logits.append(logits)
            all_targets.append(mask.reshape(-1).numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def logits_for_cache(
    model: nn.Module,
    cache: FeatureCache,
    device: torch.device,
    chunk_pixels: int = 262144,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, cache.y.shape[0], chunk_pixels):
            x = cache.x[start : start + chunk_pixels].to(device, non_blocking=True)
            logits_parts.append(model(x).detach().cpu())
    logits = torch.cat(logits_parts, dim=0).numpy()
    targets = cache.y.detach().cpu().numpy()
    return logits, targets


def compute_metrics(
    logits: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    valid = target >= 0
    target = (target[valid] == 1).astype(np.int32)
    probs = 1.0 / (1.0 + np.exp(-logits[valid]))
    pred = (probs > threshold).astype(np.uint8)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    if target.sum() == 0 or target.sum() == len(target):
        return {
            "miou": float(iou),
            "f1_at_threshold": float(f1),
            "f1_0.5": float(f1),
            "f1_best": 0.0,
            "best_threshold": 0.5,
            "precision": float(precision),
            "recall": float(recall),
            "ap": 0.0,
            "auprc": 0.0,
            "auc_roc": 0.0,
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    p_arr, r_arr, thresholds = precision_recall_curve(target, probs)
    f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
    best_idx = int(f1s.argmax())
    best_threshold = 0.5
    if thresholds.size:
        best_threshold = float(thresholds[min(best_idx, thresholds.size - 1)])
    return {
        "miou": float(iou),
        "f1_at_threshold": float(f1),
        "f1_0.5": float(compute_binary_f1(probs, target, 0.5)),
        "f1_best": float(f1s[best_idx]),
        "best_threshold": best_threshold,
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(average_precision_score(target, probs)),
        "auprc": float(auc(r_arr, p_arr)),
        "auc_roc": float(roc_auc_score(target, probs)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_binary_f1(probs: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred = (probs > threshold).astype(np.uint8)
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def save_predictions(
    model: nn.Module,
    items: list[PatchItem | LoadedPatch],
    out_dir: Path,
    device: torch.device,
    l2_normalize: bool,
    chunk_pixels: int = 262144,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for item in items:
            emb, _mask = load_patch(item, l2_normalize=l2_normalize)
            h, w = emb.shape[-2:]
            x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).contiguous()
            parts: list[torch.Tensor] = []
            for start in range(0, x.shape[0], chunk_pixels):
                xb = x[start : start + chunk_pixels].to(device, non_blocking=True)
                parts.append(torch.sigmoid(model(xb)).detach().cpu())
            prob = torch.cat(parts, dim=0).reshape(h, w).numpy().astype(np.float32)
            with rasterio.open(item.mask_path) as src:
                profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1, nodata=None, compress="lzw")
            with rasterio.open(out_dir / f"{item.patch_id}_prob.tif", "w", **profile) as dst:
                dst.write(prob, 1)


def train_fold(
    args: argparse.Namespace,
    fold_info: dict[str, Any],
    embedding_region_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    fold_idx = int(fold_info["fold"])
    out_dir = args.output_root / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_train_patch_ids = select_sparse_train_patch_ids(
        label_root=args.label_root,
        train_patch_ids=list(fold_info["train"]),
        fold_idx=fold_idx,
        seed=args.seed,
        positive_budget=args.train_positive_patches,
        negative_budget=args.train_negative_patches,
    )
    train_items = build_items(embedding_region_root, args.label_root, selected_train_patch_ids, args.month)
    val_items = build_items(embedding_region_root, args.label_root, fold_info["val"], args.month)
    test_items = build_items(embedding_region_root, args.label_root, fold_info["test"], args.month)
    (out_dir / "sparse_train_selection.json").write_text(
        json.dumps(
            {
                "fold": fold_idx,
                "seed": args.seed,
                "train_positive_patches": args.train_positive_patches,
                "train_negative_patches": args.train_negative_patches,
                "selected_train_patch_count": len(selected_train_patch_ids),
                "selected_train_patch_ids": selected_train_patch_ids,
                "full_train_patch_count": len(fold_info["train"]),
                "val_patch_count": len(fold_info["val"]),
                "test_patch_count": len(fold_info["test"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_cache_patches:
        LOGGER.info("fold=%d caching patches in memory", fold_idx)
        train_items = cache_items(train_items, l2_normalize=args.l2_normalize)
        val_items = cache_items(val_items, l2_normalize=args.l2_normalize)
        test_items = cache_items(test_items, l2_normalize=args.l2_normalize)
    train_feature_cache: FeatureCache | None = None
    val_feature_cache: FeatureCache | None = None
    test_feature_cache: FeatureCache | None = None
    if args.cache_device != "none":
        cache_device = device if args.cache_device == "npu" else None
        LOGGER.info("fold=%d materializing train pixel cache on %s", fold_idx, args.cache_device)
        train_feature_cache = materialize_sampled_features(
            train_items,
            max_pixels_per_patch=args.max_pixels_per_patch,
            positive_fraction=args.positive_fraction,
            l2_normalize=args.l2_normalize,
            device=cache_device,
        )
        LOGGER.info(
            "fold=%d train feature cache shape=%s device=%s",
            fold_idx,
            tuple(train_feature_cache.x.shape),
            train_feature_cache.x.device,
        )
    if args.eval_cache_device != "none":
        eval_device = device if args.eval_cache_device == "npu" else None
        LOGGER.info("fold=%d materializing eval caches on %s", fold_idx, args.eval_cache_device)
        val_feature_cache = materialize_full_features(
            val_items,
            l2_normalize=args.l2_normalize,
            device=eval_device,
        )
        test_feature_cache = materialize_full_features(
            test_items,
            l2_normalize=args.l2_normalize,
            device=eval_device,
        )

    model = PixelProbe(args.embed_dim, args.head, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = -1
    best_threshold = 0.5
    batch_pixels = max(1, args.batch_patches * args.max_pixels_per_patch)
    for epoch in range(args.epochs):
        if train_feature_cache is not None:
            train_loss = train_epoch_from_feature_cache(
                model,
                train_feature_cache,
                optimizer,
                loss_fn,
                batch_pixels=batch_pixels,
            )
        else:
            model.train()
            losses: list[float] = []
            for batch_items in iter_patch_batches(train_items, args.batch_patches):
                x_cpu, y_cpu = make_train_batch(
                    batch_items,
                    max_pixels_per_patch=args.max_pixels_per_patch,
                    positive_fraction=args.positive_fraction,
                    l2_normalize=args.l2_normalize,
                )
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(losses))
        scheduler.step()

        should_eval = (epoch == 0) or ((epoch + 1) % args.eval_every == 0)
        if not should_eval:
            LOGGER.info("fold=%d epoch=%d train_loss=%.4f val=skipped", fold_idx, epoch, train_loss)
            continue
        if val_feature_cache is not None:
            val_logits, val_target = logits_for_cache(model, val_feature_cache, device=device)
        else:
            val_logits, val_target = logits_for_items(
                model,
                val_items,
                device=device,
                l2_normalize=args.l2_normalize,
            )
        val_metrics = compute_metrics(val_logits, val_target)
        val_score = float(val_metrics["f1_best"])
        LOGGER.info(
            "fold=%d epoch=%d train_loss=%.4f val_f1_best=%.4f val_ap=%.4f",
            fold_idx,
            epoch,
            train_loss,
            val_score,
            val_metrics["ap"],
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_threshold = float(val_metrics["best_threshold"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_dir / "best.pt")

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for fold {fold_idx}")
    model.load_state_dict(best_state)
    if test_feature_cache is not None:
        test_logits, test_target = logits_for_cache(model, test_feature_cache, device=device)
    else:
        test_logits, test_target = logits_for_items(
            model,
            test_items,
            device=device,
            l2_normalize=args.l2_normalize,
        )
    test_metrics = compute_metrics(test_logits, test_target, threshold=best_threshold)
    test_metrics["oracle_test_f1"] = test_metrics.pop("f1_best")
    test_metrics["oracle_test_threshold"] = test_metrics.pop("best_threshold")
    result: dict[str, Any] = {
        **test_metrics,
        "paper_eligible": False,
        "protocol_status": "diagnostic_until_all_registered_evaluation_gates_pass",
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "val_threshold": best_threshold,
        "head": args.head,
        "month": args.month,
        "region": args.region,
        "l2_normalize": bool(args.l2_normalize),
        "cache_device": args.cache_device,
        "eval_cache_device": args.eval_cache_device,
        "trainer": "aef_pixel_probe",
        "train_positive_patches": args.train_positive_patches,
        "train_negative_patches": args.train_negative_patches,
        "selected_train_patch_count": len(selected_train_patch_ids),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.save_predictions:
        save_predictions(
            model,
            test_items,
            out_dir / "predictions",
            device=device,
            l2_normalize=args.l2_normalize,
        )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    split = load_split(args.label_root, args.split_path)
    embedding_region_root = args.embedding_root / args.region
    fold_infos = split["folds"]
    if args.fold is not None:
        fold_infos = [split["folds"][args.fold]]

    summary = []
    for fold_info in fold_infos:
        summary.append(train_fold(args, fold_info, embedding_region_root, device))

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "summary_5fold.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        "paper_eligible": False,
        "protocol_status": "diagnostic_until_all_registered_evaluation_gates_pass",
        "num_folds": len(summary),
        "requested_fold": args.fold,
        "is_full_5fold": len(summary) == 5 and args.fold is None,
        "summary_path": str(args.output_root / "summary.json"),
        "legacy_summary_5fold_path": str(args.output_root / "summary_5fold.json"),
        "trainer": "aef_pixel_probe",
        "device": args.device,
        "head": args.head,
        "epochs": args.epochs,
        "max_pixels_per_patch": args.max_pixels_per_patch,
        "positive_fraction": args.positive_fraction,
        "train_positive_patches": args.train_positive_patches,
        "train_negative_patches": args.train_negative_patches,
        "cache_device": args.cache_device,
        "eval_cache_device": args.eval_cache_device,
    }
    (args.output_root / "summary_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote summary to %s", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
