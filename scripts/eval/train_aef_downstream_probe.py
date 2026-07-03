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
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-patches", type=int, default=8)
    parser.add_argument("--max-pixels-per-patch", type=int, default=4096)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument(
        "--no-cache-patches",
        action="store_true",
        help="Disable in-memory patch cache. By default, each fold caches AEF tensors once.",
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


def load_split(label_root: Path) -> dict[str, Any]:
    split_path = label_root / "split_5fold.json"
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
    train_items = build_items(embedding_region_root, args.label_root, fold_info["train"], args.month)
    val_items = build_items(embedding_region_root, args.label_root, fold_info["val"], args.month)
    test_items = build_items(embedding_region_root, args.label_root, fold_info["test"], args.month)
    if not args.no_cache_patches:
        LOGGER.info("fold=%d caching patches in memory", fold_idx)
        train_items = cache_items(train_items, l2_normalize=args.l2_normalize)
        val_items = cache_items(val_items, l2_normalize=args.l2_normalize)
        test_items = cache_items(test_items, l2_normalize=args.l2_normalize)

    model = PixelProbe(args.embed_dim, args.head, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = -1
    best_threshold = 0.5
    for epoch in range(args.epochs):
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
        scheduler.step()

        should_eval = (epoch == 0) or ((epoch + 1) % args.eval_every == 0)
        if not should_eval:
            LOGGER.info("fold=%d epoch=%d train_loss=%.4f val=skipped", fold_idx, epoch, np.mean(losses))
            continue
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
            np.mean(losses),
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
    test_logits, test_target = logits_for_items(
        model,
        test_items,
        device=device,
        l2_normalize=args.l2_normalize,
    )
    test_metrics = compute_metrics(test_logits, test_target, threshold=best_threshold)
    result: dict[str, Any] = {
        **test_metrics,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "val_threshold": best_threshold,
        "head": args.head,
        "month": args.month,
        "region": args.region,
        "l2_normalize": bool(args.l2_normalize),
        "trainer": "aef_pixel_probe",
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
    split = load_split(args.label_root)
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
    }
    (args.output_root / "summary_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote summary to %s", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
