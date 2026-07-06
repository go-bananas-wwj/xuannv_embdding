#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.nn import functional as F

from scripts.eval.run_traditional_ml_benchmark import (
    DEFAULT_TASKS,
    PatchRecord,
    TaskSpec,
    add_optical_indices,
    load_binary_mask,
    load_feature_map,
    load_manifest,
    load_split,
    local_stats,
    read_raster_mean,
    resolve_mask,
    safe_index,
    select_train_patch_ids,
    task_spec,
)


@dataclass(frozen=True)
class PatchTensor:
    patch_id: str
    x: torch.Tensor
    y: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelConvHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(e3)
        d2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1).squeeze(1)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        rates = [1, 2, 4, 8]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
                for rate in rates
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(rates), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabLite(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.stem = ConvBlock(in_ch, base)
        self.low = ConvBlock(base, base * 2)
        self.high = ConvBlock(base * 2, base * 4)
        self.aspp = ASPP(base * 4, base * 2)
        self.decoder = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        l = self.low(F.avg_pool2d(s, 2))
        h = self.high(F.avg_pool2d(l, 2))
        a = self.aspp(h)
        a = F.interpolate(a, size=l.shape[-2:], mode="bilinear", align_corners=False)
        d = self.decoder(torch.cat([a, l], dim=1))
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(d).squeeze(1)


class SegFormerLite(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.proj1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.GELU())
        self.proj2 = nn.Sequential(nn.Conv2d(in_ch, base, 3, stride=2, padding=1), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(in_ch, base, 5, stride=4, padding=2), nn.GELU())
        self.mix = nn.Sequential(
            nn.Conv2d(base * 3, base * 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base, 3, padding=1, groups=max(1, base // 8)),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.proj1(x)
        p2 = F.interpolate(self.proj2(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.mix(torch.cat([p1, p2, p4], dim=1)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strong downstream segmentation benchmark. It compares Xuannv embeddings "
            "with stronger heads against supervised CNN baselines trained on raw features."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"),
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian/labels"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--tasks", nargs="+", default=["building", "road", "water", "park_green", "education", "sports_pitch"])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["xuannv_embedding", "s2_s1_landsat_highres_indices"],
    )
    parser.add_argument("--models", nargs="+", default=["pixel_conv", "unet", "deeplab_lite", "segformer_lite"])
    parser.add_argument("--shots", nargs="+", default=["50", "full"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-patches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--predict-all", action="store_true")
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


def make_device(name: str) -> torch.device:
    if name.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(name)
    if device.type == "npu":
        torch.npu.set_device(device)
    return device


def make_model(name: str, in_ch: int, base: int) -> nn.Module:
    if name == "pixel_conv":
        return PixelConvHead(in_ch, base * 2)
    if name == "unet":
        return TinyUNet(in_ch, base)
    if name == "deeplab_lite":
        return DeepLabLite(in_ch, base)
    if name == "segformer_lite":
        return SegFormerLite(in_ch, base)
    raise KeyError(f"Unknown model {name}")


def zeros(channels: int, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    return np.zeros((channels, shape[0], shape[1]), dtype=np.float32)


def fixed_highres_feature_map(record: PatchRecord, month: str) -> np.ndarray:
    parts: list[np.ndarray] = []

    s2 = read_raster_mean(record.sources.get("s2", []), month)
    if s2 is None:
        parts.append(zeros(17))
    else:
        parts.append(np.concatenate(add_optical_indices(s2, "s2"), axis=0))

    s1 = read_raster_mean(record.sources.get("s1", []), month)
    if s1 is None:
        parts.append(zeros(3))
    else:
        s1_parts = [s1]
        if s1.shape[0] >= 2:
            s1_parts.append(safe_index(s1[0], s1[1])[None, :, :])
        else:
            s1_parts.append(zeros(1))
        parts.append(np.concatenate(s1_parts, axis=0)[:3])

    landsat = read_raster_mean(record.sources.get("landsat", []), month)
    if landsat is None:
        parts.append(zeros(12))
    else:
        parts.append(np.concatenate(add_optical_indices(landsat, "landsat"), axis=0))

    highres_opt = read_raster_mean(record.sources.get("highres_optical_haidian", []), month)
    if highres_opt is None:
        parts.append(zeros(6))
    else:
        parts.append(np.concatenate([highres_opt, *local_stats(highres_opt)], axis=0)[:6])

    highres_sar = read_raster_mean(record.sources.get("highres_sar_haidian", []), month)
    if highres_sar is None:
        parts.append(zeros(4))
    else:
        parts.append(np.concatenate([highres_sar, *local_stats(highres_sar)], axis=0)[:4])

    out = np.concatenate(parts, axis=0).astype(np.float32)
    if out.shape[0] != 42:
        raise ValueError(f"Expected 42 fixed highres channels, got {out.shape}")
    out[~np.isfinite(out)] = 0.0
    return out


def load_benchmark_feature_map(
    feature_set: str,
    record: PatchRecord,
    embedding_root: Path,
    region: str,
    month: str,
) -> np.ndarray:
    if feature_set == "s2_s1_landsat_highres_indices":
        return fixed_highres_feature_map(record, month)
    return load_feature_map(feature_set, record, embedding_root, region, month)


def load_patch_tensor(
    records: dict[str, PatchRecord],
    task: TaskSpec,
    patch_id: str,
    feature_set: str,
    embedding_root: Path,
    region: str,
    month: str,
) -> PatchTensor:
    x = load_benchmark_feature_map(feature_set, records[patch_id], embedding_root, region, month)
    y = load_binary_mask(task, patch_id)
    if x.shape[-2:] != y.shape:
        raise ValueError(f"Shape mismatch patch={patch_id}: x={x.shape}, y={y.shape}")
    return PatchTensor(
        patch_id=patch_id,
        x=torch.from_numpy(x.astype(np.float32)),
        y=torch.from_numpy((y == 1).astype(np.float32)),
    )


def load_patch_list(
    records: dict[str, PatchRecord],
    task: TaskSpec,
    patch_ids: list[str],
    feature_set: str,
    embedding_root: Path,
    region: str,
    month: str,
) -> list[PatchTensor]:
    return [
        load_patch_tensor(records, task, patch_id, feature_set, embedding_root, region, month)
        for patch_id in patch_ids
    ]


def batches(items: list[PatchTensor], batch_size: int, shuffle: bool) -> list[list[PatchTensor]]:
    order = list(items)
    if shuffle:
        random.shuffle(order)
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def stack_batch(items: list[PatchTensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([item.x for item in items], dim=0).to(device, non_blocking=True)
    y = torch.stack([item.y for item in items], dim=0).to(device, non_blocking=True)
    return x, y


def pos_weight_for(items: list[PatchTensor], device: torch.device) -> torch.Tensor:
    pos = sum(float(item.y.sum()) for item in items)
    total = sum(int(item.y.numel()) for item in items)
    neg = max(1.0, total - pos)
    weight = min(50.0, neg / max(1.0, pos))
    return torch.tensor(weight, dtype=torch.float32, device=device)


def train_epoch(
    model: nn.Module,
    items: list[PatchTensor],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in batches(items, batch_size=batch_size, shuffle=True):
        x, y = stack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def logits_for_items(
    model: nn.Module,
    items: list[PatchTensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches(items, batch_size=batch_size, shuffle=False):
            x, y = stack_batch(batch, device)
            logits.append(model(x).detach().cpu().reshape(-1).numpy())
            targets.append(y.detach().cpu().reshape(-1).numpy())
    return np.concatenate(logits), np.concatenate(targets)


def compute_binary_f1(prob: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    target = (target == 1).astype(np.uint8)
    tp = ((pred == 1) & (target == 1)).sum()
    fp = ((pred == 1) & (target == 0)).sum()
    fn = ((pred == 0) & (target == 1)).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def compute_metrics(logits: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    target = (target == 1).astype(np.uint8)
    prob = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    prob[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    prob[~positive] = exp_logits / (1.0 + exp_logits)
    pred = (prob >= threshold).astype(np.uint8)
    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())
    tn = int(((pred == 0) & (target == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    out: dict[str, float | int] = {
        "f1_at_threshold": float(f1),
        "f1_0.5": float(compute_binary_f1(prob, target, 0.5)),
        "miou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_ratio": float(target.mean()) if target.size else 0.0,
    }
    if target.sum() == 0 or target.sum() == len(target):
        out.update({"f1_best": 0.0, "best_threshold": 0.5, "ap": 0.0, "auprc": 0.0, "auc_roc": 0.0})
        return out
    p_arr, r_arr, thresholds = precision_recall_curve(target, prob)
    f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
    best_idx = int(f1s.argmax())
    best_threshold = 0.5
    if thresholds.size:
        best_threshold = float(thresholds[min(best_idx, thresholds.size - 1)])
    out.update(
        {
            "f1_best": float(f1s[best_idx]),
            "best_threshold": best_threshold,
            "ap": float(average_precision_score(target, prob)),
            "auprc": float(auc(r_arr, p_arr)),
            "auc_roc": float(roc_auc_score(target, prob)),
        }
    )
    return out


def write_prediction_maps(
    model: nn.Module,
    items: list[PatchTensor],
    task: TaskSpec,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in batches(items, batch_size=batch_size, shuffle=False):
            x = torch.stack([item.x for item in batch], dim=0).to(device, non_blocking=True)
            prob = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32)
            for idx, item in enumerate(batch):
                ref_path = resolve_mask(task.label_roots[0] / "masks", item.patch_id)
                with rasterio.open(ref_path) as src:
                    profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, count=1, nodata=None, compress="lzw")
                with rasterio.open(out_dir / f"{item.patch_id}_prob.tif", "w", **profile) as dst:
                    dst.write(prob[idx], 1)


def run_one(
    args: argparse.Namespace,
    records: dict[str, PatchRecord],
    task: TaskSpec,
    split: dict[str, list[str]],
    feature_set: str,
    model_name: str,
    shot: str,
    device: torch.device,
) -> dict[str, Any]:
    train_ids = select_train_patch_ids(task, split["train"], shot, args.seed, args.fold)
    train_items = load_patch_list(records, task, train_ids, feature_set, args.embedding_root, args.region, args.month)
    val_items = load_patch_list(records, task, split["val"], feature_set, args.embedding_root, args.region, args.month)
    test_items = load_patch_list(records, task, split["test"], feature_set, args.embedding_root, args.region, args.month)
    in_ch = int(train_items[0].x.shape[0])
    model = make_model(model_name, in_ch, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_items, device))

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = -1
    best_threshold = 0.5
    start_time = time.perf_counter()
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_items, device, optimizer, loss_fn, args.batch_size)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val_logits, val_target = logits_for_items(model, val_items, device, args.batch_size)
            val_metrics = compute_metrics(val_logits, val_target)
            val_score = float(val_metrics["f1_best"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(loss),
                    "val_f1_best": val_score,
                    "val_ap": float(val_metrics["ap"]),
                }
            )
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
                best_threshold = float(val_metrics["best_threshold"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state selected")
    model.load_state_dict(best_state)
    test_logits, test_target = logits_for_items(model, test_items, device, args.batch_size)
    test_metrics = compute_metrics(test_logits, test_target, threshold=best_threshold)
    elapsed = time.perf_counter() - start_time
    params = sum(p.numel() for p in model.parameters())
    return {
        **test_metrics,
        "task": task.name,
        "feature_set": feature_set,
        "model": model_name,
        "shot": shot,
        "fold": args.fold,
        "status": "ok",
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "val_threshold": best_threshold,
        "train_patch_count": len(train_ids),
        "val_patch_count": len(split["val"]),
        "test_patch_count": len(split["test"]),
        "in_channels": in_ch,
        "parameters": int(params),
        "train_seconds": float(elapsed),
        "history": history,
        "selected_train_patch_ids": train_ids,
        "_model_state": best_state,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest, args.data_root)
    all_metrics: list[dict[str, Any]] = []
    for task_name in args.tasks:
        task = task_spec(task_name, args.label_root)
        split = load_split(task, args.fold)
        if args.smoke_patches is not None:
            for key in split:
                split[key] = split[key][: args.smoke_patches]
        for shot in args.shots:
            for feature_set in args.feature_sets:
                for model_name in args.models:
                    out_dir = args.output_root / task_name / feature_set / model_name / f"shot_{shot}" / f"fold_{args.fold}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        record = run_one(args, records, task, split, feature_set, model_name, shot, device)
                        model_state = record.pop("_model_state")
                        torch.save(model_state, out_dir / "best.pt")
                        (out_dir / "metrics.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        (out_dir / "train_selection.json").write_text(
                            json.dumps(
                                {"patch_ids": record["selected_train_patch_ids"]},
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        if args.save_predictions or args.predict_all:
                            pred_ids = sorted(records) if args.predict_all else split["test"]
                            pred_items = load_patch_list(
                                records,
                                task,
                                pred_ids,
                                feature_set,
                                args.embedding_root,
                                args.region,
                                args.month,
                            )
                            model = make_model(model_name, int(pred_items[0].x.shape[0]), args.base_channels).to(device)
                            model.load_state_dict(model_state)
                            write_prediction_maps(
                                model,
                                pred_items,
                                task,
                                out_dir / ("predictions_all" if args.predict_all else "predictions"),
                                device,
                                args.batch_size,
                            )
                    except Exception as exc:
                        record = {
                            "task": task_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "shot": shot,
                            "fold": args.fold,
                            "status": "failed",
                            "error": repr(exc),
                        }
                        (out_dir / "metrics.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    all_metrics.append(record)
                    print(
                        f"{task_name} {feature_set} {model_name} shot={shot} "
                        f"status={record['status']} f1={record.get('f1_at_threshold', float('nan')):.4f} "
                        f"ap={record.get('ap', float('nan')):.4f}",
                        flush=True,
                    )
    clean_metrics = [{k: v for k, v in row.items() if k != "selected_train_patch_ids"} for row in all_metrics]
    (args.output_root / "all_metrics.json").write_text(
        json.dumps(clean_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    keys = sorted({key for row in clean_metrics for key in row})
    with (args.output_root / "all_metrics.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in clean_metrics:
            f.write(",".join(str(row.get(key, "")).replace(",", ";") for key in keys) + "\n")
    meta = vars(args).copy()
    meta.update({"device": str(device)})
    (args.output_root / "run_meta.json").write_text(json.dumps(meta, default=str, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
