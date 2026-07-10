#!/usr/bin/env python3
"""Train a lightweight multiclass land-use head on Haidian production embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from downstreams.heads.segmentation_head import MLPProbeHead, FCNHead


LOGGER = logging.getLogger("haidian_landuse_probe")
PATCH_RE = re.compile(r"(patch_\d{6})")


class LandUseDataset(Dataset):
    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        month: str,
        augment: bool = False,
    ) -> None:
        self.embedding_root = embedding_root
        self.label_root = label_root
        self.patch_ids = patch_ids
        self.month = month
        self.augment = augment
        self.mask_dir = label_root / "masks"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        emb_path = self.embedding_root / "haidian" / patch_id / f"{self.month}_embedding_map.pt"
        mask_path = resolve_mask(self.mask_dir, patch_id)
        if not emb_path.exists():
            raise FileNotFoundError(emb_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
        with rasterio.open(mask_path) as src:
            mask = torch.from_numpy(src.read(1).astype(np.int64))
        if emb.shape[-2:] != mask.shape:
            raise ValueError(f"{patch_id} embedding {emb.shape[-2:]} != mask {tuple(mask.shape)}")
        if self.augment:
            if torch.rand(()) > 0.5:
                emb = torch.flip(emb, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if torch.rand(()) > 0.5:
                emb = torch.flip(emb, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
        return {"embedding_map": emb, "mask": mask, "patch_id": patch_id}


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/embeddings/production/"
            "20260709_haidian_embedding_v1_p10c_epoch800_production_epoch_80_"
            "haidian_embedding_v1_p10c_epoch800"
        ),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/labels/osm_landcover"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/experiments/production/"
            "haidian_v1_landuse_construction_20260710/landuse"
        ),
    )
    parser.add_argument("--month", default="202604")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--head", choices=["mlp", "conv3x3"], default="conv3x3")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-all-predictions", action="store_true")
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


def patch_id_from_path(path: Path) -> str:
    match = PATCH_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse patch id from {path}")
    return match.group(1)


def resolve_mask(mask_dir: Path, patch_id: str) -> Path:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"*{patch_id}.tif"))
    if not candidates:
        return exact
    return candidates[-1]


def load_or_create_split(label_root: Path, seed: int) -> dict[str, Any]:
    split_path = label_root / "split_5fold.json"
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))
    patch_ids = sorted(patch_id_from_path(p) for p in (label_root / "masks").glob("*.tif"))
    rng = random.Random(seed)
    rng.shuffle(patch_ids)
    folds = []
    fold_size = len(patch_ids) // 5
    for fold in range(5):
        test = patch_ids[fold * fold_size : (fold + 1) * fold_size] if fold < 4 else patch_ids[fold * fold_size :]
        val_fold = (fold + 1) % 5
        val = (
            patch_ids[val_fold * fold_size : (val_fold + 1) * fold_size]
            if val_fold < 4
            else patch_ids[val_fold * fold_size :]
        )
        held = set(test) | set(val)
        train = [pid for pid in patch_ids if pid not in held]
        folds.append({"fold": fold, "train": train, "val": val, "test": test})
    split = {"seed": seed, "folds": folds}
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return split


def load_metadata(label_root: Path) -> tuple[list[str], list[str]]:
    meta = json.loads((label_root / "metadata.json").read_text(encoding="utf-8"))
    names_by_id = {int(k): v for k, v in meta["class_names"].items()}
    colors_by_id = {int(k): v for k, v in meta["class_colors"].items()}
    max_id = max(names_by_id)
    names = ["unknown"] * (max_id + 1)
    colors = ["#ffffff"] * (max_id + 1)
    for idx, name in names_by_id.items():
        names[idx] = name
    for idx, color in colors_by_id.items():
        colors[idx] = color
    return names, colors


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embedding_map": torch.stack([x["embedding_map"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "patch_ids": [x["patch_id"] for x in batch],
    }


def build_head(head: str, num_classes: int) -> nn.Module:
    if head == "mlp":
        return MLPProbeHead(embed_dim=64, num_classes=num_classes, hidden_dim=128)
    if head == "conv3x3":
        return FCNHead(embed_dim=64, num_classes=num_classes, hidden_dim=128)
    raise ValueError(head)


def class_weights(label_root: Path, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in sorted((label_root / "masks").glob("*.tif")):
        with rasterio.open(path) as src:
            arr = src.read(1)
        values, value_counts = np.unique(arr, return_counts=True)
        for value, count in zip(values, value_counts, strict=True):
            if 0 <= int(value) < num_classes:
                counts[int(value)] += int(count)
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / np.log(1.02 + np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding_map"].to(device, non_blocking=True)
            logits = model(emb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = probs.argmax(axis=1).astype(np.int64)
            all_probs.append(np.moveaxis(probs, 1, -1).reshape(-1, num_classes))
            all_preds.append(pred.reshape(-1))
            all_targets.append(batch["mask"].numpy().reshape(-1))
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    labels = list(range(num_classes))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    miou = float(jaccard_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_iou = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_ap: list[float] = []
    per_auc: list[float] = []
    for class_id in labels:
        binary = (y_true == class_id).astype(np.uint8)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_ap.append(0.0)
            per_auc.append(0.0)
        else:
            per_ap.append(float(average_precision_score(binary, y_prob[:, class_id])))
            per_auc.append(float(roc_auc_score(binary, y_prob[:, class_id])))
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "miou": miou,
        "overall_accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": [float(x) for x in per_f1],
        "per_class_iou": [float(x) for x in per_iou],
        "per_class_ap": per_ap,
        "per_class_auc_roc": per_auc,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        emb = batch["embedding_map"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(emb), mask.long())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def save_predictions(
    model: nn.Module,
    dataset: LandUseDataset,
    out_dir: Path,
    device: torch.device,
    colors: list[str],
) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            patch_id = item["patch_id"]
            emb = item["embedding_map"][None].to(device)
            prob = torch.softmax(model(emb), dim=1)[0].detach().cpu().numpy()
            pred = prob.argmax(axis=0).astype(np.uint8)
            mask_path = resolve_mask(dataset.mask_dir, patch_id)
            with rasterio.open(mask_path) as src:
                profile = src.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
            with rasterio.open(pred_dir / f"{patch_id}_pred.tif", "w", **profile) as dst:
                dst.write(pred, 1)
            rgb = label_to_rgb(pred, colors)
            Image.fromarray(rgb).save(pred_dir / f"{patch_id}_pred.png")


def label_to_rgb(label: np.ndarray, colors: list[str]) -> np.ndarray:
    rgb = np.ones((*label.shape, 3), dtype=np.uint8) * 255
    for class_id, color in enumerate(colors):
        color = color.lstrip("#")
        if len(color) != 6:
            continue
        rgb[label == class_id] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb


def load_layout(mask_dir: Path) -> tuple[list[PatchLayout], int, int, int, int]:
    paths = sorted(mask_dir.glob("*.tif"))
    rows_raw: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    for path in paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
        patch_id = patch_id_from_path(path)
        rows_raw.append((patch_id, path, float(bounds.left), float(bounds.top), width, height))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(patch_id, row_of[round(top, 3)], col_of[round(left, 3)], path)
        for patch_id, path, left, top, _w, _h in rows_raw
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def save_full_domain(pred_dir: Path, label_root: Path, output_path: Path, colors: list[str]) -> None:
    layouts, rows, cols, tile_h, tile_w = load_layout(label_root / "masks")
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 238
    for layout in layouts:
        pred_path = pred_dir / f"{layout.patch_id}_pred.tif"
        source = pred_path if pred_path.exists() else layout.mask_path
        with rasterio.open(source) as src:
            arr = src.read(1)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = label_to_rgb(arr, colors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(canvas)
    header = 72
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), "Haidian land-use prediction, 320 patches", fill=(0, 0, 0))
    out.paste(image, (0, header))
    out.save(output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    device = make_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    names, colors = load_metadata(args.label_root)
    split = load_or_create_split(args.label_root, args.seed)
    fold_info = split["folds"][args.fold]
    train_ds = LandUseDataset(args.embedding_root, args.label_root, fold_info["train"], args.month, augment=True)
    val_ds = LandUseDataset(args.embedding_root, args.label_root, fold_info["val"], args.month)
    test_ds = LandUseDataset(args.embedding_root, args.label_root, fold_info["test"], args.month)
    all_ds = LandUseDataset(
        args.embedding_root,
        args.label_root,
        sorted(patch_id_from_path(p) for p in (args.label_root / "masks").glob("*.tif")),
        args.month,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    num_classes = len(names)
    model = build_head(args.head, num_classes).to(device)
    weights = class_weights(args.label_root, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_score = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        row: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        if epoch == 0 or (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, num_classes)
            row.update({f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, list)})
            score = float(val_metrics["macro_f1"])
            LOGGER.info(
                "epoch=%d train_loss=%.4f val_macro_f1=%.4f val_miou=%.4f",
                epoch,
                train_loss,
                score,
                val_metrics["miou"],
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                (args.output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(best_state, args.output_root / "checkpoints" / "best.pt")
        else:
            LOGGER.info("epoch=%d train_loss=%.4f val=skipped", epoch, train_loss)
        history.append(row)

    if best_state is None:
        raise RuntimeError("No best checkpoint selected")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, num_classes)
    result = {
        **test_metrics,
        "fold": args.fold,
        "head": args.head,
        "month": args.month,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "class_names": names,
        "class_colors": colors,
        "trainer": "haidian_landuse_multiclass_probe",
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.save_all_predictions:
        save_predictions(model, all_ds, args.output_root, device, colors)
        save_full_domain(
            args.output_root / "predictions",
            args.label_root,
            args.output_root / "visualizations" / "landuse_prediction_320patch_geo.png",
            colors,
        )
    LOGGER.info("Wrote metrics to %s", args.output_root / "metrics.json")


if __name__ == "__main__":
    main()
