#!/usr/bin/env python3
"""Export full-domain binary predictions for a trained Haidian downstream head."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from downstreams.data.embedding_dataset import EmbeddingDataset, collate_embeddings
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask, foreground_logits
from downstreams.utils.config import load_config


PATCH_RE = re.compile(r"(patch_\d{6})")


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prediction-title",
        default="Haidian construction-site prediction, labeled patches",
    )
    parser.add_argument(
        "--gt-title",
        default="Haidian construction-site GT, labeled patches",
    )
    return parser.parse_args()


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


def load_patch_ids(label_root: Path) -> list[str]:
    return sorted(patch_id_from_path(p) for p in (label_root / "masks").glob("*.tif"))


def infer_threshold(checkpoint: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    metrics_path = checkpoint.parent.parent / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics.get("val_threshold", metrics.get("best_threshold", 0.5)))
    return 0.5


def save_prediction_geotiffs(
    model: torch.nn.Module,
    loader: DataLoader,
    label_root: Path,
    out_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    prob_dir = out_dir / "probabilities"
    pred_dir = out_dir / "predictions"
    prob_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding_map"].to(device, non_blocking=True)
            logits = foreground_logits(model(emb))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for idx, patch_id in enumerate(batch["patch_ids"]):
                prob = probs[idx].astype(np.float32)
                pred = (prob >= threshold).astype(np.uint8)
                mask_path = resolve_mask(label_root / "masks", patch_id)
                with rasterio.open(mask_path) as src:
                    profile = src.profile.copy()
                profile.update(dtype=rasterio.float32, count=1, nodata=None, compress="lzw")
                with rasterio.open(prob_dir / f"{patch_id}_prob.tif", "w", **profile) as dst:
                    dst.write(prob, 1)
                profile.update(dtype="uint8", nodata=0)
                with rasterio.open(pred_dir / f"{patch_id}_pred.tif", "w", **profile) as dst:
                    dst.write(pred, 1)


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


def red_binary(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = (235, 20, 25)
    return rgb


def save_canvas(
    source_dir: Path,
    label_root: Path,
    output_path: Path,
    title: str,
    suffix: str,
    fallback_gt: bool = False,
) -> None:
    layouts, rows, cols, tile_h, tile_w = load_layout(label_root / "masks")
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 238
    for layout in layouts:
        pred_path = source_dir / f"{layout.patch_id}_{suffix}.tif"
        path = pred_path if pred_path.exists() else layout.mask_path if fallback_gt else None
        if path is None:
            continue
        with rasterio.open(path) as src:
            arr = src.read(1)
        tile = red_binary(arr)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(canvas)
    header = 72
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), title, fill=(0, 0, 0))
    out.paste(image, (0, header))
    out.save(output_path)


def main() -> None:
    args = parse_args()
    device = make_device(args.device)
    cfg = load_config(args.config)
    cfg["training"]["months"] = [args.month]
    cfg["training"]["temporal_mode"] = "single"
    task = ConstructionSegmentationTask(cfg)
    model = task.build_head().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    patch_ids = load_patch_ids(args.label_root)
    dataset = EmbeddingDataset(
        args.embedding_root / args.region,
        args.label_root,
        patch_ids,
        months=[args.month],
        temporal_mode="single",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_embeddings,
    )
    threshold = infer_threshold(args.checkpoint, args.threshold)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "export_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "threshold": threshold,
                "num_patches": len(patch_ids),
                "month": args.month,
                "region": args.region,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_prediction_geotiffs(model, loader, args.label_root, args.output_root, device, threshold)
    save_canvas(
        args.output_root / "predictions",
        args.label_root,
        args.output_root / "visualizations" / "construction_prediction_labeled_patches_geo.png",
        args.prediction_title,
        "pred",
    )
    save_canvas(
        args.label_root / "masks",
        args.label_root,
        args.output_root / "visualizations" / "construction_gt_labeled_patches_geo.png",
        args.gt_title,
        "",
        fallback_gt=True,
    )
    print(args.output_root / "export_meta.json")


if __name__ == "__main__":
    main()
