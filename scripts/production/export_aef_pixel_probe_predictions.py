#!/usr/bin/env python3
"""Export full-domain probabilities for pixel-wise Xuannv/AEF probe heads."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from torch import nn


PATCH_RE = re.compile(r"(patch_\d{6})")


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--head", choices=["linear", "mlp", "mlp_deep"], default="mlp")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--title", default="Full-domain prediction")
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
    candidates = sorted(mask_dir.glob(f"*{patch_id}*.tif"))
    if candidates:
        return candidates[-1]
    return exact


def load_patch_ids(label_root: Path) -> list[str]:
    return sorted(patch_id_from_path(path) for path in (label_root / "masks").glob("*.tif"))


def infer_threshold(checkpoint: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    metrics_path = checkpoint.parent.parent / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics.get("val_threshold", metrics.get("best_threshold", 0.5)))
    return 0.5


def load_embedding(path: Path, l2_normalize: bool) -> torch.Tensor:
    emb = torch.load(path, map_location="cpu", weights_only=True).float()
    if emb.ndim != 3:
        raise ValueError(f"Expected (D,H,W) embedding, got {tuple(emb.shape)}: {path}")
    if l2_normalize:
        emb = emb / emb.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return emb


def export_probabilities(
    model: nn.Module,
    patch_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
) -> None:
    prob_dir = args.output_root / "probabilities"
    pred_dir = args.output_root / "predictions"
    prob_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    region_root = args.embedding_root / args.region
    model.eval()
    with torch.no_grad():
        for idx, patch_id in enumerate(patch_ids, start=1):
            emb_path = region_root / patch_id / f"{args.month}_embedding_map.pt"
            mask_path = resolve_mask(args.label_root / "masks", patch_id)
            emb = load_embedding(emb_path, l2_normalize=args.l2_normalize)
            h, w = emb.shape[-2:]
            x = emb.permute(1, 2, 0).reshape(-1, emb.shape[0]).contiguous()
            parts: list[torch.Tensor] = []
            for start in range(0, x.shape[0], args.chunk_pixels):
                xb = x[start : start + args.chunk_pixels].to(device, non_blocking=True)
                parts.append(torch.sigmoid(model(xb)).detach().cpu())
            prob = torch.cat(parts, dim=0).reshape(h, w).numpy().astype(np.float32)
            pred = (prob >= threshold).astype(np.uint8)
            with rasterio.open(mask_path) as src:
                profile = src.profile.copy()
            profile.update(dtype=rasterio.float32, count=1, nodata=None, compress="lzw")
            with rasterio.open(prob_dir / f"{patch_id}_prob.tif", "w", **profile) as dst:
                dst.write(prob, 1)
            profile.update(dtype="uint8", nodata=0)
            with rasterio.open(pred_dir / f"{patch_id}_pred.tif", "w", **profile) as dst:
                dst.write(pred, 1)
            if idx % 50 == 0 or idx == len(patch_ids):
                print(f"exported {idx}/{len(patch_ids)} patches")


def load_layout(mask_dir: Path) -> tuple[list[PatchLayout], int, int, int, int]:
    paths = sorted(mask_dir.glob("*.tif"))
    raw: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    for path in paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
        patch_id = patch_id_from_path(path)
        raw.append((patch_id, path, float(bounds.left), float(bounds.top), width, height))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(patch_id, row_of[round(top, 3)], col_of[round(left, 3)], path)
        for patch_id, path, left, top, _width, _height in raw
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def red_binary(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = (235, 20, 25)
    return rgb


def save_canvas(source_dir: Path, label_root: Path, output_path: Path, title: str, suffix: str) -> None:
    layouts, rows, cols, tile_h, tile_w = load_layout(label_root / "masks")
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 238
    for layout in layouts:
        path = source_dir / f"{layout.patch_id}_{suffix}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = red_binary(arr)
    image = Image.fromarray(canvas)
    header = 72
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), title, fill=(0, 0, 0))
    out.paste(image, (0, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def main() -> None:
    args = parse_args()
    device = make_device(args.device)
    threshold = infer_threshold(args.checkpoint, args.threshold)
    model = PixelProbe(args.embed_dim, args.head, args.hidden_dim).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    patch_ids = load_patch_ids(args.label_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "export_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "embedding_root": str(args.embedding_root),
                "label_root": str(args.label_root),
                "region": args.region,
                "month": args.month,
                "head": args.head,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "threshold": threshold,
                "num_patches": len(patch_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    export_probabilities(model, patch_ids, args, device, threshold)
    save_canvas(
        args.output_root / "predictions",
        args.label_root,
        args.output_root / "visualizations" / "prediction_full_domain_geo.png",
        args.title,
        "pred",
    )
    print(args.output_root / "export_meta.json")


if __name__ == "__main__":
    main()
