#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from scripts.report.build_nonbuilding_fewshot_visual_report import (
    LABEL_ROOTS,
    find_highres_patch,
    load_font,
    stretch_rgb,
    to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a product-style ROI retrieval case.")
    parser.add_argument("--xuannv-root", type=Path, required=True)
    parser.add_argument("--aef-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--task", default="pitch")
    parser.add_argument("--query-patch", default=None)
    parser.add_argument("--bbox-pad", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_embedding(root: Path, region: str, patch_id: str, month: str) -> torch.Tensor:
    return torch.load(root / region / patch_id / f"{month}_embedding_map.pt", map_location="cpu", weights_only=True).float()


def load_mask(task: str, patch_id: str, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    path = LABEL_ROOTS[task] / "masks" / f"{patch_id}.tif"
    with rasterio.open(path) as src:
        mask = src.read(1) > 0
    if target_hw is not None and mask.shape != target_hw:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
            (target_hw[1], target_hw[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_img) > 0
    return mask


def choose_query_patch(task: str) -> str:
    scored: list[tuple[int, str]] = []
    for path in sorted((LABEL_ROOTS[task] / "masks").glob("patch_*.tif")):
        with rasterio.open(path) as src:
            mask = src.read(1) > 0
        count = int(mask.sum())
        # Avoid huge mixed regions and tiny specks for a clean visual query.
        if 120 <= count <= 2500:
            scored.append((count, path.stem))
    if not scored:
        raise RuntimeError(f"No suitable query patch for {task}")
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def bbox_from_mask(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    try:
        from scipy import ndimage

        labeled, num = ndimage.label(mask.astype(np.uint8))
        if num > 0:
            counts = np.bincount(labeled.reshape(-1))
            counts[0] = 0
            mask = labeled == int(counts.argmax())
    except Exception:
        pass
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return y0, y1, x0, x1


def prototype(root: Path, region: str, month: str, patch_id: str, task: str, bbox_pad: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    emb = load_embedding(root, region, patch_id, month)
    mask = load_mask(task, patch_id, emb.shape[-2:])
    bbox = bbox_from_mask(mask, bbox_pad)
    y0, y1, x0, x1 = bbox
    pixels = emb[:, y0:y1, x0:x1].reshape(emb.shape[0], -1).T
    proto = F.normalize(pixels.mean(dim=0), dim=0)
    return proto, bbox


def patch_score(emb: torch.Tensor, proto: torch.Tensor) -> float:
    emb = F.normalize(emb, dim=0)
    sim = (emb * proto.view(-1, 1, 1)).sum(dim=0).numpy()
    return float(np.percentile(sim, 99.5))


def rank_patches(root: Path, region: str, month: str, task: str, proto: torch.Tensor, query_patch: str, top_k: int) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for path in sorted((LABEL_ROOTS[task] / "masks").glob("patch_*.tif")):
        patch_id = path.stem
        if patch_id == query_patch:
            continue
        emb_path = root / region / patch_id / f"{month}_embedding_map.pt"
        if not emb_path.exists():
            continue
        emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
        rows.append((patch_id, patch_score(emb, proto)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows[:top_k]


def load_rgb(patch_id: str, size: int = 180) -> Image.Image:
    path = find_highres_patch(patch_id)
    if path is None:
        return Image.new("RGB", (size, size), "white")
    with rasterio.open(path) as src:
        arr = np.moveaxis(src.read(), 0, -1)
    return Image.fromarray(to_uint8(stretch_rgb(arr))).resize((size, size), Image.Resampling.BILINEAR)


def draw_query(task: str, patch_id: str, bbox: tuple[int, int, int, int], size: int = 270) -> Image.Image:
    img = load_rgb(patch_id, size=size)
    mask = load_mask(task, patch_id)
    h, w = mask.shape
    y0, y1, x0, x1 = bbox
    sx = size / w
    sy = size / h
    draw = ImageDraw.Draw(img)
    draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=(230, 0, 20), width=4)
    return img


def candidate_grid(title: str, candidates: list[tuple[str, float]], top_k: int) -> Image.Image:
    thumb = 150
    cols = 4
    rows = int(np.ceil(top_k / cols))
    title_h = 54
    label_h = 34
    out = Image.new("RGB", (cols * thumb, title_h + rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(out)
    title_font = load_font(24)
    small_font = load_font(16)
    draw.text((8, 14), title, fill=(0, 0, 0), font=title_font)
    for idx, (patch_id, score) in enumerate(candidates):
        row = idx // cols
        col = idx % cols
        x = col * thumb
        y = title_h + row * (thumb + label_h)
        out.paste(load_rgb(patch_id, size=thumb), (x, y))
        draw.text((x + 4, y + thumb + 6), f"{idx+1}. {patch_id}  {score:.3f}", fill=(0, 0, 0), font=small_font)
    return out


def main() -> None:
    args = parse_args()
    query_patch = args.query_patch or choose_query_patch(args.task)
    x_proto, bbox = prototype(args.xuannv_root, args.region, args.month, query_patch, args.task, args.bbox_pad)
    a_proto, _ = prototype(args.aef_root, args.region, args.aef_month, query_patch, args.task, args.bbox_pad)
    x_rank = rank_patches(args.xuannv_root, args.region, args.month, args.task, x_proto, query_patch, args.top_k)
    a_rank = rank_patches(args.aef_root, args.region, args.aef_month, args.task, a_proto, query_patch, args.top_k)

    title_font = load_font(30)
    body_font = load_font(20)
    query_img = draw_query(args.task, query_patch, bbox, size=270)
    x_grid = candidate_grid("Xuannv Top-8 candidates", x_rank, args.top_k)
    a_grid = candidate_grid("AEF Top-8 candidates", a_rank, args.top_k)
    gap = 28
    header_h = 84
    width = max(320 + gap + x_grid.width, 320 + gap + a_grid.width)
    height = header_h + max(query_img.height + 90, x_grid.height + a_grid.height + gap)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((18, 18), "ROI query retrieval case: click one pitch and retrieve similar areas", fill=(0, 0, 0), font=title_font)
    draw.text((18, header_h), f"Query ROI: {query_patch}", fill=(0, 0, 0), font=body_font)
    out.paste(query_img, (18, header_h + 34))
    draw.text((18, header_h + 34 + query_img.height + 8), "Red box = user selected ROI", fill=(0, 0, 0), font=body_font)
    x0 = 320 + gap
    out.paste(x_grid, (x0, header_h))
    out.paste(a_grid, (x0, header_h + x_grid.height + gap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)


if __name__ == "__main__":
    main()
