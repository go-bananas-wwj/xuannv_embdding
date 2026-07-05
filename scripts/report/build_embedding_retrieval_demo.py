#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score


LABEL_ROOTS = {
    "building": Path("/data/xuannv_embedding/processed/haidian/labels/building_osm"),
    "water": Path("/data/xuannv_embedding/processed/haidian/labels/osm_water"),
    "road": Path("/data/xuannv_embedding/processed/haidian/labels/road_osm"),
    "research_gov": Path("/data/xuannv_embedding/processed/haidian/labels/osm_research_gov"),
}

TASK_CN = {
    "building": "Building",
    "water": "Water",
    "road": "Road",
    "research_gov": "Research/Gov",
}


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build query-by-example retrieval demo for embeddings.")
    parser.add_argument("--xuannv-root", type=Path, required=True)
    parser.add_argument("--aef-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--xuannv-month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--tasks", nargs="+", default=["building", "water", "road"])
    parser.add_argument("--query-patches", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def load_layout(label_root: Path) -> tuple[list[PatchLayout], int, int, int, int]:
    mask_paths = sorted((label_root / "masks").glob("patch_*.tif"))
    records: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    height = width = 0
    for path in mask_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            height, width = src.height, src.width
        records.append((path.stem, path, float(bounds.left), float(bounds.top), height, width))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(patch_id, row_of[round(top, 3)], col_of[round(left, 3)], path)
        for patch_id, path, left, top, _, _ in records
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def embedding_path(root: Path, region: str, patch_id: str, month: str) -> Path:
    return root / region / patch_id / f"{month}_embedding_map.pt"


def load_embedding(root: Path, region: str, patch_id: str, month: str) -> torch.Tensor:
    return torch.load(embedding_path(root, region, patch_id, month), map_location="cpu", weights_only=True).float()


def load_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
    if mask.shape != target_hw:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize((target_hw[1], target_hw[0]), Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    return mask


def select_query_layouts(layouts: list[PatchLayout], query_patches: int) -> list[PatchLayout]:
    scored: list[tuple[int, PatchLayout]] = []
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1) > 0
        positive = int(mask.sum())
        if positive > 0:
            scored.append((positive, layout))
    scored.sort(key=lambda item: (-item[0], item[1].patch_id))
    return [layout for _, layout in scored[:query_patches]]


def build_prototype(
    root: Path,
    region: str,
    month: str,
    query_layouts: list[PatchLayout],
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for layout in query_layouts:
        emb = load_embedding(root, region, layout.patch_id, month)
        _, height, width = emb.shape
        mask = load_mask(layout.mask_path, (height, width))
        if mask.any():
            pixels = emb[:, torch.from_numpy(mask)].T
            vectors.append(pixels)
    if not vectors:
        raise ValueError("No positive query pixels found.")
    proto = torch.cat(vectors, dim=0).mean(dim=0)
    return F.normalize(proto, dim=0)


def red_probability(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0), 0.0, 1.0)
    out = np.ones((*prob.shape, 3), dtype=np.float32)
    red = np.array((0.92, 0.05, 0.08), dtype=np.float32)
    return out * (1.0 - prob[..., None]) + red * prob[..., None]


def red_binary(mask: np.ndarray) -> np.ndarray:
    out = np.ones((*mask.shape, 3), dtype=np.float32)
    out[mask.astype(bool)] = (0.92, 0.05, 0.08)
    return out


def paste(canvas: np.ndarray, layout: PatchLayout, tile: np.ndarray, tile_h: int, tile_w: int) -> None:
    y0 = layout.row * tile_h
    x0 = layout.col * tile_w
    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile


def score_full_domain(
    root: Path,
    region: str,
    month: str,
    layouts: list[PatchLayout],
    proto: torch.Tensor,
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_canvas = np.full((rows * tile_h, cols * tile_w), np.nan, dtype=np.float32)
    gt_canvas = np.zeros((rows * tile_h, cols * tile_w), dtype=bool)
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    proto = proto.view(-1, 1, 1)
    for layout in layouts:
        emb = load_embedding(root, region, layout.patch_id, month)
        emb = F.normalize(emb, dim=0)
        score = (emb * proto).sum(dim=0).numpy()
        mask = load_mask(layout.mask_path, score.shape)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        score_canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = score
        gt_canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = mask
        all_scores.append(score.reshape(-1))
        all_labels.append(mask.reshape(-1).astype(np.uint8))
    return score_canvas, gt_canvas, np.stack([np.concatenate(all_scores), np.concatenate(all_labels)])


def normalize_scores(scores: np.ndarray, low_pct: float = 80.0, high_pct: float = 99.5) -> np.ndarray:
    valid = np.isfinite(scores)
    if not valid.any():
        return np.zeros_like(scores, dtype=np.float32)
    lo, hi = np.percentile(scores[valid], [low_pct, high_pct])
    if hi <= lo:
        hi = float(scores[valid].max())
        lo = float(scores[valid].min())
    norm = (scores - lo) / max(hi - lo, 1e-6)
    norm[~valid] = 0.0
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def metrics(scores_and_labels: np.ndarray) -> dict[str, float]:
    scores = scores_and_labels[0]
    labels = scores_and_labels[1].astype(np.uint8)
    if labels.max() == labels.min():
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def task_panel(
    task: str,
    gt_rgb: np.ndarray,
    xuannv_rgb: np.ndarray,
    aef_rgb: np.ndarray,
    query_ids: list[str],
    x_metrics: dict[str, float],
    a_metrics: dict[str, float],
) -> Image.Image:
    target_h = 420
    columns = [
        ("OSM GT", Image.fromarray((gt_rgb * 255).astype(np.uint8))),
        (f"Xuannv cosine\nAUC {x_metrics['auc']:.3f} AP {x_metrics['ap']:.3f}", Image.fromarray((xuannv_rgb * 255).astype(np.uint8))),
        (f"AEF cosine\nAUC {a_metrics['auc']:.3f} AP {a_metrics['ap']:.3f}", Image.fromarray((aef_rgb * 255).astype(np.uint8))),
    ]
    resized: list[tuple[str, Image.Image]] = []
    for title, img in columns:
        scale = target_h / img.height
        resized.append((title, img.resize((int(img.width * scale), target_h), Image.Resampling.BILINEAR)))
    label_w = 250
    gap = 24
    title_h = 92
    width = label_w + sum(img.width for _, img in resized) + gap * (len(resized) - 1)
    height = title_h + target_h
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    title_font = load_font(34)
    label_font = load_font(26)
    small_font = load_font(20)
    draw.text((18, 18), f"{TASK_CN.get(task, task)} / {task}", fill=(0, 0, 0), font=title_font)
    draw.text((18, 56), "query: " + ", ".join(query_ids), fill=(0, 0, 0), font=small_font)
    x = label_w
    for title, img in resized:
        draw.multiline_text((x + 8, 14), title, fill=(0, 0, 0), font=label_font, spacing=4)
        out.paste(img, (x, title_h))
        x += img.width + gap
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    panels: list[Image.Image] = []
    for task in args.tasks:
        label_root = LABEL_ROOTS[task]
        layouts, grid_rows, grid_cols, tile_h, tile_w = load_layout(label_root)
        query_layouts = select_query_layouts(layouts, args.query_patches)
        query_ids = [layout.patch_id for layout in query_layouts]
        x_proto = build_prototype(args.xuannv_root, args.region, args.xuannv_month, query_layouts)
        a_proto = build_prototype(args.aef_root, args.region, args.aef_month, query_layouts)

        x_scores, gt, x_pair = score_full_domain(
            args.xuannv_root, args.region, args.xuannv_month, layouts, x_proto, grid_rows, grid_cols, tile_h, tile_w
        )
        a_scores, _, a_pair = score_full_domain(
            args.aef_root, args.region, args.aef_month, layouts, a_proto, grid_rows, grid_cols, tile_h, tile_w
        )
        x_metrics = metrics(x_pair)
        a_metrics = metrics(a_pair)
        rows.append(
            {
                "task": task,
                "query_patches": ",".join(query_ids),
                "xuannv_auc": x_metrics["auc"],
                "aef_auc": a_metrics["auc"],
                "delta_auc": x_metrics["auc"] - a_metrics["auc"],
                "xuannv_ap": x_metrics["ap"],
                "aef_ap": a_metrics["ap"],
                "delta_ap": x_metrics["ap"] - a_metrics["ap"],
            }
        )
        panels.append(
            task_panel(
                task,
                red_binary(gt),
                red_probability(normalize_scores(x_scores)),
                red_probability(normalize_scores(a_scores)),
                query_ids,
                x_metrics,
                a_metrics,
            )
        )

    width = max(panel.width for panel in panels)
    header_h = 92
    gap = 28
    height = header_h + sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((22, 20), "Training-free query-by-example retrieval on frozen embeddings", fill=(0, 0, 0), font=load_font(38))
    y = header_h
    for panel in panels:
        out.paste(panel, (0, y))
        y += panel.height + gap
    out.save(args.output_root / "embedding_similarity_retrieval.png")

    with (args.output_root / "embedding_similarity_retrieval_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
