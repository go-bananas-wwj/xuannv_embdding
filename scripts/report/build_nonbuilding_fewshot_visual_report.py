#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.eval.train_aef_downstream_probe import PixelProbe


LABEL_ROOTS = {
    "education": Path("/data/xuannv_embedding/processed/haidian/labels/osm_education"),
    "sports": Path("/data/xuannv_embedding/processed/haidian/labels/osm_sports"),
    "pitch": Path("/data/xuannv_embedding/processed/haidian/labels/osm_pitch"),
    "park": Path("/data/xuannv_embedding/processed/haidian/labels/osm_park"),
    "grass": Path("/data/xuannv_embedding/processed/haidian/labels/osm_grass"),
}

TASK_TITLES = {
    "education": "Education (University + School)",
    "sports": "Sports Facilities",
    "pitch": "Pitch / Playground Fields",
    "park": "Park",
    "grass": "Grass / Green Space",
}


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-building few-shot visual report.")
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--xuannv-root", type=Path, required=True)
    parser.add_argument("--aef-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tasks", nargs="+", default=["education", "sports", "pitch"])
    parser.add_argument("--shots", nargs="+", default=["5", "10", "50"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--example-shot", default="50")
    parser.add_argument("--example-count", type=int, default=3)
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


def empty_canvas(rows: int, cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    return np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.float32)


def paste(canvas: np.ndarray, layout: PatchLayout, tile: np.ndarray, tile_h: int, tile_w: int) -> None:
    y0 = layout.row * tile_h
    x0 = layout.col * tile_w
    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile


def red_binary(mask: np.ndarray) -> np.ndarray:
    out = np.ones((*mask.shape, 3), dtype=np.float32)
    out[mask.astype(bool)] = (0.92, 0.05, 0.08)
    return out


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    out = np.zeros_like(arr[..., :3], dtype=np.float32)
    for c in range(3):
        band = arr[..., c]
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.percentile(band[finite], [2, 98])
        if hi > lo:
            out[..., c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def overlay_red(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.62) -> np.ndarray:
    out = rgb.copy()
    if mask.shape != out.shape[:2]:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
            (out.shape[1], out.shape[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_img) > 0
    out[mask] = out[mask] * (1.0 - alpha) + np.array([0.95, 0.04, 0.04], dtype=np.float32) * alpha
    return out


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def metric_row(summary: pd.DataFrame, model: str, task: str, head: str, shot: str) -> pd.Series:
    rows = summary[
        (summary["model"] == model)
        & (summary["task"] == task)
        & (summary["head"] == head)
        & (summary["shot"].astype(str) == str(shot))
    ]
    if rows.empty:
        raise KeyError((model, task, head, shot))
    return rows.iloc[0]


def best_head_for(summary: pd.DataFrame, task: str, shot: str) -> str:
    rows = summary[
        (summary["model"] == "xuannv_haidian_v1")
        & (summary["task"] == task)
        & (summary["shot"].astype(str) == str(shot))
    ].copy()
    if rows.empty:
        raise KeyError((task, shot))
    rows = rows.sort_values("f1_at_threshold", ascending=False)
    return str(rows.iloc[0]["head"])


def probe_root(suite_root: Path, model: str, task: str, head: str, shot: str, fold: int) -> Path:
    return suite_root / model / task / head / f"shot_{shot}" / f"fold_{fold}" / f"fold_{fold}"


def load_probe(
    suite_root: Path,
    model: str,
    task: str,
    head: str,
    shot: str,
    fold: int,
    hidden_dim: int,
    device: torch.device,
) -> tuple[PixelProbe, float]:
    root = probe_root(suite_root, model, task, head, shot, fold)
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    model_obj = PixelProbe(embed_dim=64, head=head, hidden_dim=hidden_dim).to(device)
    state = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    model_obj.load_state_dict(state)
    model_obj.eval()
    return model_obj, float(metrics["val_threshold"])


def load_embedding(root: Path, region: str, patch_id: str, month: str) -> torch.Tensor:
    return torch.load(root / region / patch_id / f"{month}_embedding_map.pt", map_location="cpu", weights_only=True).float()


def predict_prob(model: PixelProbe, emb: torch.Tensor, device: torch.device, chunk_pixels: int) -> np.ndarray:
    channels, height, width = emb.shape
    x = emb.permute(1, 2, 0).reshape(-1, channels).contiguous()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_pixels):
            parts.append(torch.sigmoid(model(x[start : start + chunk_pixels].to(device))).cpu())
    return torch.cat(parts).reshape(height, width).numpy()


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_hw:
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
        (target_hw[1], target_hw[0]), Image.Resampling.NEAREST
    )
    return np.asarray(mask_img) > 0


def prediction_canvas(
    embedding_root: Path,
    region: str,
    month: str,
    probe: PixelProbe,
    threshold: float,
    device: torch.device,
    chunk_pixels: int,
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    canvas = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        emb = load_embedding(embedding_root, region, layout.patch_id, month)
        prob = predict_prob(probe, emb, device, chunk_pixels)
        paste(canvas, layout, red_binary(prob >= threshold), tile_h, tile_w)
    return canvas


def gt_canvas(layouts: list[PatchLayout], rows: int, cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    canvas = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1) > 0
        paste(canvas, layout, red_binary(mask), tile_h, tile_w)
    return canvas


def find_highres_patch(patch_id: str) -> Path | None:
    root = Path("/data/xuannv_embedding/processed/haidian/patches/highres_optical")
    suffix = patch_id[-6:]
    candidates = sorted(p for p in root.glob(f"highres_optical_202604*_patch_{suffix}.tif") if not p.name.endswith("_mask.tif"))
    if not candidates:
        candidates = sorted(p for p in root.glob(f"highres_optical_*_patch_{suffix}.tif") if not p.name.endswith("_mask.tif"))
    return candidates[-1] if candidates else None


def load_highres_rgb(patch_id: str, fallback_hw: tuple[int, int]) -> np.ndarray:
    path = find_highres_patch(patch_id)
    if path is None:
        return np.ones((*fallback_hw, 3), dtype=np.float32)
    with rasterio.open(path) as src:
        arr = np.moveaxis(src.read(), 0, -1)
    return stretch_rgb(arr)


def binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def choose_example_patches(
    args: argparse.Namespace,
    task: str,
    head: str,
    shot: str,
    layouts: list[PatchLayout],
    x_probe: PixelProbe,
    x_threshold: float,
    device: torch.device,
) -> list[PatchLayout]:
    scored: list[tuple[float, int, PatchLayout]] = []
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            gt = src.read(1) > 0
        if int(gt.sum()) < 80:
            continue
        emb = load_embedding(args.xuannv_root, args.region, layout.patch_id, args.month)
        prob = predict_prob(x_probe, emb, device, args.chunk_pixels)
        gt = resize_mask(gt, prob.shape)
        score = binary_f1(prob >= x_threshold, gt)
        scored.append((score, int(gt.sum()), layout))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2].patch_id))
    return [layout for _, _, layout in scored[: args.example_count]]


def build_patch_examples(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    task: str,
    layouts: list[PatchLayout],
) -> Path:
    device = torch.device(args.device)
    head = best_head_for(summary, task, args.example_shot)
    x_probe, x_threshold = load_probe(
        args.suite_root,
        "xuannv_haidian_v1",
        task,
        head,
        args.example_shot,
        args.fold,
        args.hidden_dim,
        device,
    )
    a_probe, a_threshold = load_probe(
        args.suite_root,
        "aef_annual_2025",
        task,
        head,
        args.example_shot,
        args.fold,
        args.hidden_dim,
        device,
    )
    examples = choose_example_patches(args, task, head, args.example_shot, layouts, x_probe, x_threshold, device)
    title_font = load_font(30)
    label_font = load_font(22)
    tile = 230
    row_h = tile + 54
    col_w = tile
    gap = 18
    header_h = 76
    columns = ["Optical", "OSM GT", "Xuannv", "AEF"]
    width = len(columns) * col_w + (len(columns) - 1) * gap
    height = header_h + len(examples) * row_h
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((12, 18), f"{TASK_TITLES.get(task, task)} patch examples ({args.example_shot} positive patches)", fill=(0, 0, 0), font=title_font)
    for row_idx, layout in enumerate(examples):
        emb_x = load_embedding(args.xuannv_root, args.region, layout.patch_id, args.month)
        emb_a = load_embedding(args.aef_root, args.region, layout.patch_id, args.aef_month)
        prob_x = predict_prob(x_probe, emb_x, device, args.chunk_pixels)
        prob_a = predict_prob(a_probe, emb_a, device, args.chunk_pixels)
        with rasterio.open(layout.mask_path) as src:
            gt = src.read(1) > 0
        gt = resize_mask(gt, prob_x.shape)
        rgb = load_highres_rgb(layout.patch_id, gt.shape)
        panels = [
            Image.fromarray(to_uint8(rgb)),
            Image.fromarray(to_uint8(overlay_red(rgb, gt))),
            Image.fromarray(to_uint8(overlay_red(rgb, prob_x >= x_threshold))),
            Image.fromarray(to_uint8(overlay_red(rgb, prob_a >= a_threshold))),
        ]
        y0 = header_h + row_idx * row_h
        for col_idx, (label, panel) in enumerate(zip(columns, panels)):
            x0 = col_idx * (col_w + gap)
            panel = panel.resize((tile, tile), Image.Resampling.BILINEAR)
            if row_idx == 0:
                draw.text((x0 + 8, header_h - 30), label, fill=(0, 0, 0), font=label_font)
            out.paste(panel, (x0, y0))
            if col_idx == 0:
                draw.text((x0 + 8, y0 + tile + 8), layout.patch_id, fill=(0, 0, 0), font=label_font)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / f"{task}_patch_examples_{args.example_shot}shot.png"
    out.save(out_path)
    return out_path


def resize_map(canvas: np.ndarray, target_h: int = 300) -> Image.Image:
    img = Image.fromarray(to_uint8(canvas))
    scale = target_h / img.height
    return img.resize((max(1, int(img.width * scale)), target_h), Image.Resampling.BILINEAR)


def build_task_figure(args: argparse.Namespace, summary: pd.DataFrame, task: str) -> Path:
    layouts, grid_rows, grid_cols, tile_h, tile_w = load_layout(LABEL_ROOTS[task])
    gt = gt_canvas(layouts, grid_rows, grid_cols, tile_h, tile_w)
    device = torch.device(args.device)
    title_font = load_font(34)
    label_font = load_font(24)
    small_font = load_font(21)
    row_images: list[Image.Image] = []
    for shot in args.shots:
        head = best_head_for(summary, task, shot)
        x = metric_row(summary, "xuannv_haidian_v1", task, head, shot)
        a = metric_row(summary, "aef_annual_2025", task, head, shot)
        x_probe, x_threshold = load_probe(
            args.suite_root,
            "xuannv_haidian_v1",
            task,
            head,
            shot,
            args.fold,
            args.hidden_dim,
            device,
        )
        a_probe, a_threshold = load_probe(
            args.suite_root,
            "aef_annual_2025",
            task,
            head,
            shot,
            args.fold,
            args.hidden_dim,
            device,
        )
        x_map = prediction_canvas(
            args.xuannv_root,
            args.region,
            args.month,
            x_probe,
            x_threshold,
            device,
            args.chunk_pixels,
            layouts,
            grid_rows,
            grid_cols,
            tile_h,
            tile_w,
        )
        a_map = prediction_canvas(
            args.aef_root,
            args.region,
            args.aef_month,
            a_probe,
            a_threshold,
            device,
            args.chunk_pixels,
            layouts,
            grid_rows,
            grid_cols,
            tile_h,
            tile_w,
        )
        maps = [
            ("OSM GT\nred=target", gt),
            (f"Xuannv {shot} pos patches\nF1 {x['f1_at_threshold']:.3f} AUC {x['auc_roc']:.3f}", x_map),
            (f"AEF {shot} pos patches\nF1 {a['f1_at_threshold']:.3f} AUC {a['auc_roc']:.3f}", a_map),
        ]
        resized = [(label, resize_map(canvas)) for label, canvas in maps]
        gap = 24
        row_title_w = 170
        label_h = 70
        row_h = label_h + max(img.height for _, img in resized)
        row_w = row_title_w + sum(img.width for _, img in resized) + gap * (len(resized) - 1)
        row = Image.new("RGB", (row_w, row_h), "white")
        draw = ImageDraw.Draw(row)
        draw.text((12, label_h + 92), f"{shot}\npos patch", fill=(0, 0, 0), font=label_font)
        x0 = row_title_w
        for label, img in resized:
            draw.multiline_text((x0 + 8, 6), label, fill=(0, 0, 0), font=small_font, spacing=3)
            row.paste(img, (x0, label_h))
            x0 += img.width + gap
        row_images.append(row)

    header_h = 96
    width = max(row.width for row in row_images)
    height = header_h + sum(row.height for row in row_images) + 24 * (len(row_images) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((18, 22), f"{TASK_TITLES.get(task, task)}: full-domain 320-patch mapping", fill=(0, 0, 0), font=title_font)
    y = header_h
    for row in row_images:
        out.paste(row, (0, y))
        y += row.height + 24
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / f"{task}_fewshot_5_10_50_visual.png"
    out.save(out_path)
    return out_path


def paired_metrics(summary: pd.DataFrame, tasks: list[str], shots: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in tasks:
        for shot in shots:
            head = best_head_for(summary, task, shot)
            x = metric_row(summary, "xuannv_haidian_v1", task, head, shot)
            a = metric_row(summary, "aef_annual_2025", task, head, shot)
            rows.append(
                {
                    "task": task,
                    "shot": shot,
                    "head": head,
                    "xuannv_f1": float(x["f1_at_threshold"]),
                    "aef_f1": float(a["f1_at_threshold"]),
                    "delta_f1": float(x["f1_at_threshold"] - a["f1_at_threshold"]),
                    "xuannv_auc": float(x["auc_roc"]),
                    "aef_auc": float(a["auc_roc"]),
                    "delta_auc": float(x["auc_roc"] - a["auc_roc"]),
                    "xuannv_ap": float(x["ap"]),
                    "aef_ap": float(a["ap"]),
                    "delta_ap": float(x["ap"] - a["ap"]),
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = ["task", "shot", "head", "xuannv_f1", "aef_f1", "delta_f1", "xuannv_auc", "aef_auc", "delta_auc", "xuannv_ap", "aef_ap", "delta_ap"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.suite_root / "summary.csv")
    figures = {}
    examples = {}
    for task in args.tasks:
        figures[task] = build_task_figure(args, summary, task)
        layouts, *_ = load_layout(LABEL_ROOTS[task])
        examples[task] = build_patch_examples(args, summary, task, layouts)
    metrics = paired_metrics(summary, args.tasks, args.shots)
    metrics.to_csv(args.output_root / "nonbuilding_fewshot_paired_metrics.csv", index=False)
    lines = [
        "# 玄女海淀 V1 非建筑语义制图能力展示",
        "",
        "本报告展示玄女海淀 V1 embedding 在非建筑语义类别上的少样本制图能力。评估对象包括教育区域、体育设施、操场/球场、公园和草地/绿地，避免使用建筑类作为主展示对象。",
        "",
        "评估方式为固定 embedding 后训练轻量下游头。每个类别分别使用 5、10、50 个正样本 patch 训练，并在海淀区 320 个 patch 上进行全域推理。玄女与 AEF 使用同一批 OSM 标签、同一数据划分、同一训练设置和同一阈值选择规则。",
        "",
        "图中红色表示目标类别，白色表示非目标区域。单 patch 图叠加高分辨率光学影像，用于观察预测区域是否落在真实地物上。",
        "",
        md_table(metrics),
        "",
    ]
    for task, path in figures.items():
        rel = path.name
        ex_rel = examples[task].name
        lines += [
            f"## {TASK_TITLES.get(task, task)}",
            "",
            "### 全域 320 patch 制图",
            "",
            f"![{task}]({rel})",
            "",
            "### 典型 patch 高分影像叠加",
            "",
            f"![{task} examples]({ex_rel})",
            "",
        ]
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
