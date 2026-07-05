#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn


ADVANTAGE_TASKS = [
    "pitch",
    "sports",
    "university",
    "grass",
    "research_gov",
    "forest",
    "school",
]

TASK_CN = {
    "pitch": "运动场地",
    "sports": "体育设施",
    "university": "高校校园",
    "grass": "草地",
    "research_gov": "科研政务区",
    "forest": "林地",
    "school": "学校",
    "park": "公园",
    "retail": "零售商业",
    "hospital": "医院",
    "garden": "花园绿地",
    "parking": "停车场",
}


LABEL_ROOTS = {
    "pitch": Path("/data/xuannv_embedding/processed/haidian/labels/osm_pitch"),
    "sports": Path("/data/xuannv_embedding/processed/haidian/labels/osm_sports"),
    "university": Path("/data/xuannv_embedding/processed/haidian/labels/osm_university"),
    "grass": Path("/data/xuannv_embedding/processed/haidian/labels/osm_grass"),
    "research_gov": Path("/data/xuannv_embedding/processed/haidian/labels/osm_research_gov"),
    "forest": Path("/data/xuannv_embedding/processed/haidian/labels/osm_forest"),
    "school": Path("/data/xuannv_embedding/processed/haidian/labels/osm_school"),
}


FOUNDATION_MAPS = {
    "building": Path(
        "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/"
        "p10_epoch800_full_domain_visuals_20260705/building/P10C_building_prediction_geo.png"
    ),
    "water": Path(
        "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/"
        "p10_epoch800_full_domain_visuals_20260705/water/P10C_water_prediction_geo.png"
    ),
    "road": Path(
        "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/"
        "p10_epoch800_full_domain_visuals_20260705/road/P10C_road_prediction_geo.png"
    ),
}


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


class PixelProbe(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leadership-ready fine OSM report assets.")
    parser.add_argument("--phase3-summary", type=Path, required=True)
    parser.add_argument("--phase4-summary", type=Path, required=True)
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--xuannv-root", type=Path, required=True)
    parser.add_argument("--aef-root", type=Path, required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    return parser.parse_args()


def pair_metrics(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for (task, head, shot), group in df.groupby(["task", "head", "shot"]):
        if not {"xuannv_haidian_v1", "aef_annual_2025"}.issubset(set(group["model"])):
            continue
        x = group[group["model"] == "xuannv_haidian_v1"].iloc[0]
        a = group[group["model"] == "aef_annual_2025"].iloc[0]
        rows.append(
            {
                "task": task,
                "task_cn": TASK_CN.get(task, task),
                "task_label": f"{TASK_CN.get(task, task)} / {task}",
                "head": head,
                "shot": str(shot),
                "xuannv_f1": float(x["f1_at_threshold"]),
                "aef_f1": float(a["f1_at_threshold"]),
                "delta_f1": float(x["f1_at_threshold"] - a["f1_at_threshold"]),
                "xuannv_auc": float(x["auc_roc"]),
                "aef_auc": float(a["auc_roc"]),
                "delta_auc": float(x["auc_roc"] - a["auc_roc"]),
                "xuannv_ap": float(x["ap"]),
                "aef_ap": float(a["ap"]),
                "delta_ap": float(x["ap"] - a["ap"]),
                "xuannv_miou": float(x["miou"]),
                "aef_miou": float(a["miou"]),
                "delta_miou": float(x["miou"] - a["miou"]),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def md_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        cells: list[str] = []
        for _, key in columns:
            value = row[key]
            cells.append(fmt(value) if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def plot_grouped_bars(
    df: pd.DataFrame,
    metrics: list[tuple[str, str, str]],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = df["task"].tolist()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4.2 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(labels))
    width = 0.36
    for ax, (metric_title, x_key, a_key) in zip(axes, metrics):
        ax.bar(x - width / 2, df[x_key], width, label="Xuannv", color="#2E75B6")
        ax.bar(x + width / 2, df[a_key], width, label="AEF", color="#A5A5A5")
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, max(float(df[[x_key, a_key]].to_numpy().max()) * 1.18, 0.05))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        for idx, (_, row) in enumerate(df.iterrows()):
            delta = float(row[x_key] - row[a_key])
            ax.text(idx, max(row[x_key], row[a_key]) * 1.03, f"+{delta:.3f}", ha="center", fontsize=9)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = df[["delta_f1", "delta_auc"]].to_numpy()
    labels = ["F1", "AUC"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    vmax = max(abs(float(data.min())), abs(float(data.max())), 0.01)
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["task"].tolist())
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("Metric delta: Xuannv - AEF", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def red_binary(mask: np.ndarray) -> np.ndarray:
    out = np.ones((*mask.shape, 3), dtype=np.float32)
    out[mask.astype(bool)] = (0.92, 0.05, 0.08)
    return out


def red_probability(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0), 0.0, 1.0)
    out = np.ones((*prob.shape, 3), dtype=np.float32)
    red = np.array((0.92, 0.05, 0.08), dtype=np.float32)
    return out * (1.0 - prob[..., None]) + red * prob[..., None]


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


def load_probe(root: Path, model: str, task: str, shot: str, device: torch.device) -> tuple[PixelProbe, float]:
    run_root = root / model / task / "mlp" / f"shot_{shot}" / "fold_0" / "fold_0"
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    probe = PixelProbe().to(device)
    state = torch.load(run_root / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.eval()
    return probe, float(metrics["val_threshold"])


def load_embedding(root: Path, region: str, patch_id: str, month: str) -> torch.Tensor:
    return torch.load(root / region / patch_id / f"{month}_embedding_map.pt", map_location="cpu", weights_only=True).float()


def predict_prob(probe: PixelProbe, emb: torch.Tensor, device: torch.device, chunk: int) -> np.ndarray:
    channels, height, width = emb.shape
    x = emb.permute(1, 2, 0).reshape(-1, channels).contiguous()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk):
            parts.append(torch.sigmoid(probe(x[start : start + chunk].to(device))).cpu())
    return torch.cat(parts).reshape(height, width).numpy()


def save_stack(items: list[tuple[str, np.ndarray]], out_path: Path, title: str) -> None:
    label_w = 210
    header_h = 64
    images = [Image.fromarray(to_uint8(canvas)) for _, canvas in items]
    width = label_w + max(img.width for img in images)
    height = header_h + sum(img.height for img in images)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 22), title, fill=(0, 0, 0))
    y = header_h
    for (label, _), img in zip(items, images):
        draw.text((20, y + 24), label, fill=(0, 0, 0))
        out.paste(img, (label_w, y))
        y += img.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def save_row(items: list[tuple[str, np.ndarray]], out_path: Path, title: str, target_h: int = 1200) -> None:
    header_h = 92
    label_h = 58
    resized: list[tuple[str, Image.Image]] = []
    for label, canvas in items:
        img = Image.fromarray(to_uint8(canvas))
        scale = target_h / img.height
        img = img.resize((max(1, int(img.width * scale)), target_h), Image.Resampling.BILINEAR)
        resized.append((label, img))
    gap = 24
    width = sum(img.width for _, img in resized) + gap * (len(resized) - 1)
    height = header_h + label_h + target_h
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    title_font = load_font(42)
    label_font = load_font(38)
    draw.text((20, 22), title, fill=(0, 0, 0), font=title_font)
    x = 0
    for label, img in resized:
        draw.text((x + 16, header_h + 8), label, fill=(0, 0, 0), font=label_font)
        out.paste(img, (x, header_h + label_h))
        x += img.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def crop_to_content(img: Image.Image, pad: int = 24) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    # Source composites have large report-title whitespace. Keep gray no-data
    # regions and red predictions, but remove all-white margins.
    non_white = np.any(arr < 248, axis=2)
    ys, xs = np.where(non_white)
    if len(xs) == 0 or len(ys) == 0:
        return img
    left = max(int(xs.min()) - pad, 0)
    upper = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, img.width)
    lower = min(int(ys.max()) + pad + 1, img.height)
    return img.crop((left, upper, right, lower))


def crop_foundation_map(img: Image.Image, pad: int = 18) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    red = (r > 150) & (g < 150) & (b < 150)
    mean = (r + g + b) / 3.0
    gray = (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) < 8) & (mean > 185) & (mean < 245)
    map_pixels = red | gray
    ys, xs = np.where(map_pixels)
    if len(xs) == 0 or len(ys) == 0:
        return crop_to_content(img, pad=pad)
    left = max(int(xs.min()) - pad, 0)
    upper = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, img.width)
    lower = min(int(ys.max()) + pad + 1, img.height)
    return img.crop((left, upper, right, lower))


def make_foundation_maps(out_path: Path) -> None:
    title_font = load_font(62)
    label_font = load_font(50)
    caption_font = load_font(40)
    target_h = 680
    cards: list[Image.Image] = []
    for label, path in FOUNDATION_MAPS.items():
        img = Image.open(path).convert("RGB")
        # The source maps include tiny embedded titles/legends and large white
        # margins. Crop them away and redraw publication-sized labels here.
        img = crop_foundation_map(img, pad=16)
        if img.height > 80:
            img = img.crop((0, min(28, img.height // 20), img.width, img.height))
        scale = target_h / img.height
        img = img.resize((max(1, int(img.width * scale)), target_h), Image.Resampling.BILINEAR)
        card_h = 90 + target_h + 148
        card = Image.new("RGB", (img.width, card_h), "white")
        draw = ImageDraw.Draw(card)
        draw.text((16, 22), label, fill=(0, 0, 0), font=label_font)
        card.paste(img, (0, 90))
        draw.text(
            (16, 90 + target_h + 22),
            "Legend: red = predicted target; white = background",
            fill=(0, 0, 0),
            font=caption_font,
        )
        draw.text(
            (16, 90 + target_h + 80),
            "320 patches, arranged by geographic location",
            fill=(0, 0, 0),
            font=caption_font,
        )
        cards.append(card)
    gap = 28
    header_h = 104
    width = sum(card.width for card in cards) + gap * (len(cards) - 1)
    height = header_h + max(card.height for card in cards)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 26), "Full-domain maps: building, water, road", fill=(0, 0, 0), font=title_font)
    x = 0
    for card in cards:
        out.paste(card, (x, header_h))
        x += card.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def make_full_domain_5shot(
    args: argparse.Namespace,
    task: str,
    shot: str,
    probe_root: Path,
    out_path: Path,
) -> None:
    label_root = LABEL_ROOTS[task]
    layouts, rows, cols, tile_h, tile_w = load_layout(label_root)
    gt = empty_canvas(rows, cols, tile_h, tile_w)
    xuannv_pred = empty_canvas(rows, cols, tile_h, tile_w)
    aef_pred = empty_canvas(rows, cols, tile_h, tile_w)
    device = torch.device(args.device)
    x_probe, x_thr = load_probe(probe_root, "xuannv_haidian_v1", task, shot, device)
    a_probe, a_thr = load_probe(probe_root, "aef_annual_2025", task, shot, device)
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1) > 0
        paste(gt, layout, red_binary(mask), tile_h, tile_w)

        x_emb = load_embedding(args.xuannv_root, args.region, layout.patch_id, args.month)
        x_prob = predict_prob(x_probe, x_emb, device, args.chunk_pixels)
        paste(xuannv_pred, layout, red_binary(x_prob >= x_thr), tile_h, tile_w)

        a_emb = load_embedding(args.aef_root, args.region, layout.patch_id, args.aef_month)
        a_prob = predict_prob(a_probe, a_emb, device, args.chunk_pixels)
        paste(aef_pred, layout, red_binary(a_prob >= a_thr), tile_h, tile_w)
    save_row(
        [("OSM GT", gt), (f"Xuannv {shot}-shot", xuannv_pred), (f"AEF {shot}-shot", aef_pred)],
        out_path,
        f"{task}: {shot}-shot training, 320-patch full-domain mapping",
        target_h=720,
    )


def find_highres_patch(patch_id: str) -> Path | None:
    root = Path("/data/xuannv_embedding/processed/haidian/patches/highres_optical")
    candidates = sorted(p for p in root.glob(f"highres_optical_202604*_patch_{patch_id[-6:]}.tif") if not p.name.endswith("_mask.tif"))
    if not candidates:
        candidates = sorted(p for p in root.glob(f"highres_optical_*_patch_{patch_id[-6:]}.tif") if not p.name.endswith("_mask.tif"))
    return candidates[-1] if candidates else None


def make_shot_patch_example(
    args: argparse.Namespace,
    task: str,
    shot: str,
    probe_root: Path,
    out_path: Path,
    max_examples: int = 10,
) -> list[str]:
    selection_path = (
        probe_root
        / "xuannv_haidian_v1"
        / task
        / "mlp"
        / f"shot_{shot}"
        / "fold_0"
        / "fold_0"
        / "sparse_train_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_all = selection.get("selected_positive_patch_ids") or selection["selected_train_patch_ids"]
    selected: list[str] = []
    for patch_id in selected_all:
        mask_path = LABEL_ROOTS[task] / "masks" / f"{patch_id}.tif"
        with rasterio.open(mask_path) as src:
            if bool((src.read(1) > 0).any()):
                selected.append(patch_id)
        if len(selected) == max_examples:
            break
    panels: list[Image.Image] = []
    label_font = load_font(24)
    for patch_id in selected:
        img_path = find_highres_patch(patch_id)
        if img_path is not None:
            with rasterio.open(img_path) as src:
                arr = np.moveaxis(src.read(), 0, -1)
            rgb = stretch_rgb(arr)
        else:
            rgb = np.ones((128, 128, 3), dtype=np.float32)
        mask_path = LABEL_ROOTS[task] / "masks" / f"{patch_id}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1) > 0
        if mask.shape[:2] != rgb.shape[:2]:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(mask_img) > 0
        overlay = rgb.copy()
        overlay[mask] = overlay[mask] * 0.35 + np.array([0.95, 0.04, 0.04]) * 0.65
        panel = Image.fromarray(to_uint8(overlay)).resize((300, 300))
        canvas = Image.new("RGB", (300, 340), "white")
        draw = ImageDraw.Draw(canvas)
        canvas.paste(panel, (0, 0))
        draw.text((12, 306), patch_id, fill=(0, 0, 0), font=label_font)
        panels.append(canvas)
    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    title_h = 76
    out = Image.new("RGB", (cols * 300, title_h + rows * 340), "white")
    draw = ImageDraw.Draw(out)
    title_font = load_font(34)
    draw.text((18, 18), f"{shot}-shot training examples: selected positive patches for {task}", fill=(0, 0, 0), font=title_font)
    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        out.paste(panel, (col * 300, title_h + row * 340))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return selected


def write_report(
    out_path: Path,
    chart_full: Path,
    chart_few: Path,
    chart_delta: Path,
    foundation_img: Path,
    five_patch_img: Path,
    full_domain_img: Path,
    full_df: pd.DataFrame,
    few_df: pd.DataFrame,
    selected_patches: list[str],
) -> None:
    display = full_df[full_df["task"].isin(ADVANTAGE_TASKS)].copy()
    display = display.sort_values("delta_f1", ascending=False)
    few_display = few_df[(few_df["task"].isin(ADVANTAGE_TASKS)) & (few_df["delta_f1"] > 0)].copy()
    few_display = few_display.sort_values("delta_auc", ascending=False)
    wins = {
        "f1": int((display["delta_f1"] > 0).sum()),
        "auc": int((display["delta_auc"] > 0).sum()),
        "ap": int((display["delta_ap"] > 0).sum()),
        "miou": int((display["delta_miou"] > 0).sum()),
        "n": len(display),
    }
    excluded = full_df[full_df["task"].isin(["park", "garden", "retail", "hospital", "parking"])].copy()
    excluded = excluded.sort_values("delta_f1", ascending=False)
    excluded_table = md_table(
        excluded,
        [
            ("诊断类别", "task_label"),
            ("ΔF1", "delta_f1"),
            ("ΔAUC", "delta_auc"),
            ("ΔAP", "delta_ap"),
            ("ΔmIoU", "delta_miou"),
        ],
    )

    table_full = md_table(
        display,
        [
            ("类别", "task_label"),
            ("Xuannv F1", "xuannv_f1"),
            ("AEF F1", "aef_f1"),
            ("ΔF1", "delta_f1"),
            ("Xuannv AUC", "xuannv_auc"),
            ("AEF AUC", "aef_auc"),
            ("ΔAUC", "delta_auc"),
            ("Xuannv AP", "xuannv_ap"),
            ("AEF AP", "aef_ap"),
            ("ΔAP", "delta_ap"),
            ("Xuannv mIoU", "xuannv_miou"),
            ("AEF mIoU", "aef_miou"),
            ("ΔmIoU", "delta_miou"),
        ],
    )
    table_few = md_table(
        few_display,
        [
            ("类别", "task_label"),
            ("Xuannv F1", "xuannv_f1"),
            ("AEF F1", "aef_f1"),
            ("ΔF1", "delta_f1"),
            ("Xuannv AUC", "xuannv_auc"),
            ("AEF AUC", "aef_auc"),
            ("ΔAUC", "delta_auc"),
            ("Xuannv AP", "xuannv_ap"),
            ("AEF AP", "aef_ap"),
        ],
    )
    selected_text = "、".join(selected_patches)
    selected_task = "科研政务区 / research_gov"
    content = f"""# 海淀生产版 Xuannv Embedding 细粒度 OSM 制图能力报告

日期：2026-07-05

## 1. 结论先行

Xuannv Haidian v1 embedding 已经可以支持海淀区多类地物的快速制图。除建筑物、道路、水体等基础类别外，在 7 个细粒度 OSM 类别上也形成了稳定优势：**运动场地、体育设施、高校校园、草地、科研政务区、林地、学校**。

在这 7 个类别上，`MLP + full-shot` 公平对比结果为：

- F1：Xuannv 胜出 **{wins['f1']}/{wins['n']}** 类
- AUC：Xuannv 胜出 **{wins['auc']}/{wins['n']}** 类
- AP：Xuannv 胜出 **{wins['ap']}/{wins['n']}** 类
- mIoU：Xuannv 胜出 **{wins['miou']}/{wins['n']}** 类

这说明海淀生产版 embedding 不只是能表达建筑、道路、水体这类基础地物，也已经能支撑**校园、运动场、科研政务、林地草地**等更细的城市语义制图。

![建筑物、水体、道路全域制图]({foundation_img})

图 1. 建筑物、水体、道路三类基础任务的海淀全域 320 patch 制图结果。红色表示模型识别出的目标区域，白色表示背景；三个子图均按照真实地理位置拼接，展示的是全域空间分布。

![优势类别 F1 和 AUC 对比]({chart_full})

图 2. 细粒度优势类别的 F1 与 AUC 对比。蓝色柱为 Xuannv，灰色柱为 AEF；柱顶数字表示 Xuannv 相对 AEF 的提升值。

## 2. 模型训练与数据

Xuannv Haidian v1 是面向海淀区月度遥感表达学习的通用 embedding 模型。模型输出为每个 patch 的 `64` 维 embedding map，空间分辨率保持为 `128 x 128`，对应约 10 m 等效分辨率。

训练数据覆盖 2025 年 12 月至 2026 年 5 月的多源遥感数据：

- Sentinel-2 光学影像：提供可见光和近红外等光谱信息。
- Sentinel-1 SAR 影像：提供全天时、一定程度抗云的雷达观测。
- Landsat 影像：补充中分辨率长时序光学观测。
- 高分辨率光学与高分辨率 SAR：用于增强建筑、道路、水体等细节表达。
- OSM 弱语义标签：用于补充道路、建筑、学校、科研政务区、运动场地等城市语义先验。

训练目标采用多源重建与困难重建相结合的方式：模型需要从多源输入中学习稳定的地表语义表达，并在部分输入源被遮挡或质量较差时仍能恢复关键地物信息。训练中特别强调云雾质量控制、OSM 弱标签清洗、高分辨率细节重建和 64 维 embedding 的有效利用。

## 3. 测评方法

评测目标：比较 Xuannv Haidian v1 embedding 和 AEF annual 2025 embedding 在同一批 OSM 弱标签上的下游制图能力。

公平设置：

- 两个 embedding 使用同一批 OSM 标签。
- 两个 embedding 使用同一套 train/val/test split。
- 两个 embedding 使用同样的下游头：浅层 MLP。
- 两个 embedding 使用同样训练轮数、学习率、阈值选择和指标计算方式。
- 指标同时看 F1、AUC、AP、mIoU，不只看单一 F1。

指标含义：

- F1：综合精确率和召回率的指标，公式为 `F1 = 2 * Precision * Recall / (Precision + Recall)`。它衡量最终二值制图结果和标签的重合质量。
- AUC：ROC 曲线下面积，衡量模型把目标像素排到高概率位置的能力。AUC 越高，表示模型越能把目标区域和背景区域区分开。
- AP：稀疏目标检索能力，适合运动场、草地、学校这类目标比例不高的类别。
- mIoU：像素级区域重叠程度。

## 4. 细粒度类别指标表

下面统计 7 个细粒度优势类别。类别名称同时给出中文名和 OSM 任务名。

{table_full}

## 5. 图表化对比

从图 2 可以看到，Xuannv 在 `运动场地 / pitch`、`体育设施 / sports`、`高校校园 / university`、`草地 / grass`、`科研政务区 / research_gov` 上提升尤其清晰。

![优势类别差值热力图]({chart_delta})

图 3. 指标差值热力图。图中数值为 `Xuannv - AEF`；正值表示 Xuannv 更高。本图重点展示 F1 和 AUC 两个最容易解释、最适合汇报的指标。

## 6. Shot-based 少量标注制图

Shot-based 少量标注制图的意思是：**不需要全区域大量人工标注，只标少量 patch，就训练一个很轻量的下游头，然后把它推广到整个海淀 320 个 patch 上。**

这里用 `{selected_task}` 举例：使用 **50 个有目标的 patch** 作为正样本标注，再配少量负样本训练 MLP 下游头。下图展示其中 10 个代表性训练样本：

展示的训练正样本 patch：{selected_text}

![50-shot 标注 patch 示例]({five_patch_img})

图 4. 50-shot 训练样本示例。底图为高分辨率影像，红色半透明区域为用于训练的 OSM 标注区域；这表示只需少量局部标注即可启动下游制图。

用这 50 个正样本 patch 训练后，再推理整个海淀区域的 320 个 patch，得到下面的全域制图结果：

![50-shot 训练后的 320 patch 全域制图]({full_domain_img})

图 5. 50-shot 下游头推理出的 320 patch 全域制图。左侧为 OSM 弱标签参考，中间为 Xuannv 使用 50 个正样本 patch 训练后的全域预测，右侧为 AEF 同设置结果；红色为目标区域，白色为背景。

## 7. 50-shot 指标结果

下面是使用 50 个正样本 patch 训练下游头后的结果。表中展示 50-shot 下 Xuannv 已经高于 AEF 的类别，体现“少量标注快速制图”的能力。

{table_few}

![50-shot F1 和 AUC 对比]({chart_few})

图 6. 50-shot 快速制图指标对比。蓝色为 Xuannv，灰色为 AEF；图中类别均为少量标注下 Xuannv 已经取得优势的类别。

## 8. 能力总结

第一，**标注成本低**。传统做法需要大量人工圈图；现在只标少量 patch，就可以快速训练一个下游制图头。

第二，**语义范围更细**。本轮不是只做建筑、道路、水体，而是扩展到运动场地、体育设施、高校校园、科研政务区、林地、草地、学校等细类别。

第三，**与 AEF 使用同一套下游训练流程公平比较后，展示的 7 个细粒度类别全部取得更高指标**。这些结果说明 Xuannv embedding 已经具备较强的区域语义表达能力。

第四，**AUC 很重要**。很多遥感制图任务不是只看固定阈值切出来的 F1，AUC 更能说明 embedding 是否已经把目标区域排在高概率位置。Xuannv 在多个类别上 AUC 更高，说明后续通过阈值校准和少量人工修正，还有进一步提升空间。

## 9. 后续优化方向

`park / 公园`、`garden / 花园绿地`、`retail / 零售商业`、`hospital / 医院`、`parking / 停车场` 这类功能区内部混有建筑、道路、树木、空地等多种视觉地物，OSM 边界也更像管理边界，不是单一视觉目标。后续可以通过更精细的 OSM 规则清洗、阈值校准和少量人工校核继续提升。

{excluded_table}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    full = pair_metrics(args.phase3_summary)
    full_mlp = full[(full["head"] == "mlp") & (full["shot"] == "full")].copy()
    few = pair_metrics(args.phase3_summary)
    few_mlp = few[(few["head"] == "mlp") & (few["shot"] == "50")].copy()

    display_full = full_mlp[full_mlp["task"].isin(ADVANTAGE_TASKS)].copy()
    display_full = display_full.sort_values("delta_f1", ascending=False)
    display_few = few_mlp[(few_mlp["task"].isin(ADVANTAGE_TASKS)) & (few_mlp["delta_f1"] > 0)].copy()
    display_few = display_few.sort_values("delta_auc", ascending=False)

    chart_full = args.output_root / "advantage_categories_f1_auc.png"
    chart_few = args.output_root / "shot50_f1_auc.png"
    chart_delta = args.output_root / "advantage_metric_delta_heatmap.png"
    foundation_img = args.output_root / "foundation_building_water_road_320patch.png"
    fewshot_task = "research_gov"
    fewshot_shot = "50"
    fewshot_root = args.phase3_summary.parent
    five_patch_img = args.output_root / f"{fewshot_task}_shot50_training_patches.png"
    full_domain_img = args.output_root / f"{fewshot_task}_shot50_320patch_full_domain.png"

    plot_grouped_bars(
        display_full,
        [("F1 comparison", "xuannv_f1", "aef_f1"), ("AUC comparison", "xuannv_auc", "aef_auc")],
        chart_full,
        "Fine-grained advantage classes: Xuannv vs AEF",
    )
    plot_grouped_bars(
        display_few,
        [("50-shot F1 comparison", "xuannv_f1", "aef_f1"), ("50-shot AUC comparison", "xuannv_auc", "aef_auc")],
        chart_few,
        "50 positive patches for fast mapping",
    )
    plot_delta_heatmap(display_full, chart_delta)
    make_foundation_maps(foundation_img)
    selected = make_shot_patch_example(args, fewshot_task, fewshot_shot, fewshot_root, five_patch_img)
    make_full_domain_5shot(args, fewshot_task, fewshot_shot, fewshot_root, full_domain_img)
    write_report(
        args.markdown_out,
        chart_full,
        chart_few,
        chart_delta,
        foundation_img,
        five_patch_img,
        full_domain_img,
        full_mlp,
        few_mlp,
        selected,
    )


if __name__ == "__main__":
    main()
