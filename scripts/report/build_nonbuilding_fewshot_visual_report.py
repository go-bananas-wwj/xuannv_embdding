#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont


LABEL_ROOTS = {
    "education": Path("/data/xuannv_embedding/processed/haidian/labels/osm_education"),
    "sports": Path("/data/xuannv_embedding/processed/haidian/labels/osm_sports"),
    "pitch": Path("/data/xuannv_embedding/processed/haidian/labels/osm_pitch"),
    "park": Path("/data/xuannv_embedding/processed/haidian/labels/osm_park"),
    "grass": Path("/data/xuannv_embedding/processed/haidian/labels/osm_grass"),
}

TASK_TITLES = {
    "education": "Education = University OR School",
    "sports": "Sports facilities",
    "pitch": "Pitch / playground-like fields",
    "park": "Park",
    "grass": "Grass",
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["education", "sports", "pitch"])
    parser.add_argument("--shots", nargs="+", default=["5", "10", "50"])
    parser.add_argument("--fold", type=int, default=0)
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


def prediction_canvas(
    suite_root: Path,
    model: str,
    task: str,
    head: str,
    shot: str,
    fold: int,
    threshold: float,
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    canvas = empty_canvas(rows, cols, tile_h, tile_w)
    pred_dir = suite_root / model / task / head / f"shot_{shot}" / f"fold_{fold}" / f"fold_{fold}" / "predictions"
    for layout in layouts:
        path = pred_dir / f"{layout.patch_id}_prob.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            prob = src.read(1)
        paste(canvas, layout, red_binary(prob >= threshold), tile_h, tile_w)
    return canvas


def gt_canvas(layouts: list[PatchLayout], rows: int, cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    canvas = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1) > 0
        paste(canvas, layout, red_binary(mask), tile_h, tile_w)
    return canvas


def resize_map(canvas: np.ndarray, target_h: int = 300) -> Image.Image:
    img = Image.fromarray(to_uint8(canvas))
    scale = target_h / img.height
    return img.resize((max(1, int(img.width * scale)), target_h), Image.Resampling.BILINEAR)


def build_task_figure(args: argparse.Namespace, summary: pd.DataFrame, task: str) -> Path:
    layouts, grid_rows, grid_cols, tile_h, tile_w = load_layout(LABEL_ROOTS[task])
    gt = gt_canvas(layouts, grid_rows, grid_cols, tile_h, tile_w)
    title_font = load_font(34)
    label_font = load_font(24)
    small_font = load_font(18)
    row_images: list[Image.Image] = []
    for shot in args.shots:
        head = best_head_for(summary, task, shot)
        x = metric_row(summary, "xuannv_haidian_v1", task, head, shot)
        a = metric_row(summary, "aef_annual_2025", task, head, shot)
        x_map = prediction_canvas(
            args.suite_root,
            "xuannv_haidian_v1",
            task,
            head,
            shot,
            args.fold,
            float(x["val_threshold"]),
            layouts,
            grid_rows,
            grid_cols,
            tile_h,
            tile_w,
        )
        a_map = prediction_canvas(
            args.suite_root,
            "aef_annual_2025",
            task,
            head,
            shot,
            args.fold,
            float(a["val_threshold"]),
            layouts,
            grid_rows,
            grid_cols,
            tile_h,
            tile_w,
        )
        maps = [
            ("OSM GT", gt),
            (f"Xuannv {shot}-shot {head}\nF1 {x['f1_at_threshold']:.3f} AUC {x['auc_roc']:.3f}", x_map),
            (f"AEF {shot}-shot {head}\nF1 {a['f1_at_threshold']:.3f} AUC {a['auc_roc']:.3f}", a_map),
        ]
        resized = [(label, resize_map(canvas)) for label, canvas in maps]
        gap = 24
        row_title_w = 140
        label_h = 58
        row_h = label_h + max(img.height for _, img in resized)
        row_w = row_title_w + sum(img.width for _, img in resized) + gap * (len(resized) - 1)
        row = Image.new("RGB", (row_w, row_h), "white")
        draw = ImageDraw.Draw(row)
        draw.text((12, label_h + 92), f"{shot}-shot", fill=(0, 0, 0), font=label_font)
        x0 = row_title_w
        for label, img in resized:
            draw.multiline_text((x0 + 8, 6), label, fill=(0, 0, 0), font=small_font, spacing=3)
            row.paste(img, (x0, label_h))
            x0 += img.width + gap
        row_images.append(row)

    header_h = 82
    width = max(row.width for row in row_images)
    height = header_h + sum(row.height for row in row_images) + 24 * (len(row_images) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((18, 22), TASK_TITLES.get(task, task), fill=(0, 0, 0), font=title_font)
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
    figures = {task: build_task_figure(args, summary, task) for task in args.tasks}
    metrics = paired_metrics(summary, args.tasks, args.shots)
    metrics.to_csv(args.output_root / "nonbuilding_fewshot_paired_metrics.csv", index=False)
    lines = [
        "# Non-building Few-shot Semantic Mapping",
        "",
        "This diagnostic avoids building-related showcase tasks and focuses on semantic OSM classes.",
        "For each task and shot, the displayed head is the Xuannv head with the best F1 among `mlp` and `mlp_deep`; AEF uses the same head for fairness.",
        "",
        md_table(metrics),
        "",
    ]
    for task, path in figures.items():
        rel = path.name
        lines += [f"## {TASK_TITLES.get(task, task)}", "", f"![{task}]({rel})", ""]
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
