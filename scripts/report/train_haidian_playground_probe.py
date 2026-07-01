#!/usr/bin/env python3
"""Train a lightweight Haidian playground downstream head on P2A embeddings."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EMBEDDING_ROOT = Path("/data/xuannv_embedding/modelscope_upload/haidian/v1/api_ready/data/haidian/embeddings/v1")
DEFAULT_LABEL_DIR = Path("/data/xuannv_embedding/processed/haidian/labels/osm_playground/masks")
DEFAULT_S2_DIR = Path("/data/xuannv_embedding/processed/haidian/patches/s2")
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/experiments/haidian_playground_probe_20260701")


@dataclass(frozen=True)
class PatchItem:
    patch_id: str
    embedding_path: Path
    label_path: Path
    s2_path: Path | None
    positive_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="202605")
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-pos", type=int, default=140_000)
    parser.add_argument("--max-neg", type=int, default=220_000)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def patch_id_from_path(path: Path) -> str:
    return path.stem


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return (src.read(1) > 0).astype(np.uint8)


def read_embedding(path: Path) -> np.ndarray:
    emb = np.load(path).astype(np.float32, copy=False)
    if emb.ndim != 3:
        raise ValueError(f"expected CHW embedding, got {emb.shape}: {path}")
    return np.moveaxis(emb, 0, -1).reshape(-1, emb.shape[0])


def find_s2_patch(s2_dir: Path, patch_id: str) -> Path | None:
    matches = sorted(s2_dir.glob(f"*_{patch_id}.tif"))
    return matches[-1] if matches else None


def list_items(args: argparse.Namespace) -> list[PatchItem]:
    month_dir = args.embedding_root / args.month
    items: list[PatchItem] = []
    for emb_path in sorted(month_dir.glob("patch_*.npy")):
        patch_id = patch_id_from_path(emb_path)
        label_path = args.label_dir / f"{patch_id}.tif"
        if not label_path.exists():
            continue
        mask = read_mask(label_path)
        items.append(
            PatchItem(
                patch_id=patch_id,
                embedding_path=emb_path,
                label_path=label_path,
                s2_path=find_s2_patch(args.s2_dir, patch_id),
                positive_pixels=int(mask.sum()),
            )
        )
    if not items:
        raise RuntimeError(f"no matched embeddings and labels under {month_dir} / {args.label_dir}")
    return items


def split_items(items: list[PatchItem], seed: int) -> tuple[list[PatchItem], list[PatchItem]]:
    rng = np.random.default_rng(seed)
    pos = [item for item in items if item.positive_pixels > 0]
    neg = [item for item in items if item.positive_pixels == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    test_pos_n = max(1, int(round(len(pos) * 0.2)))
    test_neg_n = max(1, int(round(len(neg) * 0.2))) if neg else 0
    test_ids = {item.patch_id for item in pos[:test_pos_n] + neg[:test_neg_n]}
    train = [item for item in items if item.patch_id not in test_ids]
    test = [item for item in items if item.patch_id in test_ids]
    return train, test


def sample_pixels(items: list[PatchItem], max_pos: int, max_neg: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_chunks: list[np.ndarray] = []
    neg_chunks: list[np.ndarray] = []
    pos_budget = max(1, max_pos // max(1, sum(item.positive_pixels > 0 for item in items)))
    neg_budget = max(1, max_neg // max(1, len(items)))

    for item in items:
        emb = read_embedding(item.embedding_path)
        y = read_mask(item.label_path).reshape(-1)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx):
            take = min(len(pos_idx), pos_budget)
            pos_chunks.append(emb[rng.choice(pos_idx, size=take, replace=False)])
        if len(neg_idx):
            take = min(len(neg_idx), neg_budget)
            neg_chunks.append(emb[rng.choice(neg_idx, size=take, replace=False)])

    if not pos_chunks or not neg_chunks:
        raise RuntimeError("need both positive and negative pixels to train")
    x_pos = np.concatenate(pos_chunks, axis=0)
    x_neg = np.concatenate(neg_chunks, axis=0)
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones(len(x_pos), dtype=np.uint8), np.zeros(len(x_neg), dtype=np.uint8)])
    order = rng.permutation(len(y))
    return x[order], y[order]


def train_probe(x: np.ndarray, y: np.ndarray) -> Any:
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=20260701,
            n_jobs=-1,
        ),
    ).fit(x, y)


def predict_patch(model: Any, item: PatchItem) -> np.ndarray:
    emb = read_embedding(item.embedding_path)
    prob = model.predict_proba(emb)[:, 1]
    return prob.reshape(128, 128).astype(np.float32)


def evaluate(model: Any, items: list[PatchItem]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    probs_by_patch: dict[str, np.ndarray] = {}
    for item in items:
        y = read_mask(item.label_path).reshape(-1)
        p = predict_patch(model, item).reshape(-1)
        y_all.append(y)
        p_all.append(p)
        probs_by_patch[item.patch_id] = p.reshape(128, 128)
    y_true = np.concatenate(y_all)
    prob = np.concatenate(p_all)

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thresholds[max(0, best_idx - 1)]) if len(thresholds) else 0.5
    pred = (prob >= best_threshold).astype(np.uint8)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, prob)),
        "best_threshold": best_threshold,
        "best_f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "iou": float(jaccard_score(y_true, pred, zero_division=0)),
        "positive_pixel_ratio": float(y_true.mean()),
        "num_eval_pixels": int(len(y_true)),
    }
    return metrics, probs_by_patch


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    arr = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    arr[mask > 0] = (220, 0, 0)
    return Image.fromarray(arr)


def prob_to_rgb(prob: np.ndarray) -> Image.Image:
    p = np.clip(prob, 0.0, 1.0)
    arr = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
    arr[..., 0] = (255 * p).astype(np.uint8)
    arr[..., 1] = (255 * (1.0 - np.abs(p - 0.5) * 2)).astype(np.uint8)
    arr[..., 2] = (255 * (1.0 - p)).astype(np.uint8)
    return Image.fromarray(arr)


def s2_to_rgb(path: Path | None) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGB", (128, 128), (235, 235, 235))
    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        bands = [desc.index(name) + 1 for name in ("B04", "B03", "B02") if name in desc]
        if len(bands) != 3:
            bands = [1, 2, 3]
        arr = src.read(bands).astype(np.float32)
    rgb = np.moveaxis(arr, 0, -1)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [2, 98])
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = ImageFont.load_default()
    draw.rectangle([xy, (xy[0] + 74, xy[1] + 13)], fill=(255, 255, 255))
    draw.text((xy[0] + 2, xy[1] + 2), text, fill=(0, 0, 0), font=font)


def make_grid(images: list[Image.Image], columns: int, tile: int = 128, gap: int = 2) -> Image.Image:
    rows = int(math.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * tile + (columns - 1) * gap, rows * tile + (rows - 1) * gap), "white")
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        canvas.paste(image.resize((tile, tile)), (col * (tile + gap), row * (tile + gap)))
    return canvas


def save_full_domain(items: list[PatchItem], probs: dict[str, np.ndarray], threshold: float, output_dir: Path) -> None:
    gt_tiles = []
    pred_tiles = []
    prob_tiles = []
    for item in sorted(items, key=lambda x: x.patch_id):
        gt = read_mask(item.label_path)
        prob = probs[item.patch_id]
        gt_tiles.append(mask_to_rgb(gt))
        pred_tiles.append(mask_to_rgb((prob >= threshold).astype(np.uint8)))
        prob_tiles.append(prob_to_rgb(prob))
    make_grid(gt_tiles, columns=20).save(output_dir / "haidian_playground_osm_gt_full_domain.png")
    make_grid(pred_tiles, columns=20).save(output_dir / "haidian_playground_pred_full_domain.png")
    make_grid(prob_tiles, columns=20).save(output_dir / "haidian_playground_probability_full_domain.png")


def save_samples(items: list[PatchItem], probs: dict[str, np.ndarray], threshold: float, output_dir: Path) -> None:
    ranked = sorted(items, key=lambda item: item.positive_pixels, reverse=True)
    selected = ranked[:12]
    tile_w, tile_h = 128, 128
    gap = 8
    row_h = tile_h + 18
    canvas = Image.new("RGB", (4 * tile_w + 3 * gap, len(selected) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    for row, item in enumerate(selected):
        y0 = row * row_h + 16
        gt = read_mask(item.label_path)
        prob = probs[item.patch_id]
        panels = [
            ("RGB", s2_to_rgb(item.s2_path)),
            ("OSM GT", mask_to_rgb(gt)),
            ("Prob", prob_to_rgb(prob)),
            ("Pred", mask_to_rgb((prob >= threshold).astype(np.uint8))),
        ]
        for col, (title, image) in enumerate(panels):
            x0 = col * (tile_w + gap)
            canvas.paste(image.resize((tile_w, tile_h)), (x0, y0))
            draw_label(draw, (x0, y0 - 15), title)
        draw.text((2, y0 + tile_h - 12), item.patch_id, fill=(0, 0, 0), font=ImageFont.load_default())
    canvas.save(output_dir / "haidian_playground_probe_samples.png")


def copy_patch_masks(items: list[PatchItem], probs: dict[str, np.ndarray], threshold: float, output_dir: Path) -> None:
    pred_dir = output_dir / "pred_masks"
    prob_dir = output_dir / "probability_png"
    gt_dir = output_dir / "osm_gt_masks_png"
    for path in [pred_dir, prob_dir, gt_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for item in items:
        prob = probs[item.patch_id]
        mask_to_rgb((prob >= threshold).astype(np.uint8)).save(pred_dir / f"{item.patch_id}.png")
        prob_to_rgb(prob).save(prob_dir / f"{item.patch_id}.png")
        mask_to_rgb(read_mask(item.label_path)).save(gt_dir / f"{item.patch_id}.png")


def write_readme(output_dir: Path, metrics: dict[str, Any], args: argparse.Namespace, label_summary: dict[str, Any]) -> None:
    text = f"""# 海淀区操场/运动场识别下游头实验

本实验使用 P2A embedding 的 `{args.month}` 月数据，基于 OSM 弱标签训练一个像素级 linear probe 下游头，用于初步判断 embedding 对操场/运动场类目标的可分性。

## 标签口径

- OSM 来源：`haidian_activity.gpkg` 与 `haidian_landuse.gpkg`
- 选入标签：`leisure=pitch/track/stadium/sports_centre/playground`，以及 `landuse=recreation_ground`
- 标签性质：OSM 弱标签，不是人工精标 GT

## 结果

- ROC-AUC：{metrics['roc_auc']:.4f}
- AP：{metrics['average_precision']:.4f}
- 最优阈值：{metrics['best_threshold']:.4f}
- Best F1：{metrics['best_f1']:.4f}
- Precision：{metrics['precision']:.4f}
- Recall：{metrics['recall']:.4f}
- IoU：{metrics['iou']:.4f}
- 验证像素正样本比例：{metrics['positive_pixel_ratio']:.4f}

## 输出文件

- `playground_linear_probe.joblib`：下游 linear probe 权重
- `summary.json`：实验配置、划分、指标
- `haidian_playground_osm_gt_full_domain.png`：全域 OSM 弱标签
- `haidian_playground_pred_full_domain.png`：全域预测结果
- `haidian_playground_probability_full_domain.png`：全域概率图
- `haidian_playground_probe_samples.png`：典型 patch 可视化
- `pred_masks/`、`probability_png/`、`osm_gt_masks_png/`：逐 patch 结果

## 标签覆盖

- patch 总数：{label_summary['num_patches']}
- 有操场/运动场标签的 patch：{label_summary['positive_patches']}
- 平均正样本像素比例：{label_summary['positive_ratio_mean']:.4f}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def zip_outputs(output_dir: Path) -> None:
    archive_dir = output_dir / "archives"
    archive_dir.mkdir(exist_ok=True)
    zip_path = archive_dir / f"{output_dir.name}_summary.zip"
    if zip_path.exists():
        zip_path.unlink()
    include_files = [
        "README.md",
        "summary.json",
        "playground_linear_probe.joblib",
        "haidian_playground_osm_gt_full_domain.png",
        "haidian_playground_pred_full_domain.png",
        "haidian_playground_probability_full_domain.png",
        "haidian_playground_probe_samples.png",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name in include_files:
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def build_labels_if_needed() -> None:
    metadata = DEFAULT_LABEL_DIR.parent / "metadata.json"
    if metadata.exists():
        return
    subprocess.run(
        [
            "python",
            "/root/workspace/xuannv/scripts/data/build_osm_weak_semantic_labels.py",
            "--regions",
            "haidian",
            "--tasks",
            "osm_playground",
            "--overwrite",
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    build_labels_if_needed()
    output_dir = args.output_root / f"month_{args.month}"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list_items(args)
    train_items, test_items = split_items(items, args.seed)
    x_train, y_train = sample_pixels(train_items, args.max_pos, args.max_neg, args.seed)
    model = train_probe(x_train, y_train)
    metrics, test_probs = evaluate(model, test_items)
    _, all_probs = evaluate(model, items)

    model_path = output_dir / "playground_linear_probe.joblib"
    joblib.dump(model, model_path)
    save_full_domain(items, all_probs, metrics["best_threshold"], output_dir)
    save_samples(items, all_probs, metrics["best_threshold"], output_dir)
    copy_patch_masks(items, all_probs, metrics["best_threshold"], output_dir)

    label_meta = json.loads((args.label_dir.parent / "metadata.json").read_text(encoding="utf-8"))
    summary = {
        "task": "haidian_playground_osm_linear_probe",
        "month": args.month,
        "embedding_root": str(args.embedding_root / args.month),
        "label_dir": str(args.label_dir),
        "output_dir": str(output_dir),
        "train_patch_count": len(train_items),
        "test_patch_count": len(test_items),
        "train_positive_pixels_sampled": int(y_train.sum()),
        "train_negative_pixels_sampled": int((y_train == 0).sum()),
        "test_patch_ids": [item.patch_id for item in test_items],
        "metrics": metrics,
        "label_summary": label_meta["summary"],
        "label_rule": label_meta["rule"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, metrics, args, label_meta["summary"])
    zip_outputs(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
