#!/usr/bin/env python3
"""Evaluate binary-mask post-processing for Haidian downstream tasks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage


PATCH_RE = re.compile(r"(patch_\d{6})")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    zh_name: str
    probability_dir: Path
    label_root: Path
    threshold: float
    min_areas: tuple[int, ...]
    full_probability_dir: Path | None = None
    full_layout_root: Path | None = None


@dataclass(frozen=True)
class PatchLayout:
    patch_id: str
    row: int
    col: int
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/experiments/production/"
            "haidian_v1_postprocess_false_positive_20260710"
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=["building", "road", "water", "construction"])
    return parser.parse_args()


def default_task_specs() -> dict[str, TaskSpec]:
    core_root = Path(
        "/data/xuannv_embedding/experiments/production/"
        "haidian_v1_p10c_epoch800_pixel_probe_full_domain_20260710"
    )
    production_root = Path(
        "/data/xuannv_embedding/experiments/production/"
        "haidian_v1_landuse_construction_20260710"
    )
    return {
        "building": TaskSpec(
            name="building",
            zh_name="建筑物",
            probability_dir=core_root / "building" / "probabilities",
            label_root=Path("/data/xuannv_embedding/processed/haidian/labels/building_osm"),
            threshold=0.5611526370048523,
            min_areas=(4, 8, 16, 32, 64, 128),
        ),
        "road": TaskSpec(
            name="road",
            zh_name="道路",
            probability_dir=core_root / "road" / "probabilities",
            label_root=Path("/data/xuannv_embedding/processed/haidian/labels/road_osm"),
            threshold=0.6360677480697632,
            min_areas=(4, 8, 16, 32, 64, 128),
        ),
        "water": TaskSpec(
            name="water",
            zh_name="水体",
            probability_dir=core_root / "water" / "probabilities",
            label_root=Path("/data/xuannv_embedding/processed/haidian/labels/osm_water"),
            threshold=0.659624457359314,
            min_areas=(8, 16, 32, 64, 128, 256),
        ),
        "construction": TaskSpec(
            name="construction",
            zh_name="施工工地",
            probability_dir=production_root / "construction_conv3x3_full_export" / "probabilities",
            label_root=Path("/data/xuannv_embedding/processed/haidian/labels/construction"),
            threshold=0.7708274126052856,
            min_areas=(8, 16, 32, 64, 128, 256, 512),
            full_probability_dir=production_root / "construction_conv3x3_320patch_export" / "probabilities",
            full_layout_root=production_root / "all320_layout_masks",
        ),
    }


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


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def save_binary_raster(reference_path: Path, output_path: Path, mask: np.ndarray) -> None:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def remove_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int, int]:
    labels, num = ndimage.label(mask > 0)
    if num == 0:
        return mask.astype(np.uint8), 0, 0
    sizes = np.bincount(labels.reshape(-1))
    keep = sizes >= min_area
    keep[0] = False
    cleaned = keep[labels]
    removed_count = int((~keep[1:]).sum())
    removed_pixels = int(((mask > 0) & (~cleaned)).sum())
    return cleaned.astype(np.uint8), removed_count, removed_pixels


def apply_method(mask: np.ndarray, method: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    if method["kind"] == "raw":
        return mask.astype(np.uint8), {"removed_components": 0, "removed_pixels": 0}
    cleaned, removed_count, removed_pixels = remove_small_components(mask, int(method["min_area"]))
    return cleaned, {"removed_components": removed_count, "removed_pixels": removed_pixels}


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    valid = target >= 0
    pred_bool = (pred[valid] > 0)
    target_bool = (target[valid] == 1)
    tp = int((pred_bool & target_bool).sum())
    fp = int((pred_bool & (~target_bool)).sum())
    fn = int(((~pred_bool) & target_bool).sum())
    tn = int(((~pred_bool) & (~target_bool)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def merge_counts(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    tn = int(sum(int(row["tn"]) for row in rows))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def red_binary(mask: np.ndarray) -> np.ndarray:
    rgb = np.ones((*mask.shape, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = (235, 20, 25)
    return rgb


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


def save_canvas(source_dir: Path, layout_root: Path, output_path: Path, title: str, suffix: str) -> None:
    layouts, rows, cols, tile_h, tile_w = load_layout(layout_root / "masks")
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 238
    for layout in layouts:
        path = source_dir / f"{layout.patch_id}_{suffix}.tif"
        if not path.exists():
            continue
        arr = load_raster(path)
        y0 = layout.row * tile_h
        x0 = layout.col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = red_binary(arr)
    image = Image.fromarray(canvas)
    header = 76
    out = Image.new("RGB", (image.width, image.height + header), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 24), title, fill=(0, 0, 0))
    out.paste(image, (0, header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def save_sample_comparison(
    task: TaskSpec,
    raw_dir: Path,
    best_dir: Path,
    output_path: Path,
    max_samples: int = 6,
) -> None:
    mask_dir = task.label_root / "masks"
    candidates: list[tuple[int, str]] = []
    for prob_path in sorted(task.probability_dir.glob("*_prob.tif")):
        patch_id = patch_id_from_path(prob_path)
        mask_path = resolve_mask(mask_dir, patch_id)
        if not mask_path.exists():
            continue
        raw = load_raster(raw_dir / f"{patch_id}_pred.tif")
        best = load_raster(best_dir / f"{patch_id}_pred.tif")
        gt = load_raster(mask_path)
        fp_reduced = int(((raw == 1) & (gt != 1)).sum() - ((best == 1) & (gt != 1)).sum())
        positives = int(gt.sum())
        candidates.append((fp_reduced + positives // 10, patch_id))
    selected = [patch_id for _score, patch_id in sorted(candidates, reverse=True)[:max_samples]]
    if not selected:
        return
    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    for row_idx, patch_id in enumerate(selected):
        gt = load_raster(resolve_mask(mask_dir, patch_id))
        raw = load_raster(raw_dir / f"{patch_id}_pred.tif")
        best = load_raster(best_dir / f"{patch_id}_pred.tif")
        for col_idx, (title, arr) in enumerate(
            [("GT", gt), ("Raw", raw), ("Post-processed", best)]
        ):
            axes[row_idx, col_idx].imshow(red_binary(arr))
            axes[row_idx, col_idx].set_title(f"{patch_id} {title}", fontsize=10)
            axes[row_idx, col_idx].axis("off")
    fig.suptitle(f"{task.name}: sample patch comparison", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_predictions_for_method(
    spec: TaskSpec,
    method: dict[str, Any],
    source_probability_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_components = 0
    removed_pixels = 0
    for prob_path in sorted(source_probability_dir.glob("*_prob.tif")):
        patch_id = patch_id_from_path(prob_path)
        prob = load_raster(prob_path)
        raw = (prob >= spec.threshold).astype(np.uint8)
        pred, stats = apply_method(raw, method)
        removed_components += int(stats["removed_components"])
        removed_pixels += int(stats["removed_pixels"])
        save_binary_raster(prob_path, output_dir / f"{patch_id}_pred.tif", pred)
    return {"removed_components": removed_components, "removed_pixels": removed_pixels}


def evaluate_method(spec: TaskSpec, pred_dir: Path) -> dict[str, float | int]:
    rows: list[dict[str, float | int]] = []
    for mask_path in sorted((spec.label_root / "masks").glob("*.tif")):
        patch_id = patch_id_from_path(mask_path)
        pred_path = pred_dir / f"{patch_id}_pred.tif"
        if not pred_path.exists():
            continue
        rows.append(compute_binary_metrics(load_raster(pred_path), load_raster(mask_path)))
    if not rows:
        raise RuntimeError(f"No overlapping predictions and masks for {spec.name}")
    return merge_counts(rows) | {"evaluated_patches": len(rows)}


def method_name(method: dict[str, Any]) -> str:
    if method["kind"] == "raw":
        return "raw"
    return f"remove_small_area_{method['min_area']}"


def evaluate_task(spec: TaskSpec, output_root: Path) -> dict[str, Any]:
    task_root = output_root / spec.name
    methods = [{"kind": "raw"}] + [{"kind": "remove_small", "min_area": n} for n in spec.min_areas]
    method_summaries: list[dict[str, Any]] = []
    for method in methods:
        name = method_name(method)
        pred_dir = task_root / "predictions" / name
        stats = write_predictions_for_method(spec, method, spec.probability_dir, pred_dir)
        metrics = evaluate_method(spec, pred_dir)
        method_summaries.append({"method": name, "params": method, "stats": stats, "metrics": metrics})

    best = max(method_summaries, key=lambda row: (float(row["metrics"]["f1"]), float(row["metrics"]["precision"])))
    raw = next(row for row in method_summaries if row["method"] == "raw")
    best_name = str(best["method"])
    raw_dir = task_root / "predictions" / "raw"
    best_dir = task_root / "predictions" / best_name
    vis_dir = task_root / "visualizations"
    save_canvas(
        raw_dir,
        spec.label_root,
        vis_dir / f"{spec.name}_raw_eval_domain.png",
        f"{spec.name} raw prediction, threshold={spec.threshold:.3f}",
        "pred",
    )
    save_canvas(
        best_dir,
        spec.label_root,
        vis_dir / f"{spec.name}_postprocessed_eval_domain.png",
        f"{spec.name} post-processed prediction ({best_name})",
        "pred",
    )
    save_canvas(
        spec.label_root / "masks",
        spec.label_root,
        vis_dir / f"{spec.name}_gt_eval_domain.png",
        f"{spec.name} GT / weak label domain",
        "",
    )
    save_sample_comparison(spec, raw_dir, best_dir, vis_dir / f"{spec.name}_sample_patch_comparison.png")

    full_prob_dir = spec.full_probability_dir or spec.probability_dir
    full_layout_root = spec.full_layout_root or spec.label_root
    full_raw_dir = task_root / "predictions_full_domain" / "raw"
    full_best_dir = task_root / "predictions_full_domain" / best_name
    write_predictions_for_method(spec, {"kind": "raw"}, full_prob_dir, full_raw_dir)
    write_predictions_for_method(spec, dict(best["params"]), full_prob_dir, full_best_dir)
    save_canvas(
        full_raw_dir,
        full_layout_root,
        vis_dir / f"{spec.name}_raw_full_320patch_geo.png",
        f"{spec.name} raw prediction, 320 patches",
        "pred",
    )
    save_canvas(
        full_best_dir,
        full_layout_root,
        vis_dir / f"{spec.name}_postprocessed_full_320patch_geo.png",
        f"{spec.name} post-processed prediction, 320 patches ({best_name})",
        "pred",
    )

    summary = {
        "task": spec.name,
        "zh_name": spec.zh_name,
        "threshold": spec.threshold,
        "raw": raw,
        "best": best,
        "methods": method_summaries,
        "visualizations": {
            "raw_eval_domain": str(vis_dir / f"{spec.name}_raw_eval_domain.png"),
            "postprocessed_eval_domain": str(vis_dir / f"{spec.name}_postprocessed_eval_domain.png"),
            "gt_eval_domain": str(vis_dir / f"{spec.name}_gt_eval_domain.png"),
            "sample_patch_comparison": str(vis_dir / f"{spec.name}_sample_patch_comparison.png"),
            "raw_full_320patch_geo": str(vis_dir / f"{spec.name}_raw_full_320patch_geo.png"),
            "postprocessed_full_320patch_geo": str(
                vis_dir / f"{spec.name}_postprocessed_full_320patch_geo.png"
            ),
        },
    }
    (task_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# 海淀区二值任务假正例后处理评估",
        "",
        "本次评估针对阈值分割后出现的离散小斑点，统一测试连通域面积过滤。"
        "该方法不会改变模型概率，只在二值化结果上删除面积过小的孤立连通域。",
        "",
        "## 指标汇总",
        "",
        "| 任务 | 原始 F1 | 后处理 F1 | 原始 Precision | 后处理 Precision | 原始 Recall | 后处理 Recall | 最优方法 | FP 变化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for summary in summaries:
        raw_metrics = summary["raw"]["metrics"]
        best_metrics = summary["best"]["metrics"]
        fp_delta = int(best_metrics["fp"]) - int(raw_metrics["fp"])
        lines.append(
            "| {name} | {raw_f1} | {best_f1} | {raw_p} | {best_p} | {raw_r} | {best_r} | {method} | {fp_delta} |".format(
                name=f"{summary['zh_name']} / {summary['task']}",
                raw_f1=format_metric(raw_metrics["f1"]),
                best_f1=format_metric(best_metrics["f1"]),
                raw_p=format_metric(raw_metrics["precision"]),
                best_p=format_metric(best_metrics["precision"]),
                raw_r=format_metric(raw_metrics["recall"]),
                best_r=format_metric(best_metrics["recall"]),
                method=summary["best"]["method"],
                fp_delta=fp_delta,
            )
        )
    lines.extend(["", "## 可视化", ""])
    for summary in summaries:
        vis = summary["visualizations"]
        lines.extend(
            [
                f"### {summary['zh_name']} / {summary['task']}",
                "",
                "原始 320 patch 全域预测：",
                f"![{summary['task']} raw full]({vis['raw_full_320patch_geo']})",
                "",
                "后处理 320 patch 全域预测：",
                f"![{summary['task']} post full]({vis['postprocessed_full_320patch_geo']})",
                "",
                "有标签区域局部对比（GT / 原始预测 / 后处理预测）：",
                f"![{summary['task']} samples]({vis['sample_patch_comparison']})",
                "",
            ]
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = default_task_specs()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for task_name in args.tasks:
        if task_name not in specs:
            raise KeyError(f"Unknown task {task_name}; available={sorted(specs)}")
        print(f"evaluating {task_name}")
        summaries.append(evaluate_task(specs[task_name], args.output_root))
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_root, summaries)
    print(args.output_root / "summary.md")


if __name__ == "__main__":
    main()
