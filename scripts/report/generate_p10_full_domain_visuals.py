#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from downstreams.heads.linear_probe import LinearProbeHead
from downstreams.heads.segmentation_head import MLPProbeHead
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from torch import nn


TASKS = {
    "construction": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/construction",
        "title": "Construction",
    },
    "building": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
        "title": "Building",
    },
    "road": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
        "title": "Road",
    },
    "water": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
        "title": "Water",
    },
}


@dataclass(frozen=True)
class Spec:
    name: str
    embedding_root: Path
    benchmark_root: Path
    month: str | None = None


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


def parse_spec(raw: str) -> Spec:
    parts = raw.split("|")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("spec must be NAME|EMBEDDING_ROOT|BENCHMARK_ROOT or NAME|EMBEDDING_ROOT|BENCHMARK_ROOT|MONTH")
    month = parts[3] if len(parts) == 4 and parts[3] else None
    return Spec(parts[0], Path(parts[1]), Path(parts[2]), month)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P10 full-domain PCA and downstream maps.")
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--pca-sample-pixels-per-spec", type=int, default=220000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    parser.add_argument("--keep-probability-maps", action="store_true")
    parser.add_argument("--skip-downstream-maps", action="store_true")
    return parser.parse_args()


def stretch_rgb(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    out = np.zeros_like(arr, dtype=np.float32)
    if valid is None:
        valid = np.ones(arr.shape[:2], dtype=bool)
    for channel in range(arr.shape[-1]):
        band = arr[..., channel]
        vals = band[valid]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [2, 98])
        if hi > lo:
            out[..., channel] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
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
    if not mask_paths:
        raise FileNotFoundError(f"No patch masks under {label_root / 'masks'}")
    records: list[tuple[str, Path, float, float, int, int]] = []
    lefts: list[float] = []
    tops: list[float] = []
    width = height = 0
    for path in mask_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
        patch_id = path.stem
        records.append((patch_id, path, float(bounds.left), float(bounds.top), width, height))
        lefts.append(float(bounds.left))
        tops.append(float(bounds.top))
    unique_lefts = sorted({round(x, 3) for x in lefts})
    unique_tops = sorted({round(y, 3) for y in tops}, reverse=True)
    col_of = {value: idx for idx, value in enumerate(unique_lefts)}
    row_of = {value: idx for idx, value in enumerate(unique_tops)}
    layouts = [
        PatchLayout(
            patch_id=patch_id,
            row=row_of[round(top, 3)],
            col=col_of[round(left, 3)],
            mask_path=path,
        )
        for patch_id, path, left, top, _width, _height in records
    ]
    return layouts, len(unique_tops), len(unique_lefts), height, width


def empty_canvas(rows: int, cols: int, tile_h: int, tile_w: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.float32) * 0.92
    valid = np.zeros((rows * tile_h, cols * tile_w), dtype=bool)
    return canvas, valid


def paste_tile(
    canvas: np.ndarray,
    valid: np.ndarray,
    tile: np.ndarray,
    layout: PatchLayout,
    tile_h: int,
    tile_w: int,
) -> None:
    y0 = layout.row * tile_h
    x0 = layout.col * tile_w
    canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    valid[y0 : y0 + tile_h, x0 : x0 + tile_w] = True


def save_canvas(path: Path, canvas: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(to_uint8(canvas))
    if title:
        header_h = 70
        out = Image.new("RGB", (img.width, img.height + header_h), "white")
        draw = ImageDraw.Draw(out)
        draw.text((24, 22), title, fill=(0, 0, 0))
        out.paste(img, (0, header_h))
        img = out
    img.save(path)


def embedding_path(spec: Spec, region: str, patch_id: str, month: str) -> Path:
    spec_month = spec.month or month
    return spec.embedding_root / region / patch_id / f"{spec_month}_embedding_map.pt"


def load_embedding(spec: Spec, region: str, patch_id: str, month: str) -> torch.Tensor:
    path = embedding_path(spec, region, patch_id, month)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True).float()


def fit_common_pca(
    specs: list[Spec],
    layouts: list[PatchLayout],
    region: str,
    month: str,
    max_pixels_per_spec: int,
    seed: int,
) -> PCA:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    per_patch = max(64, max_pixels_per_spec // max(len(layouts), 1))
    for spec in specs:
        sampled = 0
        for layout in layouts:
            emb = load_embedding(spec, region, layout.patch_id, month).numpy()
            channels, height, width = emb.shape
            flat = emb.reshape(channels, height * width).T
            take = min(per_patch, flat.shape[0], max_pixels_per_spec - sampled)
            if take <= 0:
                break
            idx = rng.choice(flat.shape[0], size=take, replace=False)
            chunks.append(flat[idx])
            sampled += take
    data = np.concatenate(chunks, axis=0)
    return PCA(n_components=3, random_state=seed).fit(data)


def make_pca_canvas(
    spec: Spec,
    pca: PCA,
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    region: str,
    month: str,
) -> np.ndarray:
    canvas, valid = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        emb = load_embedding(spec, region, layout.patch_id, month).numpy()
        channels, height, width = emb.shape
        flat = emb.reshape(channels, height * width).T
        rgb = pca.transform(flat).reshape(height, width, 3)
        paste_tile(canvas, valid, rgb, layout, tile_h, tile_w)
    stretched = stretch_rgb(canvas, valid)
    stretched[~valid] = 0.92
    return stretched


def _build_probe_from_state(state: dict[str, torch.Tensor]) -> nn.Module:
    if "net.0.weight" in state and state["net.0.weight"].ndim == 4:
        return MLPProbeHead(embed_dim=state["net.0.weight"].shape[1], num_classes=state["net.2.weight"].shape[0])
    if "conv.weight" in state and state["conv.weight"].ndim == 4:
        return LinearProbeHead(embed_dim=state["conv.weight"].shape[1], num_classes=state["conv.weight"].shape[0])
    return PixelProbe(embed_dim=state["net.0.weight"].shape[1])


def load_probe(benchmark_root: Path, task: str, device: torch.device) -> tuple[nn.Module, float]:
    metrics_path = benchmark_root / task / "fold_0" / "metrics.json"
    ckpt_path = benchmark_root / task / "fold_0" / "checkpoints" / "best.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(metrics.get("val_threshold", metrics.get("threshold", 0.5)))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = _build_probe_from_state(state).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, threshold


def predict_probability(
    model: nn.Module,
    emb: torch.Tensor,
    device: torch.device,
    chunk_pixels: int,
) -> np.ndarray:
    channels, height, width = emb.shape
    if isinstance(model, (MLPProbeHead, LinearProbeHead)):
        with torch.no_grad():
            logits = model(emb.unsqueeze(0).to(device, non_blocking=True))[:, 1]
            return torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
    x = emb.permute(1, 2, 0).reshape(-1, channels).contiguous()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_pixels):
            xb = x[start : start + chunk_pixels].to(device, non_blocking=True)
            parts.append(torch.sigmoid(model(xb)).detach().cpu())
    return torch.cat(parts, dim=0).reshape(height, width).numpy().astype(np.float32)


def make_prediction_canvases(
    spec: Spec,
    task: str,
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
    region: str,
    month: str,
    device: torch.device,
    chunk_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, threshold = load_probe(spec.benchmark_root, task, device)
    pred_canvas, pred_valid = empty_canvas(rows, cols, tile_h, tile_w)
    prob_canvas, prob_valid = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        emb = load_embedding(spec, region, layout.patch_id, month)
        prob = predict_probability(model, emb, device, chunk_pixels)
        paste_tile(prob_canvas, prob_valid, red_probability(prob), layout, tile_h, tile_w)
        paste_tile(pred_canvas, pred_valid, red_binary(prob >= threshold), layout, tile_h, tile_w)
    return pred_canvas, prob_canvas


def make_gt_canvas(
    layouts: list[PatchLayout],
    rows: int,
    cols: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    canvas, valid = empty_canvas(rows, cols, tile_h, tile_w)
    for layout in layouts:
        with rasterio.open(layout.mask_path) as src:
            mask = src.read(1) > 0
        paste_tile(canvas, valid, red_binary(mask), layout, tile_h, tile_w)
    return canvas


def stack_with_labels(items: list[tuple[str, np.ndarray]], out_path: Path, title: str) -> None:
    label_w = 220
    header_h = 70
    images = [Image.fromarray(to_uint8(canvas)) for _, canvas in items]
    width = label_w + max(img.width for img in images)
    height = header_h + sum(img.height for img in images)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 22), title, fill=(0, 0, 0))
    y = header_h
    for label, img in zip((label for label, _ in items), images):
        draw.text((24, y + 24), label, fill=(0, 0, 0))
        out.paste(img, (label_w, y))
        y += img.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main() -> None:
    args = parse_args()
    specs: list[Spec] = args.spec
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)

    layouts, rows, cols, tile_h, tile_w = load_layout(Path(TASKS["building"]["label_root"]))
    pca = fit_common_pca(
        specs,
        layouts,
        args.region,
        args.month,
        args.pca_sample_pixels_per_spec,
        args.seed,
    )
    metadata: dict[str, Any] = {
        "month": args.month,
        "region": args.region,
        "rows": rows,
        "cols": cols,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "specs": [spec.__dict__ | {"embedding_root": str(spec.embedding_root), "benchmark_root": str(spec.benchmark_root)} for spec in specs],
        "outputs": [],
    }

    pca_items: list[tuple[str, np.ndarray]] = []
    for spec in specs:
        pca_canvas = make_pca_canvas(
            spec, pca, layouts, rows, cols, tile_h, tile_w, args.region, args.month
        )
        pca_items.append((spec.name, pca_canvas))
        out_path = output_root / f"{spec.name}_embedding_pca_{args.month}_geo.png"
        save_canvas(out_path, pca_canvas, f"{spec.name} embedding PCA {args.month}")
        metadata["outputs"].append(str(out_path))
    pca_compare = output_root / f"p10_embedding_pca_{args.month}_compare.png"
    stack_with_labels(pca_items, pca_compare, f"P10 embedding PCA common projection {args.month}")
    metadata["outputs"].append(str(pca_compare))

    if args.skip_downstream_maps:
        (output_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    for task in args.tasks:
        task_root = output_root / task
        label_root = Path(TASKS[task]["label_root"])
        task_layouts, task_rows, task_cols, task_tile_h, task_tile_w = load_layout(label_root)
        gt_canvas = make_gt_canvas(task_layouts, task_rows, task_cols, task_tile_h, task_tile_w)
        gt_path = task_root / f"{task}_gt_geo.png"
        save_canvas(gt_path, gt_canvas, f"{TASKS[task]['title']} GT")
        metadata["outputs"].append(str(gt_path))

        pred_items: list[tuple[str, np.ndarray]] = [("GT", gt_canvas)]
        prob_items: list[tuple[str, np.ndarray]] = [("GT", gt_canvas)]
        for spec in specs:
            pred_canvas, prob_canvas = make_prediction_canvases(
                spec,
                task,
                task_layouts,
                task_rows,
                task_cols,
                task_tile_h,
                task_tile_w,
                args.region,
                args.month,
                device,
                args.chunk_pixels,
            )
            pred_items.append((spec.name, pred_canvas))
            prob_items.append((spec.name, prob_canvas))
            pred_path = task_root / f"{spec.name}_{task}_prediction_geo.png"
            prob_path = task_root / f"{spec.name}_{task}_probability_geo.png"
            save_canvas(pred_path, pred_canvas, f"{spec.name} {TASKS[task]['title']} prediction")
            if args.keep_probability_maps:
                save_canvas(prob_path, prob_canvas, f"{spec.name} {TASKS[task]['title']} probability")
                metadata["outputs"].append(str(prob_path))
            metadata["outputs"].append(str(pred_path))
        pred_compare = task_root / f"{task}_gt_p10a_p10b_p10c_prediction_compare.png"
        stack_with_labels(pred_items, pred_compare, f"{TASKS[task]['title']} prediction comparison")
        metadata["outputs"].append(str(pred_compare))
        if args.keep_probability_maps:
            prob_compare = task_root / f"{task}_gt_p10a_p10b_p10c_probability_compare.png"
            stack_with_labels(prob_items, prob_compare, f"{TASKS[task]['title']} probability comparison")
            metadata["outputs"].append(str(prob_compare))

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
