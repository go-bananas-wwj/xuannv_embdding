#!/usr/bin/env python3
"""全域嵌入 PCA 假彩镶嵌图：把 320 个 patch 的 64 维嵌入按地理位置拼接，
PCA 降到 3 维映射 RGB，输出整个海淀区的嵌入可视化大图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--region", default="haidian")
    p.add_argument("--month", default="202604")
    p.add_argument("--ref-label-root", type=Path,
                   default=Path("/data/xuannv_embedding/processed/haidian/labels/osm_water"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-width", type=int, default=2400)
    args = p.parse_args()

    emb_root = args.embedding_root / args.region
    patch_ids = sorted(d.name for d in emb_root.iterdir() if d.is_dir())

    # 读地理边界，建立网格索引
    infos = {}
    for pid in patch_ids:
        ref = args.ref_label_root / "masks" / f"{pid}.tif"
        with rasterio.open(ref) as src:
            infos[pid] = src.bounds
    xs = sorted({round(b.left) for b in infos.values()})
    ys = sorted({round(b.top) for b in infos.values()}, reverse=True)
    col_of = {x: i for i, x in enumerate(xs)}
    row_of = {y: i for i, y in enumerate(ys)}

    H, W, P = len(ys), len(xs), 128
    canvas = np.zeros((H * P, W * P, 64), dtype=np.float32)
    valid = np.zeros((H * P, W * P), dtype=bool)
    for pid in patch_ids:
        t = torch.load(emb_root / pid / f"{args.month}_embedding_map.pt",
                       map_location="cpu", weights_only=True).float().numpy()
        b = infos[pid]
        r, c = row_of[round(b.top)], col_of[round(b.left)]
        canvas[r * P:(r + 1) * P, c * P:(c + 1) * P] = t.transpose(1, 2, 0)
        valid[r * P:(r + 1) * P, c * P:(c + 1) * P] = True

    # 全域 PCA（子采样拟合）
    flat = canvas[valid]
    rng = np.random.default_rng(0)
    sub = flat[rng.choice(len(flat), min(500_000, len(flat)), replace=False)]
    mean = sub.mean(0)
    _, _, vt = np.linalg.svd(sub - mean, full_matrices=False)
    proj = (flat - mean) @ vt[:3].T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    rgb_flat = np.clip((proj - lo) / (hi - lo), 0, 1)

    rgb = np.zeros((H * P, W * P, 3), dtype=np.float32)
    rgb[valid] = rgb_flat
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    if img.width > args.max_width:
        img = img.resize((args.max_width, int(img.height * args.max_width / img.width)),
                         Image.Resampling.LANCZOS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print(f"saved {args.output} size={img.size} grid={H}x{W}")


if __name__ == "__main__":
    main()
