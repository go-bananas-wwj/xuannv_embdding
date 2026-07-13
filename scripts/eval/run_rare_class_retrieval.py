#!/usr/bin/env python3
"""稀有类原型检索评测：单样例 → 全区找相似（18 类评测阶段 4）。

协议：对每个稀有类（wetland/landfill/stadium 等），取正像素最多的 1 个 patch 作
query，用其正像素平均嵌入作原型；对其余 319 个 patch 计算逐像素 cosine 相似度，
取 patch 内 top-0.5% 像素均值作 patch 分数；按分数排序，报告真值正 patch 的
命中率（P@5 / P@10 / 全体正 patch 的平均排名）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


def load_mask(label_root: Path, patch_id: str) -> np.ndarray:
    with rasterio.open(label_root / "masks" / f"{patch_id}.tif") as src:
        return src.read(1) > 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--region", default="haidian")
    p.add_argument("--month", default="202604")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    emb_root = args.embedding_root / args.region
    patch_ids = sorted(d.name for d in emb_root.iterdir() if d.is_dir())

    # 找正像素最多的 patch 作 query，其余正 patch 作真值
    pos_counts = {}
    for pid in patch_ids:
        mask_path = args.label_root / "masks" / f"{pid}.tif"
        if mask_path.exists():
            pos_counts[pid] = int(load_mask(args.label_root, pid).sum())
    positives = [pid for pid, c in pos_counts.items() if c > 0]
    if len(positives) < 2:
        raise SystemExit(f"正 patch 不足 2 个（{len(positives)}），无法做检索评测")
    query_id = max(positives, key=lambda x: pos_counts[x])

    def load_emb(pid: str) -> torch.Tensor:
        t = torch.load(emb_root / pid / f"{args.month}_embedding_map.pt",
                       map_location="cpu", weights_only=True).float()
        return F.normalize(t, dim=0)

    q_emb = load_emb(query_id)
    q_mask = torch.from_numpy(load_mask(args.label_root, query_id))
    proto = F.normalize(q_emb[:, q_mask].mean(dim=1), dim=0)  # (64,)

    scores = {}
    for pid in patch_ids:
        if pid == query_id:
            continue
        sim = (proto.view(-1, 1, 1) * load_emb(pid)).sum(dim=0)  # (128,128)
        k = max(1, int(sim.numel() * 0.005))
        scores[pid] = float(sim.flatten().topk(k).values.mean())

    ranked = sorted(scores, key=scores.get, reverse=True)
    gt = set(positives) - {query_id}
    ranks = {pid: i + 1 for i, pid in enumerate(ranked)}
    gt_ranks = sorted(ranks[pid] for pid in gt)
    result = {
        "task": args.label_root.name,
        "query_patch": query_id,
        "query_positive_pixels": pos_counts[query_id],
        "num_gt_positives": len(gt),
        "num_candidates": len(ranked),
        "precision_at_5": sum(1 for pid in ranked[:5] if pid in gt) / 5,
        "precision_at_10": sum(1 for pid in ranked[:10] if pid in gt) / 10,
        "gt_ranks": gt_ranks,
        "mean_gt_rank": float(np.mean(gt_ranks)),
        "top10": [{"patch": pid, "score": scores[pid], "is_gt": pid in gt} for pid in ranked[:10]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["task", "query_patch", "num_gt_positives",
          "precision_at_5", "precision_at_10", "mean_gt_rank"]}, indent=1))


if __name__ == "__main__":
    main()
