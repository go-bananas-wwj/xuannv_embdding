#!/usr/bin/env python3
"""Measure temporal embedding deltas on labeled change/stability regions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

_MONTH_SUFFIX_RE = re.compile(r"^(?P<patch>.+)_(?P<month>\d{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs=2, default=["202512", "202605"])
    parser.add_argument("--max-patches", type=int, default=0)
    return parser.parse_args()


def _valid_yyyymm(value: str) -> bool:
    year = int(value[:4])
    month = int(value[4:])
    return 1900 <= year <= 2100 and 1 <= month <= 12


def base_patch_id(patch_id: str) -> str:
    match = _MONTH_SUFFIX_RE.match(patch_id)
    if match is None or not _valid_yyyymm(match.group("month")):
        return patch_id
    return match.group("patch")


def resolve_source_region(region: str, patch_id: str) -> tuple[str, str]:
    for source_region in ("haidian", "harbin"):
        prefix = f"{source_region}_"
        if patch_id.startswith(prefix):
            return source_region, patch_id
    return region, patch_id


def resolve_embedding_dir(embedding_root: Path, region: str, patch_id: str) -> Path:
    patch_id = base_patch_id(patch_id)
    source_region, source_patch = resolve_source_region(region, patch_id)
    candidates = [
        embedding_root / region / patch_id,
        embedding_root / region / source_patch,
        embedding_root / region / f"{region}_{patch_id}",
        embedding_root / region / f"{source_region}_{source_patch}",
        embedding_root / source_region / source_patch,
        embedding_root / source_region / patch_id,
        embedding_root / source_region / f"{source_region}_{source_patch}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_mask_path(mask_dir: Path, patch_id: str) -> Path:
    exact = mask_dir / f"{patch_id}.tif"
    if exact.exists():
        return exact
    candidates = sorted(mask_dir.glob(f"{patch_id}_*.tif"))
    if not candidates:
        return exact

    def _month(path: Path) -> int:
        match = _MONTH_SUFFIX_RE.match(path.stem)
        if match is None or not _valid_yyyymm(match.group("month")):
            return -1
        return int(match.group("month"))

    return max(candidates, key=_month)


def iter_patch_ids(label_root: Path) -> list[str]:
    mask_dir = label_root / "masks"
    patch_ids = []
    for path in sorted(mask_dir.glob("*.tif")):
        patch_ids.append(base_patch_id(path.stem))
    return sorted(set(patch_ids))


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def patch_stats(
    embedding_root: Path,
    label_root: Path,
    region: str,
    patch_id: str,
    months: list[str],
) -> dict[str, Any] | None:
    emb_dir = resolve_embedding_dir(embedding_root, region, patch_id)
    before_path = emb_dir / f"{months[0]}_embedding_map.pt"
    after_path = emb_dir / f"{months[1]}_embedding_map.pt"
    mask_path = resolve_mask_path(label_root / "masks", patch_id)
    if not before_path.exists() or not after_path.exists() or not mask_path.exists():
        return None

    before = torch.load(before_path, map_location="cpu", weights_only=True).float()
    after = torch.load(after_path, map_location="cpu", weights_only=True).float()
    with rasterio.open(mask_path) as src:
        mask = torch.from_numpy(src.read(1)).bool()
    if before.shape[-2:] != mask.shape:
        return None

    l2 = (after - before).pow(2).sum(dim=0).sqrt()
    cosine_distance = 1.0 - (F.normalize(before, dim=0) * F.normalize(after, dim=0)).sum(dim=0)
    positive = mask
    negative = ~mask
    return {
        "patch_id": patch_id,
        "positive_pixels": int(positive.sum().item()),
        "negative_pixels": int(negative.sum().item()),
        "l2_positive": float(l2[positive].mean().item()) if positive.any() else None,
        "l2_negative": float(l2[negative].mean().item()) if negative.any() else None,
        "cos_positive": (
            float(cosine_distance[positive].mean().item()) if positive.any() else None
        ),
        "cos_negative": (
            float(cosine_distance[negative].mean().item()) if negative.any() else None
        ),
    }


def main() -> None:
    args = parse_args()
    patch_ids = iter_patch_ids(args.label_root)
    if args.max_patches > 0:
        patch_ids = patch_ids[: args.max_patches]

    records = []
    for patch_id in patch_ids:
        item = patch_stats(
            args.embedding_root,
            args.label_root,
            args.region,
            patch_id,
            args.months,
        )
        if item is not None:
            records.append(item)

    l2_pos = [item["l2_positive"] for item in records if item["l2_positive"] is not None]
    l2_neg = [item["l2_negative"] for item in records if item["l2_negative"] is not None]
    cos_pos = [item["cos_positive"] for item in records if item["cos_positive"] is not None]
    cos_neg = [item["cos_negative"] for item in records if item["cos_negative"] is not None]
    summary = {
        "embedding_root": str(args.embedding_root),
        "label_root": str(args.label_root),
        "region": args.region,
        "months": args.months,
        "num_patches": len(records),
        "l2_positive": summarize(l2_pos),
        "l2_negative": summarize(l2_neg),
        "l2_pos_neg_ratio": mean(l2_pos) / max(mean(l2_neg), 1e-8) if l2_pos and l2_neg else None,
        "cos_positive": summarize(cos_pos),
        "cos_negative": summarize(cos_neg),
        "cos_pos_neg_ratio": (
            mean(cos_pos) / max(mean(cos_neg), 1e-8) if cos_pos and cos_neg else None
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
