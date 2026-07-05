#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple binary label roots with pixel-wise OR.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dirs = [root / "masks" for root in args.inputs]
    for mask_dir in mask_dirs:
        if not mask_dir.exists():
            raise FileNotFoundError(mask_dir)
    first_masks = sorted(mask_dirs[0].glob("patch_*.tif"))
    if not first_masks:
        raise RuntimeError(f"No masks found in {mask_dirs[0]}")
    out_masks = args.output / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for first_path in first_masks:
        merged: np.ndarray | None = None
        profile = None
        for mask_dir in mask_dirs:
            path = mask_dir / first_path.name
            if not path.exists():
                raise FileNotFoundError(path)
            with rasterio.open(path) as src:
                arr = src.read(1)
                profile = src.profile.copy()
            binary = arr > 0
            merged = binary if merged is None else (merged | binary)
        assert merged is not None and profile is not None
        profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
        out_path = out_masks / first_path.name
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(merged.astype(np.uint8), 1)
        counts[first_path.stem] = int(merged.sum())

    split_src = args.inputs[0] / "split_5fold.json"
    if split_src.exists():
        shutil.copy2(split_src, args.output / "split_5fold.json")
    manifest = {
        "name": args.name,
        "operation": "pixel_or",
        "inputs": [str(path) for path in args.inputs],
        "output": str(args.output),
        "patch_count": len(first_masks),
        "positive_patch_count": sum(value > 0 for value in counts.values()),
        "positive_pixel_count": sum(counts.values()),
    }
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
