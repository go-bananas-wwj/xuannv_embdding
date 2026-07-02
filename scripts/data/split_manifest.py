#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly split a manifest JSON file.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest: list[dict[str, Any]] = json.load(f)
    if len(manifest) < 2:
        raise ValueError("manifest must contain at least two samples")

    rng = random.Random(args.seed)
    indices = list(range(len(manifest)))
    rng.shuffle(indices)
    val_count = max(1, round(len(indices) * args.val_ratio))
    val_indices = set(indices[:val_count])

    train_manifest = [item for idx, item in enumerate(manifest) if idx not in val_indices]
    val_manifest = [item for idx, item in enumerate(manifest) if idx in val_indices]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    with args.train_output.open("w", encoding="utf-8") as f:
        json.dump(train_manifest, f, ensure_ascii=False, indent=2)
    with args.val_output.open("w", encoding="utf-8") as f:
        json.dump(val_manifest, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "train_count": len(train_manifest),
                "val_count": len(val_manifest),
                "train_output": str(args.train_output),
                "val_output": str(args.val_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
