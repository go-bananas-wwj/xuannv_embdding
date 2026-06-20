#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from downstreams.data.split import create_stratified_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    split = create_stratified_folds(
        mask_dir=args.mask_dir,
        n_folds=args.n_folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    logger.info("split 已保存到 %s", args.out_path)


if __name__ == "__main__":
    main()
