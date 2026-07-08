#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from downstreams.inference import (
    build_inference_loader,
    load_model_for_inference,
    precompute_embeddings,
    write_meta_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--suffix", default="")
    p.add_argument("--split", default="all")
    p.add_argument("--device", default=None, help="Override inference device, e.g. npu:0 or cpu.")
    p.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Split patches across this many export workers.",
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Current export worker id in [0, num_shards).",
    )
    p.add_argument(
        "--context-margin",
        type=int,
        default=None,
        help="Temporarily override data.context_margin for overlap/context export.",
    )
    p.add_argument(
        "--center-crop-size",
        type=int,
        default=None,
        help="Center-crop exported embedding maps to this spatial size, e.g. 128.",
    )
    p.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Only save selected YYYYMM embedding months. Defaults to all model months.",
    )
    init_group = p.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--checkpoint", type=Path, default=None)
    init_group.add_argument(
        "--random-init",
        action="store_true",
        help="随机初始化 backbone，生成 random-init 基线 embedding",
    )
    args = p.parse_args()

    if not args.config.exists():
        p.error(f"config 不存在: {args.config}")
    if args.checkpoint is not None and not args.checkpoint.exists():
        p.error(f"checkpoint 不存在: {args.checkpoint}")

    model, cfg, device = load_model_for_inference(
        args.config,
        args.checkpoint,
        random_init=args.random_init,
        device_preference=args.device,
    )

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    exp_name = cfg.experiment.name
    sha = args.checkpoint.stem[:8] if args.checkpoint else "random"
    init_tag = "_random_init" if args.random_init else ""
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_root = args.output_root / f"{date_str}_{exp_name}_{sha}{init_tag}{suffix}"

    for region in args.regions:
        logger.info("生成 %s embedding", region)
        loader = build_inference_loader(
            cfg,
            region,
            split=args.split,
            context_margin=args.context_margin,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        region_dir = out_root / region
        precompute_embeddings(
            model,
            loader,
            device,
            region_dir,
            months=args.months,
            center_crop_size=args.center_crop_size,
        )

    write_meta_json(out_root, args.checkpoint, args.config, " ".join(sys.argv))
    logger.info("embedding 保存至 %s", out_root)


if __name__ == "__main__":
    main()
