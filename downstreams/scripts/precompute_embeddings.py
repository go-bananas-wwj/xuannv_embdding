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
from xuannv_embedding.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--suffix", default="")
    p.add_argument("--split", default="all")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--random-init",
        action="store_true",
        help="随机初始化 backbone，生成 random-init 基线 embedding",
    )
    p.add_argument(
        "--use-base-embedding",
        action="store_true",
        help="保存 base_embedding_map（蒸馏监督点）作为下游 embedding_map，而不是 vMF 瓶颈后的输出",
    )
    args = p.parse_args()

    if not args.config.exists():
        p.error(f"config 不存在: {args.config}")

    if not args.random_init:
        if args.checkpoint is None:
            cfg_tmp = Config.from_yaml(args.config)
            default_ckpt = (
                Path("/data/xuannv_embedding/outputs") / cfg_tmp.experiment.name / "best.pt"
            )
            if default_ckpt.exists():
                args.checkpoint = default_ckpt
                logger.info("未指定 checkpoint，使用默认路径: %s", args.checkpoint)
            else:
                p.error(
                    f"未指定 checkpoint 且默认路径不存在: {default_ckpt}"
                )
        if not args.checkpoint.exists():
            p.error(f"checkpoint 不存在: {args.checkpoint}")

    model, cfg, device = load_model_for_inference(
        args.config, args.checkpoint, random_init=args.random_init
    )

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    exp_name = cfg.experiment.name
    sha = args.checkpoint.stem[:8] if args.checkpoint else "random"
    init_tag = "_random_init" if args.random_init else ""
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_root = args.output_root / f"{date_str}_{exp_name}_{sha}{init_tag}{suffix}"

    for region in args.regions:
        logger.info("生成 %s embedding", region)
        loader = build_inference_loader(cfg, region, split=args.split)
        region_dir = out_root / region
        precompute_embeddings(model, loader, device, region_dir, use_base_embedding=args.use_base_embedding)

    write_meta_json(out_root, args.checkpoint, args.config, " ".join(sys.argv))
    logger.info("embedding 保存至 %s", out_root)


if __name__ == "__main__":
    main()
