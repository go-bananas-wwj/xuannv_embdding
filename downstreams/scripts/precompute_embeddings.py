#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from downstreams.inference import (
    build_inference_loader,
    load_model_for_inference,
    manifest_provenance,
    precompute_embeddings,
    write_meta_json,
)

from scripts.export_paths import resolve_export_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--export-name",
        default=None,
        help="Use this exact child directory name under --output-root for one sealed export.",
    )
    p.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Override the config manifest for auditable full-domain export.",
    )
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
    p.add_argument(
        "--skip-meta",
        action="store_true",
        help=(
            "Do not write root-level meta.json. Intended only for a coordinated sharded export, "
            "whose launcher writes metadata once after every shard has succeeded."
        ),
    )
    init_group = p.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--checkpoint", type=Path, default=None)
    init_group.add_argument(
        "--random-init",
        action="store_true",
        help="随机初始化 backbone，生成 random-init 基线 embedding",
    )
    args = p.parse_args()

    if args.skip_meta and (
        args.num_shards is None
        or args.shard_id is None
        or args.num_shards < 2
        or not 0 <= args.shard_id < args.num_shards
    ):
        p.error("--skip-meta requires --num-shards >= 2 and a valid --shard-id")
    if not args.config.exists():
        p.error(f"config 不存在: {args.config}")
    if args.checkpoint is not None and not args.checkpoint.exists():
        p.error(f"checkpoint 不存在: {args.checkpoint}")
    if args.manifest_path is not None and not args.manifest_path.is_file():
        p.error(f"manifest 不存在: {args.manifest_path}")
    if args.manifest_path is not None and len(args.regions) != 1:
        p.error("--manifest-path requires exactly one --regions value")
    if args.manifest_path is not None:
        from downstreams.inference import validate_manifest_region

        try:
            validate_manifest_region(args.manifest_path, args.regions[0])
        except ValueError as exc:
            p.error(str(exc))
        try:
            manifest_provenance(args.manifest_path)
        except ValueError as exc:
            p.error(str(exc))
    if (args.shard_id is None) != (args.num_shards is None):
        p.error("--shard-id and --num-shards must be provided together")
    if args.export_name is not None:
        try:
            resolve_export_root(args.output_root, args.export_name, "default")
        except ValueError as exc:
            p.error(str(exc))

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
    default_name = f"{date_str}_{exp_name}_{sha}{init_tag}{suffix}"
    try:
        out_root = resolve_export_root(args.output_root, args.export_name, default_name)
    except ValueError as exc:
        p.error(str(exc))

    for region in args.regions:
        logger.info("生成 %s embedding", region)
        loader = build_inference_loader(
            cfg,
            region,
            split=args.split,
            context_margin=args.context_margin,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            manifest_path=args.manifest_path,
        )
        region_dir = out_root / region
        produced_patch_ids = precompute_embeddings(
            model,
            loader,
            device,
            region_dir,
            months=args.months,
            center_crop_size=args.center_crop_size,
        )
        shard_name = (
            f"produced_patch_ids_shard_{args.shard_id}.json"
            if args.shard_id is not None
            else "produced_patch_ids.json"
        )
        (region_dir / shard_name).write_text(
            json.dumps(sorted(produced_patch_ids), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not args.skip_meta:
        write_meta_json(
            out_root,
            args.checkpoint,
            args.config,
            " ".join(sys.argv),
            manifest_path=args.manifest_path,
        )
    logger.info("embedding 保存至 %s", out_root)


if __name__ == "__main__":
    main()
