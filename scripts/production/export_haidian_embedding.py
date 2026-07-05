#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/production/haidian_embedding_v1.yaml")
DEFAULT_CHECKPOINT = Path(
    "/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt"
)
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/embeddings/production")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Haidian production embeddings from haidian-embedding-v1."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--months",
        nargs="+",
        default=["202512", "202601", "202602", "202603", "202604", "202605"],
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--suffix", default="haidian_embedding_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "downstreams/scripts/precompute_embeddings.py",
        "--config",
        str(args.config),
        "--regions",
        "haidian",
        "--output-root",
        str(args.output_root),
        "--checkpoint",
        str(args.checkpoint),
        "--suffix",
        args.suffix,
        "--months",
        *args.months,
        "--device",
        args.device,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
