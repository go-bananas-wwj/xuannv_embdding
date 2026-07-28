#!/usr/bin/env python3
"""Export the lock-pinned Harbin AEF annual-2025 maps without zero-filled nodata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from scripts.eval.export_aef_v5_embeddings import read_complete_aef_patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-inventory",
        type=Path,
        default=REPO_ROOT / "configs/eval/harbin_aef_annual_2025_coverage_inventory_20260728.json",
    )
    parser.add_argument(
        "--cog-lock",
        type=Path,
        default=REPO_ROOT / "configs/eval/harbin_aef_annual_2025_cog_lock_20260728.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/xuannv_embedding/cache/aef_official_2025_cogs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual_harbin_20260728"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.coverage_inventory.read_text(encoding="utf-8"))
    lock = json.loads(args.cog_lock.read_text(encoding="utf-8"))
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != 380:
        raise ValueError("Harbin AEF export requires the exact 380-patch coverage inventory")
    locked = {item["filename"]: item["sha256"] for item in lock.get("assets", [])}
    if len(locked) != 2:
        raise ValueError("Harbin AEF COG lock must contain exactly two assets")
    for filename, checksum in locked.items():
        path = args.cache_dir / filename
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"cached COG does not match the Harbin source lock: {filename}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent)
    )
    try:
        for index, record in enumerate(records, start=1):
            patch_id = str(record["patch_id"])
            reference = Path(record["reference_grid"]["path"])
            source = args.cache_dir / Path(str(record["aef_2025_uri"])).name
            embedding = read_complete_aef_patch(source, reference)
            if tuple(embedding.shape) != (64, 128, 128) or not bool(
                torch.isfinite(embedding).all()
            ):
                raise ValueError(f"invalid complete AEF embedding for {patch_id}")
            output = staging / "harbin" / patch_id
            output.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, output / "annual_2025_embedding_map.pt")
            print(f"[{index}/380] {patch_id}", flush=True)
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "protocol_id": "harbin_aef_annual_2025_locked",
                    "region": "harbin",
                    "output_identifier": "annual_2025",
                    "coverage_inventory_sha256": sha256_file(args.coverage_inventory),
                    "cog_lock_sha256": sha256_file(args.cog_lock),
                    "patch_count": 380,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite existing export: {args.output_root}")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
