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
from typing import Any

import torch

from scripts.eval.export_aef_v5_embeddings import read_complete_aef_mosaic

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_output_index(root: Path, patch_ids: list[str]) -> dict[str, Any]:
    """Read and hash exactly the sealed annual maps; reject partial outputs."""
    expected = set(patch_ids)
    observed = {
        path.parent.name
        for path in (root / "harbin").glob("*/annual_2025_embedding_map.pt")
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "AEF annual map set differs from coverage inventory: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    files: list[dict[str, Any]] = []
    for patch_id in sorted(patch_ids):
        path = root / "harbin" / patch_id / "annual_2025_embedding_map.pt"
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(embedding.shape) != (64, 128, 128) or not bool(torch.isfinite(embedding).all()):
            raise ValueError(f"invalid complete AEF embedding for {patch_id}")
        files.append(
            {
                "patch_id": patch_id,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "shape": [64, 128, 128],
            }
        )
    return {
        "schema_version": 1,
        "region": "harbin",
        "month": "annual_2025",
        "patch_count": len(files),
        "files": files,
    }


def seal_export(
    root: Path, coverage_inventory: Path, cog_lock: Path, patch_ids: list[str]
) -> dict[str, Any]:
    """Write a content-addressed index and metadata only after full verification."""
    index = build_output_index(root, patch_ids)
    index_path = root / "embedding_file_index.json"
    temporary_index = index_path.with_suffix(".json.tmp")
    temporary_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_index, index_path)
    meta = {
        "protocol_id": "harbin_aef_annual_2025_locked",
        "region": "harbin",
        "output_identifier": "annual_2025",
        "coverage_inventory": {
            "path": str(coverage_inventory.resolve()),
            "sha256": sha256_file(coverage_inventory),
        },
        "cog_lock": {"path": str(cog_lock.resolve()), "sha256": sha256_file(cog_lock)},
        "embedding_file_index": {
            "path": index_path.name,
            "sha256": sha256_file(index_path),
        },
        "patch_count": len(patch_ids),
    }
    meta_path = root / "meta.json"
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_meta, meta_path)
    return meta


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
    parser.add_argument(
        "--seal-existing",
        action="store_true",
        help="Verify and index an already completed export without rewriting any map.",
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
    patch_ids = [str(record["patch_id"]) for record in records]
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError("Harbin AEF coverage inventory has duplicate patch IDs")
    if args.seal_existing:
        if not args.output_root.is_dir():
            raise FileNotFoundError(f"Missing completed export: {args.output_root}")
        seal_export(args.output_root, args.coverage_inventory, args.cog_lock, patch_ids)
        return
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent)
    )
    try:
        for index, record in enumerate(records, start=1):
            patch_id = str(record["patch_id"])
            reference = Path(record["reference_grid"]["path"])
            source = args.cache_dir / Path(str(record["aef_2025_uri"])).name
            candidates = [
                source,
                *sorted(
                    path for path in (args.cache_dir / name for name in locked) if path != source
                ),
            ]
            embedding, _validity, _selector = read_complete_aef_mosaic(candidates, reference)
            if tuple(embedding.shape) != (64, 128, 128) or not bool(
                torch.isfinite(embedding).all()
            ):
                raise ValueError(f"invalid complete AEF embedding for {patch_id}")
            output = staging / "harbin" / patch_id
            output.mkdir(parents=True, exist_ok=True)
            torch.save(embedding, output / "annual_2025_embedding_map.pt")
            print(f"[{index}/380] {patch_id}", flush=True)
        seal_export(staging, args.coverage_inventory, args.cog_lock, patch_ids)
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite existing export: {args.output_root}")
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
