#!/usr/bin/env python3
"""Deterministically select and split China V1 candidate points into shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rank(seed: int, patch_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{patch_id}".encode("utf-8")).digest()


def select_points(input_path: Path, count: int, seed: int) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    seen: set[str] = set()
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            point = json.loads(line)
            patch_id = point.get("patch_id")
            if not isinstance(patch_id, str) or not patch_id:
                raise ValueError(f"line {line_number} has no patch_id")
            if patch_id in seen:
                raise ValueError(f"duplicate patch_id: {patch_id}")
            seen.add(patch_id)
            points.append(point)
    if count > len(points):
        raise ValueError(f"requested {count} points but only {len(points)} available")
    return sorted(points, key=lambda point: _rank(seed, str(point["patch_id"])))[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--prefix", default="pilot")
    args = parser.parse_args()
    if args.count <= 0 or args.shard_size <= 0:
        raise ValueError("--count and --shard-size must be positive")
    selected = select_points(args.input, args.count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, len(selected), args.shard_size)):
        output = args.output_dir / f"{args.prefix}_{index:03d}.jsonl"
        chunk = selected[start:start + args.shard_size]
        output.write_text("".join(json.dumps(point, ensure_ascii=False) + "\n" for point in chunk), encoding="utf-8")
    manifest = {
        "schema_version": "china_v1_shard_selection_v1", "input": str(args.input), "count": len(selected),
        "shard_size": args.shard_size, "seed": args.seed, "prefix": args.prefix,
        "patch_ids_sha256": hashlib.sha256("\n".join(str(point["patch_id"]) for point in selected).encode("utf-8")).hexdigest(),
    }
    (args.output_dir / f"{args.prefix}_selection.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
