#!/usr/bin/env python3
"""Build disjoint, nested manifests for the paper experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--spatial-split", type=Path, required=True)
    parser.add_argument(
        "--subset-registry",
        type=Path,
        default=Path("configs/eval/haidian_paper_subsets_40_80_150_seed42.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", type=int, default=(40, 80, 150))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="paper_20260715")
    return parser.parse_args()


def load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a list manifest: {path}")
    return value


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    source = load_list(args.source_manifest)
    by_id = {str(record["patch_id"]): record for record in source}
    split = json.loads(args.spatial_split.read_text(encoding="utf-8"))
    registry = json.loads(args.subset_registry.read_text(encoding="utf-8"))
    split_hash = sha256(args.spatial_split)
    if registry["source_split_sha256"] != split_hash:
        raise ValueError(
            "Subset registry was built from a different spatial split: "
            f"{registry['source_split_sha256']} != {split_hash}"
        )
    fold = split["folds"][args.fold]

    train_ids = list(fold["train"])
    val_ids = list(fold["val"])
    test_ids = list(fold["test"])
    buffer_ids = list(fold.get("buffer", []))
    required = set(train_ids) | set(val_ids) | set(test_ids) | set(buffer_ids)
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"Source manifest is missing patch IDs: {missing[:10]}")

    sizes = sorted(set(args.sizes))
    if not sizes or sizes[-1] > len(train_ids):
        raise ValueError(f"Requested sizes {sizes} exceed train pool of {len(train_ids)}")
    registry_fold = registry["folds"].get(str(args.fold))
    if registry_fold is None:
        raise KeyError(f"Subset registry has no fold {args.fold}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    selected_sets: dict[int, set[str]] = {}
    for size in sizes:
        selected = list(registry_fold.get(str(size), []))
        if len(selected) != size:
            raise ValueError(f"Registry fold {args.fold} has no exact {size}-patch subset")
        if not set(selected).issubset(train_ids):
            raise ValueError(f"Registry fold {args.fold} size {size} is outside train pool")
        selected_sets[size] = set(selected)
        path = args.output_dir / f"{args.prefix}_train_{size}_fold{args.fold}_seed{args.seed}.json"
        write_manifest(path, [by_id[patch_id] for patch_id in selected])
        manifests[f"train_{size}"] = str(path)

    val_path = args.output_dir / f"{args.prefix}_val_fold{args.fold}.json"
    test_path = args.output_dir / f"{args.prefix}_test_fold{args.fold}.json"
    buffer_path = args.output_dir / f"{args.prefix}_buffer_fold{args.fold}.json"
    write_manifest(val_path, [by_id[patch_id] for patch_id in val_ids])
    write_manifest(test_path, [by_id[patch_id] for patch_id in test_ids])
    write_manifest(buffer_path, [by_id[patch_id] for patch_id in buffer_ids])
    manifests.update(val=str(val_path), test=str(test_path), buffer=str(buffer_path))

    audit: dict[str, Any] = {
        "source_manifest": str(args.source_manifest),
        "spatial_split": str(args.spatial_split),
        "spatial_split_sha256": split_hash,
        "subset_registry": str(args.subset_registry),
        "subset_registry_sha256": sha256(args.subset_registry),
        "selection_algorithm": registry["selection_algorithm"],
        "fold": args.fold,
        "seed": args.seed,
        "sizes": sizes,
        "train_pool_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "buffer_count": len(buffer_ids),
        "manifests": manifests,
        "overlap": {
            "train_pool_val": len(set(train_ids) & set(val_ids)),
            "train_pool_test": len(set(train_ids) & set(test_ids)),
            "train_pool_buffer": len(set(train_ids) & set(buffer_ids)),
            "val_test": len(set(val_ids) & set(test_ids)),
        },
        "strictly_nested": all(
            selected_sets[left] < selected_sets[right]
            for left, right in zip(sizes, sizes[1:])
        ),
    }
    audit_path = args.output_dir / f"{args.prefix}_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
