#!/usr/bin/env python3
"""Build deterministic nested spatial subsets for the JRS scaling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json"),
    )
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=Path("configs/regions/haidian_patches.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json"),
    )
    parser.add_argument("--protocol-name", default="rse_v5_registered_20260726")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_centers(path: Path) -> dict[str, np.ndarray]:
    records = json.loads(path.read_text())
    centers: dict[str, np.ndarray] = {}
    for record in records:
        xmin, ymin, xmax, ymax = record["bounds"]
        centers[record["patch_id"]] = np.asarray(
            [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=np.float64
        )
    return centers


def farthest_point_order(ids: list[str], centers: dict[str, np.ndarray]) -> list[str]:
    """Return a label-free, deterministic maximin spatial ordering."""
    ordered_ids = sorted(ids)
    coords = np.stack([centers[patch_id] for patch_id in ordered_ids])
    scale = np.ptp(coords, axis=0)
    scale[scale == 0] = 1.0
    coords = (coords - coords.mean(axis=0)) / scale

    center_dist = np.sum(coords**2, axis=1)
    first = min(range(len(ordered_ids)), key=lambda i: (center_dist[i], ordered_ids[i]))
    selected = [first]
    remaining = set(range(len(ordered_ids))) - {first}
    min_dist = np.sum((coords - coords[first]) ** 2, axis=1)

    while remaining:
        next_idx = min(
            remaining,
            key=lambda i: (-float(min_dist[i]), ordered_ids[i]),
        )
        selected.append(next_idx)
        remaining.remove(next_idx)
        dist = np.sum((coords - coords[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)

    return [ordered_ids[i] for i in selected]


def build_registry(
    split_path: Path,
    patch_metadata_path: Path,
    budgets: tuple[int, ...] = (40, 80, 150),
    protocol_name: str = "rse_v5_registered_20260726",
) -> dict[str, object]:
    """Build a deterministic, label-free subset registry bound to one split."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    centers = patch_centers(patch_metadata_path)
    folds: dict[str, dict[str, list[str]]] = {}
    for fold in split["folds"]:
        fold_id = str(fold["fold"])
        train_ids = list(fold["train"])
        if len(train_ids) < max(budgets):
            raise ValueError(f"Fold {fold_id} has only {len(train_ids)} eligible patches")
        order = farthest_point_order(train_ids, centers)
        folds[fold_id] = {str(budget): order[:budget] for budget in budgets}
    return {
        "schema_version": 2,
        "protocol_name": protocol_name,
        "seed": 42,
        "selection_algorithm": "deterministic_label_free_maximin_farthest_point",
        "source_split": str(split_path),
        "source_split_sha256": sha256(split_path),
        "patch_metadata": str(patch_metadata_path),
        "patch_metadata_sha256": sha256(patch_metadata_path),
        "budgets": list(budgets),
        "folds": folds,
    }


def validate_registry_split(registry: dict[str, object], split_path: Path) -> None:
    """Reject a registry that was not created from the exact requested split."""
    expected = sha256(split_path)
    actual = registry.get("source_split_sha256")
    if actual != expected:
        raise ValueError(f"Subset registry source split hash mismatch: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    budgets = (40, 80, 150)
    payload = build_registry(args.split, args.patch_metadata, budgets, args.protocol_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
