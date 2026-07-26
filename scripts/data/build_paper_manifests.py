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
        default=Path("configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", type=int, default=(40, 80, 150))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="paper_registered_v5_20260726")
    parser.add_argument("--protocol-name", default="rse_v5_registered_20260726")
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


def require_unique_patch_ids(patch_ids: list[str], context: str) -> None:
    """Reject duplicate patch IDs before dictionaries or sets can hide them."""
    duplicates = sorted({patch_id for patch_id in patch_ids if patch_ids.count(patch_id) > 1})
    if duplicates:
        raise ValueError(f"{context} has duplicate patch IDs: {duplicates[:10]}")


def validate_manifest_audit_sidecar(
    audit_path: Path,
    source_manifest_path: Path,
    spatial_split_path: Path,
    subset_registry_path: Path,
) -> dict[str, Any]:
    """Load a fold manifest audit and verify every immutable input hash."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "source_manifest_sha256": sha256(source_manifest_path),
        "spatial_split_sha256": sha256(spatial_split_path),
        "subset_registry_sha256": sha256(subset_registry_path),
    }
    for key, expected_hash in expected.items():
        if audit.get(key) != expected_hash:
            subject = key.removesuffix("_sha256").replace("_", " ")
            raise ValueError(f"Manifest audit {subject} hash mismatch")
    for name, manifest in audit.get("manifests", {}).items():
        path = Path(manifest["path"])
        if not path.exists() or manifest.get("sha256") != sha256(path):
            raise ValueError(f"Manifest audit output hash mismatch: {name}")
    return audit


def materialize_fold_manifests(
    source_manifest_path: Path,
    spatial_split_path: Path,
    subset_registry_path: Path,
    output_dir: Path,
    fold_id: int,
    sizes: tuple[int, ...] = (40, 80, 150),
    seed: int = 42,
    prefix: str = "paper_registered_v5_20260726",
    protocol_name: str = "rse_v5_registered_20260726",
) -> Path:
    """Write one fold's manifests plus a hash-sealed provenance sidecar."""
    source = load_list(source_manifest_path)
    source_ids = [str(record["patch_id"]) for record in source]
    require_unique_patch_ids(source_ids, "Source manifest")
    by_id = dict(zip(source_ids, source, strict=True))
    split = json.loads(spatial_split_path.read_text(encoding="utf-8"))
    registry = json.loads(subset_registry_path.read_text(encoding="utf-8"))
    split_hash = sha256(spatial_split_path)
    if registry.get("source_split_sha256") != split_hash:
        raise ValueError("Subset registry was built from a different spatial split")
    if registry.get("protocol_name") not in (None, protocol_name):
        raise ValueError("Subset registry protocol name does not match manifest protocol")
    try:
        fold = next(item for item in split["folds"] if int(item["fold"]) == fold_id)
    except StopIteration as exc:
        raise ValueError(f"Spatial split has no fold {fold_id}") from exc

    train_ids = list(fold["train"])
    val_ids = list(fold["val"])
    test_ids = list(fold["test"])
    buffer_ids = list(fold.get("buffer", []))
    for name, patch_ids in (
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
        ("buffer", buffer_ids),
    ):
        require_unique_patch_ids(patch_ids, f"Fold {fold_id} {name}")
    memberships = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
        "buffer": set(buffer_ids),
    }
    membership_names = tuple(memberships)
    for index, left in enumerate(membership_names):
        for right in membership_names[index + 1 :]:
            overlap = memberships[left] & memberships[right]
            if overlap:
                raise ValueError(
                    f"Fold {fold_id} membership overlap between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )
    required = set(train_ids) | set(val_ids) | set(test_ids) | set(buffer_ids)
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"Source manifest is missing patch IDs: {missing[:10]}")

    normalized_sizes = tuple(sorted(set(sizes)))
    registry_fold = registry.get("folds", {}).get(str(fold_id))
    if registry_fold is None:
        raise ValueError(f"Subset registry has no fold {fold_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, str]] = {}
    selected_sets: dict[int, set[str]] = {}

    def record_manifest(name: str, ids: list[str]) -> None:
        path = output_dir / f"{prefix}_{name}_fold{fold_id}.json"
        write_manifest(path, [by_id[patch_id] for patch_id in ids])
        manifests[name] = {"path": str(path), "sha256": sha256(path)}

    for size in normalized_sizes:
        selected = list(registry_fold.get(str(size), []))
        if len(set(selected)) != len(selected):
            raise ValueError(f"Registry fold {fold_id} has duplicate selected patch IDs")
        if len(selected) != size or not set(selected).issubset(train_ids):
            raise ValueError(f"Registry fold {fold_id} has invalid {size}-patch subset")
        selected_sets[size] = set(selected)
        path = output_dir / f"{prefix}_train_{size}_fold{fold_id}_seed{seed}.json"
        write_manifest(path, [by_id[patch_id] for patch_id in selected])
        manifests[f"train_{size}"] = {"path": str(path), "sha256": sha256(path)}

    record_manifest("train_pool", train_ids)
    record_manifest("val", val_ids)
    record_manifest("test", test_ids)
    record_manifest("buffer", buffer_ids)
    audit = {
        "schema_version": 2,
        "protocol_name": protocol_name,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "spatial_split": str(spatial_split_path),
        "spatial_split_sha256": split_hash,
        "subset_registry": str(subset_registry_path),
        "subset_registry_sha256": sha256(subset_registry_path),
        "selection_algorithm": registry["selection_algorithm"],
        "fold": fold_id,
        "seed": seed,
        "sizes": list(normalized_sizes),
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
            for left, right in zip(normalized_sizes, normalized_sizes[1:])
        ),
    }
    if any(audit["overlap"].values()) or not audit["strictly_nested"]:
        raise ValueError("Fold manifests are overlapping or not strictly nested")
    audit_path = output_dir / f"{prefix}_fold{fold_id}_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit_path


def main() -> None:
    args = parse_args()
    audit_path = materialize_fold_manifests(
        args.source_manifest,
        args.spatial_split,
        args.subset_registry,
        args.output_dir,
        args.fold,
        tuple(args.sizes),
        args.seed,
        args.prefix,
        args.protocol_name,
    )
    print(audit_path)


if __name__ == "__main__":
    main()
