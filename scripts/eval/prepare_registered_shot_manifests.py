#!/usr/bin/env python3
"""Generate immutable shared few-shot schedules for registered paper probes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from scripts.eval.run_registered_paper_downstream import (
    _assert_binary_label_roots,
    _label_tree_hash,
    _positive_background_pixel_counts,
    _verify_frozen_split,
    build_registered_mixed_shot_schedule,
    sha256_file,
)
from scripts.eval.run_traditional_ml_benchmark import task_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--spatial-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("v4_diagnostic", "v5_osm_assisted"),
        default="v4_diagnostic",
    )
    parser.add_argument("--tasks", nargs="+", default=["building", "road", "water"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return parser.parse_args()


def write_or_verify(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace a different frozen shot schedule: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise ValueError(f"Refusing to replace a different frozen shot schedule: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    split = json.loads(args.spatial_split.read_text(encoding="utf-8"))
    split_hash = sha256_file(args.spatial_split)
    for fold_data in split["folds"]:
        _verify_frozen_split(args.spatial_split, int(fold_data["fold"]), args.protocol)
    index: list[dict[str, object]] = []
    for task_name in args.tasks:
        task = task_spec(task_name, args.label_root)
        _assert_binary_label_roots(task.label_roots)
        label_hash = _label_tree_hash(task.label_roots)
        for fold_data in split["folds"]:
            fold = int(fold_data["fold"])
            train_ids = [str(patch_id) for patch_id in fold_data["train"]]
            for seed in args.seeds:
                schedule = build_registered_mixed_shot_schedule(
                    task_name=task_name,
                    train_ids=train_ids,
                    fold=fold,
                    seed=seed,
                    label_sha256=label_hash,
                    split_sha256=split_hash,
                    pixel_counts=lambda patch_id, spec=task: _positive_background_pixel_counts(
                        spec, patch_id
                    ),
                    protocol_id=args.protocol,
                )
                path = args.output_root / f"{task_name}_fold{fold}_seed{seed}.json"
                write_or_verify(path, schedule)
                index.append(
                    {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "task": task_name,
                        "fold": fold,
                        "seed": seed,
                        "available_budgets": sorted(schedule["sets"]),
                        "unavailable_budgets": sorted(schedule["unavailable"]),
                    }
                )
    index_payload: dict[str, object] = {
        "schema_version": 1,
        "split_sha256": split_hash,
        "schedules": index,
    }
    if args.protocol == "v5_osm_assisted":
        index_payload["protocol_id"] = args.protocol
    write_or_verify(args.output_root / "index.json", index_payload)


if __name__ == "__main__":
    main()
