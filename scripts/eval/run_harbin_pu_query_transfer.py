#!/usr/bin/env python3
"""Prepare the lock-driven Harbin polygon-prompt PU+Query transfer protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import label

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_label_id(patch_id: str, mapping: dict[str, str]) -> str:
    if patch_id not in mapping:
        raise KeyError(f"unlocked label mapping for {patch_id}")
    return mapping[patch_id]


@dataclass(frozen=True)
class PreparedProtocol:
    patch_ids: list[str]
    label_ids: dict[str, str]
    schedules: dict[str, dict[str, Any]]
    matrix: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    output_root: Path | None = None

    def schedule_for(self, task: str, fold: int, seed: int, polygon_count: int) -> dict[str, Any]:
        key = f"{task}|{fold}|{seed}|{polygon_count}"
        if key not in self.schedules:
            raise KeyError(f"unlocked polygon schedule for {key}")
        return self.schedules[key]


def read_mask(label_root: Path, source_patch_id: str) -> np.ndarray:
    path = label_root / "masks" / f"{source_patch_id}.tif"
    with rasterio.open(path) as source:
        return source.read(1).astype(np.uint8) == 1


def polygon_candidates(
    label_root: Path,
    train_patch_ids: list[str],
    label_ids: dict[str, str],
    minimum_area: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for patch_id in sorted(train_patch_ids):
        source_patch_id = resolve_label_id(patch_id, label_ids)
        components, count = label(
            read_mask(label_root, source_patch_id), structure=np.ones((3, 3), dtype=np.uint8)
        )
        for component_index in range(1, count + 1):
            area = int((components == component_index).sum())
            if area >= minimum_area:
                candidates.append(
                    {
                        "patch_id": patch_id,
                        "source_patch_id": source_patch_id,
                        "component_index": component_index,
                        "area_pixels": area,
                    }
                )
    return candidates


def build_schedule(
    candidates: list[dict[str, Any]],
    *,
    task: str,
    fold: int,
    seed: int,
    polygon_counts: list[int],
) -> dict[str, dict[str, Any]]:
    if not candidates:
        raise ValueError(f"{task}/fold{fold} has no usable train polygons")
    order = np.random.default_rng(seed + fold * 1009 + sum(map(ord, task))).permutation(
        len(candidates)
    )
    ordered = [candidates[int(index)] for index in order]
    largest = max(polygon_counts)
    if len(ordered) < largest:
        raise ValueError(f"{task}/fold{fold} has {len(ordered)} polygons, below required {largest}")
    schedules: dict[str, dict[str, Any]] = {}
    for polygon_count in polygon_counts:
        selected = ordered[:polygon_count]
        schedules[f"{task}|{fold}|{seed}|{polygon_count}"] = {
            "task": task,
            "fold": fold,
            "seed": seed,
            "polygon_count": polygon_count,
            "candidate_count": len(ordered),
            "support_polygons": selected,
            "support_polygon_sha256": canonical_sha256(selected),
        }
    return schedules


def _locked_file(record: dict[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"PU+Query config lacks {key} lock")
    path = resolve_path(value["path"])
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"PU+Query config {key} lock does not match")
    return path


def prepare_harbin_pu_query(config_path: Path, output_root: Path) -> PreparedProtocol:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("coverage_patch_count") != 380:
        raise ValueError("Harbin PU+Query protocol requires exactly 380 coverage patches")
    if config.get("protocol_id") != "harbin_pu_query_transfer_20260729":
        raise ValueError("unexpected Harbin PU+Query protocol ID")
    matrix_path = _locked_file(config, "base_matrix")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("protocol_id") != "harbin_aef_locked_380_strict_conv3x3":
        raise ValueError("PU+Query base matrix is not the locked Harbin paired protocol")
    input_lock_path = _locked_file(config, "base_matrix_input_lock")
    input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))
    if input_lock.get("matrix_sha256") != sha256_file(matrix_path):
        raise ValueError("base matrix input lock does not bind the selected matrix")
    if input_lock.get("coverage_patch_count") != 380 or input_lock.get("job_count") != 270:
        raise ValueError("base matrix input lock has an unexpected coverage or cell count")
    coverage_ref = matrix["coverage_inventory"]
    coverage_path = resolve_path(coverage_ref["path"])
    if sha256_file(coverage_path) != coverage_ref["sha256"]:
        raise ValueError("Harbin coverage inventory differs from base matrix lock")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    records = coverage.get("records")
    if not isinstance(records, list) or len(records) != 380:
        raise ValueError("Harbin coverage inventory must contain exactly 380 records")
    label_ids = {str(record["patch_id"]): str(record["source_patch_id"]) for record in records}
    patch_ids = sorted(label_ids)
    if len(patch_ids) != 380 or len(set(label_ids.values())) != 380:
        raise ValueError("Harbin coverage inventory has a non-bijective label mapping")
    split_ref = matrix["spatial_split"]
    split_path = resolve_path(split_ref["path"])
    if sha256_file(split_path) != split_ref["sha256"]:
        raise ValueError("Harbin spatial split differs from base matrix lock")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split.get("folds", [])) != 5:
        raise ValueError("Harbin PU+Query protocol requires five spatial folds")
    label_root = Path(matrix["label_root"])
    polygon_counts = list(config["polygon_counts"])
    if polygon_counts != [1, 3, 5, 9]:
        raise ValueError("polygon prompt budgets must be exactly [1, 3, 5, 9]")
    schedules: dict[str, dict[str, Any]] = {}
    schedule_dir = output_root / "frozen_polygon_schedules"
    for task in matrix["tasks"]:
        task_root = (
            label_root
            / {"building": "building_osm", "road": "road_osm", "water": "osm_water"}[task]
        )
        for fold_index in matrix["folds"]:
            fold = split["folds"][fold_index]
            universe = (
                set(fold["train"]) | set(fold["val"]) | set(fold["test"]) | set(fold["buffer"])
            )
            if universe != set(patch_ids):
                raise ValueError(f"fold {fold_index} does not partition the 380 locked patches")
            candidates = polygon_candidates(
                task_root,
                list(fold["train"]),
                label_ids,
                int(config["minimum_component_area"]),
            )
            for seed in matrix["seeds"]:
                schedules.update(
                    build_schedule(
                        candidates,
                        task=task,
                        fold=fold_index,
                        seed=seed,
                        polygon_counts=polygon_counts,
                    )
                )
                payload = {
                    "protocol_id": config["protocol_id"],
                    "base_matrix_sha256": sha256_file(matrix_path),
                    "coverage_inventory_sha256": sha256_file(coverage_path),
                    "spatial_split_sha256": sha256_file(split_path),
                    "task": task,
                    "fold": fold_index,
                    "seed": seed,
                    "sets": {
                        str(count): schedules[f"{task}|{fold_index}|{seed}|{count}"]
                        for count in polygon_counts
                    },
                }
                destination = schedule_dir / f"{task}_fold{fold_index}_seed{seed}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (
                    destination.exists()
                    and json.loads(destination.read_text(encoding="utf-8")) != payload
                ):
                    raise ValueError(f"existing frozen polygon schedule differs: {destination}")
                destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return PreparedProtocol(
        patch_ids=patch_ids,
        label_ids=label_ids,
        schedules=schedules,
        matrix=matrix,
        config=config,
        output_root=output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_harbin_pu_query(args.config, args.output_root)
    print(
        "prepared "
        f"{len(prepared.patch_ids)} patches and {len(prepared.schedules)} polygon schedules",
        flush=True,
    )
    if not args.prepare_only:
        raise NotImplementedError("PU+Query scoring is added in Task 2")


if __name__ == "__main__":
    main()
