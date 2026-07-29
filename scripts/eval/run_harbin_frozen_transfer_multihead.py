#!/usr/bin/env python3
"""Prepare the sealed Harbin frozen-P10C versus AEF multihead matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.eval import run_harbin_strict_conv3x3_matrix as strict
from scripts.eval.run_registered_paper_downstream import (
    _label_tree_hash,
    build_registered_mixed_shot_schedule,
)
from scripts.eval.run_strong_downstream_benchmark import load_manifest, task_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FAMILIES = ("p10c_haidian_frozen_harbin", "aef_annual_2025")
MATCHED_HEADS = ("linear", "wide_mlp", "deep_wide_mlp", "conv3x3", "unet", "deeplab_lite")


@dataclass(frozen=True)
class PreparedMultihead:
    config: dict[str, Any]
    matrix: dict[str, Any]
    families: tuple[dict[str, Any], ...]
    heads: tuple[str, ...]
    patch_ids: tuple[str, ...]
    label_ids: dict[str, str]
    schedules: dict[str, dict[str, Any]]
    lock_path: Path


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_matching_primary_families(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = config.get("primary_families")
    if not isinstance(raw, list):
        raise ValueError("multihead protocol lacks primary families")
    ids = tuple(str(item.get("id")) for item in raw if isinstance(item, dict))
    if any("scratch" in family_id for family_id in ids):
        raise ValueError("Harbin scratch family is prohibited from the primary matrix")
    if ids != PRIMARY_FAMILIES:
        raise ValueError("primary matrix families must be frozen Haidian P10C and AEF only")
    return tuple(raw)


def _require_base_matrix(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    matrix_path = strict.require_file_lock(config, "base_matrix")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("protocol_id") != "harbin_aef_locked_380_strict_conv3x3":
        raise ValueError("multihead base matrix is not the locked Harbin 380-patch protocol")
    return matrix, matrix_path


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    families = require_matching_primary_families(config)
    if config.get("protocol_id") != "harbin_frozen_transfer_multihead_20260729":
        raise ValueError("unexpected Harbin frozen-transfer multihead protocol")
    if tuple(config.get("heads", ())) != MATCHED_HEADS:
        raise ValueError("matched reader suite differs from the locked six-head contract")
    if config.get("highres_mask_diagnostic", {}).get("primary_table") is not False:
        raise ValueError("high-resolution masking diagnostic must remain outside the primary table")
    if (
        config["highres_mask_diagnostic"].get("status")
        != "requires_separate_masked_embedding_export"
    ):
        raise ValueError("high-resolution masking diagnostic must require a separate masked export")
    return families


def _validate_base_contract(
    config: dict[str, Any], matrix: dict[str, Any], families: tuple[dict[str, Any], ...]
) -> None:
    if tuple(config.get("tasks", ())) != tuple(matrix["tasks"]):
        raise ValueError("multihead tasks differ from the locked base matrix")
    for key in ("folds", "shots", "seeds"):
        if tuple(config.get(key, ())) != tuple(matrix[key]):
            raise ValueError(f"multihead {key} differ from the locked base matrix")
    if config.get("threshold_rule") != matrix.get("threshold_rule"):
        raise ValueError("multihead threshold rule differs from the locked base matrix")
    base_families = {str(item["id"]): item for item in matrix["families"]}
    for family in families:
        base = base_families.get(str(family["id"]))
        if base is None or any(
            family.get(key) != base.get(key)
            for key in ("embedding_root", "month", "seal_file", "encoder_status")
        ):
            raise ValueError(
                f"primary family differs from the independently sealed base export: {family}"
            )


def _write_schedule(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"frozen schedule differs from deterministic contract: {path}")
        return
    write_json_atomically(path, payload)


def build_jobs(prepared: PreparedMultihead) -> list[dict[str, Any]]:
    return [
        {
            "family": family["id"],
            "task": task,
            "fold": fold,
            "shot": shot,
            "seed": seed,
            "head": head,
        }
        for family in prepared.families
        for task in prepared.config["tasks"]
        for fold in prepared.config["folds"]
        for shot in prepared.config["shots"]
        for seed in prepared.config["seeds"]
        for head in prepared.heads
    ]


def prepare_multihead_matrix(config_path: Path, output_root: Path) -> PreparedMultihead:
    """Seal the two-family, matched-reader contract before any reader training starts."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    families = _validate_config(config)
    matrix, matrix_path = _require_base_matrix(config)
    _validate_base_contract(config, matrix, families)
    coverage_path = strict.require_file_lock(matrix, "coverage_inventory")
    split_path = strict.require_file_lock(matrix, "spatial_split")
    manifest_path = strict.require_file_lock(matrix, "common_manifest")
    patch_ids = strict.coverage_patch_ids(coverage_path)
    label_ids = strict.coverage_label_ids(coverage_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split.get("folds", [])) != 5:
        raise ValueError("multihead protocol requires five spatial folds")
    for fold in split["folds"]:
        universe = set(fold["train"]) | set(fold["val"]) | set(fold["test"]) | set(fold["buffer"])
        if universe != set(patch_ids):
            raise ValueError("a spatial fold does not partition the frozen 380-patch universe")
    if set(load_manifest(manifest_path, manifest_path.parent)) != set(patch_ids):
        raise ValueError("common manifest differs from the frozen coverage universe")
    family_seals = {
        family["id"]: strict.sealed_family_index(family, patch_ids) for family in families
    }
    label_root = Path(str(matrix["label_root"]))
    schedules: dict[str, dict[str, Any]] = {}
    schedule_root = output_root / "frozen_shot_schedules"
    split_sha256 = strict.sha256_file(split_path)
    label_hashes: dict[str, str] = {}
    for task_name in config["tasks"]:
        task = task_spec(task_name, label_root)
        label_hash = _label_tree_hash(task.label_roots)
        if label_hash != matrix["labels"][task_name]["tree_sha256"]:
            raise ValueError(f"label tree hash changed for {task_name}")
        label_hashes[task_name] = label_hash
        for fold_index in config["folds"]:
            for seed in config["seeds"]:
                schedule = build_registered_mixed_shot_schedule(
                    task_name,
                    list(split["folds"][fold_index]["train"]),
                    fold_index,
                    seed,
                    label_hash,
                    split_sha256,
                    lambda patch_id: strict.label_pixel_counts(task, label_ids, patch_id),
                    budgets=tuple(config["shots"]),
                    protocol_id=config["protocol_id"],
                )
                path = schedule_root / f"{task_name}_fold{fold_index}_seed{seed}.json"
                _write_schedule(path, schedule)
                schedules[str(path.relative_to(output_root))] = {
                    "sha256": strict.sha256_file(path),
                    "available_shots": sorted(schedule["sets"]),
                }
    provisional = PreparedMultihead(
        config=config,
        matrix=matrix,
        families=families,
        heads=tuple(config["heads"]),
        patch_ids=tuple(patch_ids),
        label_ids=label_ids,
        schedules=schedules,
        lock_path=output_root / "matrix_input_lock.json",
    )
    lock = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": canonical_sha256(config),
        "base_matrix_path": str(matrix_path.resolve()),
        "base_matrix_sha256": strict.sha256_file(matrix_path),
        "coverage_patch_count": len(patch_ids),
        "coverage_patch_ids_sha256": hashlib.sha256("\n".join(patch_ids).encode()).hexdigest(),
        "label_id_mapping_sha256": canonical_sha256(label_ids),
        "primary_family_ids": [family["id"] for family in families],
        "family_seals": family_seals,
        "label_hashes": label_hashes,
        "schedules": schedules,
        "head_contract": list(provisional.heads),
        "job_count": len(build_jobs(provisional)),
        "reader": config["reader"],
        "threshold_rule": config["threshold_rule"],
        "highres_mask_diagnostic": config["highres_mask_diagnostic"],
    }
    if (
        provisional.lock_path.exists()
        and json.loads(provisional.lock_path.read_text(encoding="utf-8")) != lock
    ):
        raise ValueError("matrix input lock differs; refuse to mix inputs in one result root")
    write_json_atomically(provisional.lock_path, lock)
    return provisional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_multihead_matrix(args.config, args.output_root)
    print(
        f"[prepared] {len(build_jobs(prepared))} frozen-transfer multihead cells; "
        f"input lock: {prepared.lock_path}",
        flush=True,
    )
    if not args.prepare_only:
        raise NotImplementedError("reader execution is introduced in Task 2")


if __name__ == "__main__":
    main()
