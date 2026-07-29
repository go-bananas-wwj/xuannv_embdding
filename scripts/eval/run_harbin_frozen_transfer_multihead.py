#!/usr/bin/env python3
"""Prepare the sealed Harbin frozen-P10C versus AEF multihead matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.eval import run_harbin_strict_conv3x3_matrix as strict
from scripts.eval.run_registered_paper_downstream import (
    _confusion,
    _fit_train_standardizer,
    _label_tree_hash,
    _logits_for_items,
    _normalize_items,
    _set_seed,
    build_registered_mixed_shot_schedule,
    build_validation_threshold_selection,
    compute_registered_metrics,
    load_registered_shot_ids,
    sigmoid_probabilities,
    target_support_sha256,
)
from scripts.eval.run_strong_downstream_benchmark import (
    load_manifest,
    make_device,
    make_model,
    pos_weight_for,
    task_spec,
    train_epoch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FAMILIES = ("p10c_haidian_frozen_harbin", "aef_annual_2025")
MATCHED_HEADS = ("linear", "wide_mlp", "deep_wide_mlp", "conv3x3", "unet", "deeplab_lite")


class LinearReader(torch.nn.Conv2d):
    """A binary 1×1 probe with no spatial receptive field beyond one pixel."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__(embed_dim, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return super().forward(values).squeeze(1)


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


def build_reader(head: str, embed_dim: int) -> torch.nn.Module:
    """Build one locked reader architecture for either primary representation."""
    if head == "linear":
        return LinearReader(embed_dim)
    if head not in MATCHED_HEADS:
        raise ValueError(f"unsupported matched reader: {head}")
    return make_model(head, embed_dim, 32)


def write_cell_atomically(
    output: Path,
    metrics: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Publish a cell only after its metrics and both prediction archives are complete."""
    if output.exists():
        raise FileExistsError(f"refusing to replace existing result output: {output}")
    if set(predictions) != {"validation", "test"}:
        raise ValueError("atomic result bundle requires validation and test predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    try:
        staging.mkdir()
        for split, payload in predictions.items():
            np.savez_compressed(staging / f"predictions_{split}.npz", **payload)
        strict._write_json(staging / "metrics.json", metrics)
        if model_state is not None:
            torch.save(model_state, staging / "final_reader.pt")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def output_for(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "results"
        / job["family"]
        / job["task"]
        / f"fold{job['fold']}"
        / f"shot{job['shot']}"
        / f"seed{job['seed']}"
        / job["head"]
    )


def _load_worker_prepared(config_path: Path, output_root: Path) -> PreparedMultihead:
    """Load the already sealed parent contract without rehashing every embedding on each NPU."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    families = _validate_config(config)
    matrix, matrix_path = _require_base_matrix(config)
    _validate_base_contract(config, matrix, families)
    lock_path = output_root / "matrix_input_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("protocol_id") != config["protocol_id"]
        or lock.get("config_sha256") != canonical_sha256(config)
        or lock.get("base_matrix_sha256") != strict.sha256_file(matrix_path)
        or lock.get("primary_family_ids") != [family["id"] for family in families]
    ):
        raise ValueError("matrix worker input lock does not match the sealed primary contract")
    coverage_path = strict.require_file_lock(matrix, "coverage_inventory")
    return PreparedMultihead(
        config=config,
        matrix=matrix,
        families=families,
        heads=tuple(config["heads"]),
        patch_ids=tuple(strict.coverage_patch_ids(coverage_path)),
        label_ids=strict.coverage_label_ids(coverage_path),
        schedules=lock["schedules"],
        lock_path=lock_path,
    )


def run_reader_cell(
    prepared: PreparedMultihead,
    family: str,
    task_name: str,
    fold: int,
    shot: int,
    seed: int,
    head: str,
    device: str,
) -> dict[str, Any]:
    """Train one matched reader from support only and score it with validation calibration."""
    job = {
        "family": family,
        "task": task_name,
        "fold": fold,
        "shot": shot,
        "seed": seed,
        "head": head,
    }
    output = output_for(prepared.lock_path.parent, job)
    required = (
        "metrics.json",
        "predictions_validation.npz",
        "predictions_test.npz",
        "final_reader.pt",
    )
    if output.is_dir() and all((output / name).is_file() for name in required):
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"refusing to reuse incomplete result directory: {output}")
    family_specs = {item["id"]: item for item in prepared.families}
    if family not in family_specs or "scratch" in family:
        raise ValueError("reader cell family is outside the frozen primary matrix")
    if head not in prepared.heads:
        raise ValueError("reader cell head is outside the matched contract")
    split_path = strict.repo_path(prepared.matrix["spatial_split"]["path"])
    split = json.loads(split_path.read_text(encoding="utf-8"))["folds"][fold]
    task = task_spec(task_name, Path(str(prepared.matrix["label_root"])))
    schedule_path = (
        prepared.lock_path.parent
        / "frozen_shot_schedules"
        / f"{task_name}_fold{fold}_seed{seed}.json"
    )
    expected_schedule = build_registered_mixed_shot_schedule(
        task_name,
        list(split["train"]),
        fold,
        seed,
        _label_tree_hash(task.label_roots),
        strict.sha256_file(split_path),
        lambda patch_id: strict.label_pixel_counts(task, prepared.label_ids, patch_id),
        budgets=tuple(prepared.config["shots"]),
        protocol_id=prepared.config["protocol_id"],
    )
    train_ids, schedule = load_registered_shot_ids(
        schedule_path,
        expected_schedule=expected_schedule,
        shot=str(shot),
        protocol_id=prepared.config["protocol_id"],
    )
    probe_seed = int.from_bytes(
        hashlib.sha256(f"{fold}|{task_name}|{seed}|{head}".encode()).digest()[:4], "little"
    )
    _set_seed(probe_seed)
    spec = family_specs[family]
    embedding_root = Path(str(spec["embedding_root"]))
    month = str(spec["month"])
    train_items = strict.load_harbin_items(
        task, train_ids, prepared.label_ids, embedding_root, month
    )
    validation_items = strict.load_harbin_items(
        task, list(split["val"]), prepared.label_ids, embedding_root, month
    )
    test_items = strict.load_harbin_items(
        task, list(split["test"]), prepared.label_ids, embedding_root, month
    )
    mean, std = _fit_train_standardizer(train_items)
    train_items = _normalize_items(train_items, mean, std)
    validation_items = _normalize_items(validation_items, mean, std)
    test_items = _normalize_items(test_items, mean, std)
    runtime_device = make_device(device)
    model = build_reader(head, 64).to(runtime_device)
    reader = prepared.config["reader"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=reader["lr"], weight_decay=reader["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=reader["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_items, runtime_device))
    start = time.perf_counter()
    for _ in range(reader["epochs"]):
        train_epoch(model, train_items, runtime_device, optimizer, loss_fn, reader["batch_size"])
        scheduler.step()
    _, _, validation_by_patch = _logits_for_items(
        model, validation_items, runtime_device, reader["batch_size"]
    )
    _, _, test_by_patch = _logits_for_items(model, test_items, runtime_device, reader["batch_size"])
    validation_ids = sorted(validation_by_patch)
    validation_probabilities = np.stack(
        [sigmoid_probabilities(validation_by_patch[item][0]) for item in validation_ids]
    )
    validation_targets = np.stack([validation_by_patch[item][1] for item in validation_ids])
    selection = build_validation_threshold_selection(
        validation_probabilities.reshape(-1), validation_targets.reshape(-1)
    )
    test_ids = sorted(test_by_patch)
    test_probabilities = np.stack(
        [sigmoid_probabilities(test_by_patch[item][0]) for item in test_ids]
    )
    test_targets = np.stack([test_by_patch[item][1] for item in test_ids])
    threshold = float(selection["selected_threshold"])
    metrics = compute_registered_metrics(
        test_probabilities.reshape(-1), test_targets.reshape(-1), threshold
    )
    payload = {
        **metrics,
        "average_precision": float(metrics["ap"]),
        "protocol_id": prepared.config["protocol_id"],
        **job,
        "probe_seed": probe_seed,
        "reader": {"head": head, **reader},
        "threshold": threshold,
        "validation_threshold_selection": selection,
        "matrix_input_lock": str(prepared.lock_path.resolve()),
        "matrix_input_lock_sha256": strict.sha256_file(prepared.lock_path),
        "shot_schedule": str(schedule_path.resolve()),
        "shot_schedule_sha256": strict.sha256_file(schedule_path),
        "shot_schedule_record": schedule,
        "normalization": {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()},
        "train_seconds": time.perf_counter() - start,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "per_patch_confusion": {
            patch_id: _confusion(test_targets[index], test_probabilities[index], threshold)
            for index, patch_id in enumerate(test_ids)
        },
        "per_patch_target_support_sha256": {
            patch_id: target_support_sha256(test_targets[index])
            for index, patch_id in enumerate(test_ids)
        },
    }
    write_cell_atomically(
        output,
        payload,
        {
            "validation": {
                "patch_ids": np.asarray(validation_ids),
                "probability_maps": validation_probabilities,
                "target_maps": validation_targets,
            },
            "test": {
                "patch_ids": np.asarray(test_ids),
                "probability_maps": test_probabilities,
                "target_maps": test_targets,
            },
        },
        model.state_dict(),
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument(
        "--devices", nargs="+", default=["npu:0", "npu:1", "npu:2", "npu:3", "npu:4", "npu:5"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.device is None:
            raise ValueError("worker requires --device")
        if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
            raise ValueError("worker index must be inside the positive worker count")
        prepared = _load_worker_prepared(args.config, args.output_root)
        for index, job in enumerate(build_jobs(prepared)):
            if index % args.worker_count == args.worker_index:
                run_reader_cell(prepared, device=args.device, **job)
        return
    prepared = prepare_multihead_matrix(args.config, args.output_root)
    print(
        f"[prepared] {len(build_jobs(prepared))} frozen-transfer multihead cells; "
        f"input lock: {prepared.lock_path}",
        flush=True,
    )
    if args.prepare_only:
        return
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--output-root",
            str(args.output_root),
            "--worker",
            "--device",
            device,
            "--worker-index",
            str(index),
            "--worker-count",
            str(len(args.devices)),
        ]
        for index, device in enumerate(args.devices)
    ]
    processes = [subprocess.Popen(command) for command in commands]
    exit_codes = [process.wait() for process in processes]
    if any(exit_codes):
        raise RuntimeError(f"multihead matrix worker failures: {exit_codes}")


if __name__ == "__main__":
    main()
