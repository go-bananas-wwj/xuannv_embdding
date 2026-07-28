#!/usr/bin/env python3
"""Run the sealed 380-patch Harbin three-family Conv3x3 evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.eval.run_registered_paper_downstream import (
    FROZEN_PROBE,
    _confusion,
    _fit_train_standardizer,
    _label_tree_hash,
    _logits_for_items,
    _normalize_items,
    _set_seed,
    _write_json,
    build_registered_mixed_shot_schedule,
    build_validation_threshold_selection,
    compute_registered_metrics,
    load_registered_shot_ids,
    prepare_result_output,
    sigmoid_probabilities,
    target_support_sha256,
    write_prediction_archive,
)
from scripts.eval.run_strong_downstream_benchmark import (
    PatchTensor,
    load_manifest,
    make_device,
    make_model,
    pos_weight_for,
    task_spec,
    train_epoch,
)
from scripts.eval.run_traditional_ml_benchmark import load_binary_mask

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


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def require_file_lock(record: dict[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"matrix lacks locked {key}")
    path = repo_path(value["path"])
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"matrix {key} does not match its SHA-256 lock")
    return path


def coverage_patch_ids(coverage_path: Path) -> list[str]:
    raw = json.loads(coverage_path.read_text(encoding="utf-8"))
    records = raw.get("records")
    if not isinstance(records, list):
        raise ValueError("coverage inventory lacks records")
    patch_ids = sorted(str(record["patch_id"]) for record in records)
    if len(patch_ids) != 380 or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("strict Harbin matrix requires exactly 380 unique covered patches")
    return patch_ids


def coverage_label_ids(coverage_path: Path) -> dict[str, str]:
    """Bind each exported Harbin map ID to the label-mask ID frozen in inventory."""
    records = json.loads(coverage_path.read_text(encoding="utf-8")).get("records")
    if not isinstance(records, list):
        raise ValueError("coverage inventory lacks records")
    mapping = {str(record["patch_id"]): str(record["source_patch_id"]) for record in records}
    if len(mapping) != 380 or len(set(mapping.values())) != 380:
        raise ValueError("coverage inventory lacks a one-to-one embedding/label ID mapping")
    return mapping


def label_pixel_counts(task: Any, label_ids: dict[str, str], patch_id: str) -> tuple[int, int]:
    values = load_binary_mask(task, label_ids[patch_id])
    return int((values == 1).sum()), int((values == 0).sum())


def load_harbin_items(
    task: Any,
    patch_ids: list[str],
    label_ids: dict[str, str],
    embedding_root: Path,
    month: str,
) -> list[PatchTensor]:
    """Pair each representation map with its locked source-label patch, never by guessing IDs."""
    items: list[PatchTensor] = []
    for patch_id in patch_ids:
        path = embedding_root / "harbin" / patch_id / f"{month}_embedding_map.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing sealed embedding map: {path}")
        embedding = torch.load(path, map_location="cpu", weights_only=True).float()
        target = load_binary_mask(task, label_ids[patch_id])
        if tuple(embedding.shape) != (64, 128, 128) or embedding.shape[-2:] != target.shape:
            raise ValueError(f"embedding/label shape mismatch for {patch_id}")
        items.append(
            PatchTensor(
                patch_id=patch_id,
                x=embedding,
                y=torch.from_numpy((target == 1).astype(np.float32)),
            )
        )
    return items


def sealed_family_index(family: dict[str, Any], expected_patch_ids: list[str]) -> dict[str, Any]:
    """Load the existing independent export seal and validate every map hash once."""
    root = Path(str(family["embedding_root"]))
    seal_path = root / str(family["seal_file"])
    if not seal_path.is_file():
        raise FileNotFoundError(f"Missing family seal: {seal_path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if family["id"] == "aef_annual_2025":
        if seal.get("protocol_id") != "harbin_aef_annual_2025_locked":
            raise ValueError("AEF metadata has an unexpected protocol ID")
        index_ref = seal.get("embedding_file_index")
        if not isinstance(index_ref, dict) or index_ref.get("path") != "embedding_file_index.json":
            raise ValueError("AEF metadata does not bind an embedding file index")
        index_path = root / str(index_ref["path"])
        if sha256_file(index_path) != index_ref.get("sha256"):
            raise ValueError("AEF embedding file index hash differs from metadata")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = index.get("files")
    else:
        verification = seal.get("verification")
        if not isinstance(verification, dict):
            raise ValueError(f"{family['id']} export provenance lacks verification")
        files = [
            {
                "patch_id": patch_id,
                "path": f"harbin/{patch_id}/{family['month']}_embedding_map.pt",
                "sha256": checksum,
            }
            for patch_id, checksum in verification.get("files", {}).items()
        ]
    if not isinstance(files, list):
        raise ValueError(f"{family['id']} seal lacks indexed maps")
    indexed = {str(item.get("patch_id")): item for item in files if isinstance(item, dict)}
    if sorted(indexed) != expected_patch_ids:
        raise ValueError(f"{family['id']} sealed patch set differs from frozen coverage inventory")
    for patch_id in expected_patch_ids:
        item = indexed[patch_id]
        path = root / str(item["path"])
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"{family['id']} embedding map does not match seal: {patch_id}")
    return {
        "seal_path": str(seal_path.resolve()),
        "seal_sha256": sha256_file(seal_path),
        "map_index_sha256": canonical_sha256(files),
        "patch_count": len(indexed),
    }


def build_jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"family": family["id"], "task": task, "fold": fold, "shot": shot, "seed": seed}
        for family in matrix["families"]
        for task in matrix["tasks"]
        for fold in matrix["folds"]
        for shot in matrix["shots"]
        for seed in matrix["seeds"]
    ]


def prepare_matrix(matrix_path: Path, output_root: Path) -> tuple[dict[str, Any], Path]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("protocol_id") != "harbin_aef_locked_380_strict_conv3x3":
        raise ValueError("unexpected strict Harbin matrix protocol")
    coverage_path = require_file_lock(matrix, "coverage_inventory")
    split_path = require_file_lock(matrix, "spatial_split")
    manifest_path = require_file_lock(matrix, "common_manifest")
    patch_ids = coverage_patch_ids(coverage_path)
    label_ids = coverage_label_ids(coverage_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(split.get("folds", [])) != 5:
        raise ValueError("strict Harbin matrix requires five spatial folds")
    for fold in split["folds"]:
        observed = set(fold["train"]) | set(fold["val"]) | set(fold["test"]) | set(fold["buffer"])
        if observed != set(patch_ids):
            raise ValueError("a spatial fold does not partition the frozen 380-patch universe")
    records = load_manifest(manifest_path, manifest_path.parent)
    if set(records) != set(patch_ids):
        raise ValueError("common manifest does not match the frozen 380-patch universe")
    family_seals = {
        family["id"]: sealed_family_index(family, patch_ids) for family in matrix["families"]
    }
    label_root = Path(str(matrix["label_root"]))
    label_hashes: dict[str, str] = {}
    schedules: dict[str, dict[str, Any]] = {}
    schedule_root = output_root / "frozen_shot_schedules"
    schedule_root.mkdir(parents=True, exist_ok=True)
    split_sha = sha256_file(split_path)
    for task_name in matrix["tasks"]:
        task = task_spec(task_name, label_root)
        label_hash = _label_tree_hash(task.label_roots)
        if label_hash != matrix["labels"][task_name]["tree_sha256"]:
            raise ValueError(f"label tree hash changed for {task_name}")
        label_hashes[task_name] = label_hash
        for fold_index in matrix["folds"]:
            train_ids = list(split["folds"][fold_index]["train"])
            for seed in matrix["seeds"]:
                schedule = build_registered_mixed_shot_schedule(
                    task_name,
                    train_ids,
                    fold_index,
                    seed,
                    label_hash,
                    split_sha,
                    lambda patch_id: label_pixel_counts(task, label_ids, patch_id),
                    budgets=tuple(matrix["shots"]),
                    protocol_id="harbin_aef_locked_380_strict_conv3x3",
                )
                schedule_path = schedule_root / f"{task_name}_fold{fold_index}_seed{seed}.json"
                if schedule_path.exists():
                    if json.loads(schedule_path.read_text(encoding="utf-8")) != schedule:
                        raise ValueError(
                            f"frozen schedule differs from deterministic contract: {schedule_path}"
                        )
                else:
                    _write_json(schedule_path, schedule)
                schedules[str(schedule_path.relative_to(output_root))] = {
                    "sha256": sha256_file(schedule_path),
                    "available_shots": sorted(schedule["sets"]),
                }
    lock = {
        "schema_version": 1,
        "protocol_id": matrix["protocol_id"],
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "coverage_patch_count": len(patch_ids),
        "coverage_patch_ids_sha256": hashlib.sha256("\n".join(patch_ids).encode()).hexdigest(),
        "label_id_mapping_sha256": canonical_sha256(label_ids),
        "family_seals": family_seals,
        "label_hashes": label_hashes,
        "schedules": schedules,
        "job_count": len(build_jobs(matrix)),
        "threshold_rule": matrix["threshold_rule"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "matrix_input_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("matrix input lock differs; refuse to mix inputs in one result root")
    _write_json(lock_path, lock)
    return matrix, lock_path


def output_for(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "results"
        / job["family"]
        / job["task"]
        / f"fold{job['fold']}"
        / f"shot{job['shot']}"
        / f"seed{job['seed']}"
    )


def run_job(
    matrix: dict[str, Any], output_root: Path, lock_path: Path, job: dict[str, Any], device: str
) -> None:
    output = output_for(output_root, job)
    if (output / "metrics.json").is_file() and (output / "predictions_test.npz").is_file():
        print(f"[complete] {job}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"refusing to reuse incomplete result directory: {output}")
    families = {family["id"]: family for family in matrix["families"]}
    family = families[job["family"]]
    split_path = repo_path(matrix["spatial_split"]["path"])
    coverage_path = repo_path(matrix["coverage_inventory"]["path"])
    label_ids = coverage_label_ids(coverage_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))["folds"][job["fold"]]
    task = task_spec(job["task"], Path(str(matrix["label_root"])))
    schedule_path = (
        output_root
        / "frozen_shot_schedules"
        / (f"{job['task']}_fold{job['fold']}_seed{job['seed']}.json")
    )
    expected_schedule = build_registered_mixed_shot_schedule(
        job["task"],
        list(split["train"]),
        job["fold"],
        job["seed"],
        _label_tree_hash(task.label_roots),
        sha256_file(split_path),
        lambda patch_id: label_pixel_counts(task, label_ids, patch_id),
        budgets=tuple(matrix["shots"]),
        protocol_id=matrix["protocol_id"],
    )
    train_ids, schedule = load_registered_shot_ids(
        schedule_path,
        expected_schedule=expected_schedule,
        shot=str(job["shot"]),
        protocol_id=matrix["protocol_id"],
    )
    probe_seed = int.from_bytes(
        hashlib.sha256(f"{job['fold']}|{job['task']}|{job['seed']}".encode()).digest()[:4], "little"
    )
    _set_seed(probe_seed)
    root = Path(str(family["embedding_root"]))
    month = str(family["month"])
    train_items = load_harbin_items(task, train_ids, label_ids, root, month)
    val_items = load_harbin_items(task, list(split["val"]), label_ids, root, month)
    test_items = load_harbin_items(task, list(split["test"]), label_ids, root, month)
    mean, std = _fit_train_standardizer(train_items)
    train_items = _normalize_items(train_items, mean, std)
    val_items = _normalize_items(val_items, mean, std)
    test_items = _normalize_items(test_items, mean, std)
    prepare_result_output(output)
    runtime_device = make_device(device)
    model = make_model("conv3x3", 64, 64).to(runtime_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_PROBE["lr"], weight_decay=FROZEN_PROBE["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_PROBE["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_items, runtime_device))
    start = time.perf_counter()
    for _ in range(FROZEN_PROBE["epochs"]):
        train_epoch(
            model, train_items, runtime_device, optimizer, loss_fn, FROZEN_PROBE["batch_size"]
        )
        scheduler.step()
    _, _, val_by_patch = _logits_for_items(
        model, val_items, runtime_device, FROZEN_PROBE["batch_size"]
    )
    _, _, test_by_patch = _logits_for_items(
        model, test_items, runtime_device, FROZEN_PROBE["batch_size"]
    )
    val_ids = sorted(val_by_patch)
    val_probs = np.stack([sigmoid_probabilities(val_by_patch[key][0]) for key in val_ids])
    val_targets = np.stack([val_by_patch[key][1] for key in val_ids])
    selection = build_validation_threshold_selection(val_probs.reshape(-1), val_targets.reshape(-1))
    test_ids = sorted(test_by_patch)
    test_probs = np.stack([sigmoid_probabilities(test_by_patch[key][0]) for key in test_ids])
    test_targets = np.stack([test_by_patch[key][1] for key in test_ids])
    write_prediction_archive(
        output / "predictions_validation.npz",
        patch_ids=val_ids,
        probability_maps=val_probs,
        target_maps=val_targets,
    )
    write_prediction_archive(
        output / "predictions_test.npz",
        patch_ids=test_ids,
        probability_maps=test_probs,
        target_maps=test_targets,
    )
    threshold = float(selection["selected_threshold"])
    metrics = compute_registered_metrics(
        test_probs.reshape(-1), test_targets.reshape(-1), threshold
    )
    torch.save(model.state_dict(), output / "final_probe.pt")
    _write_json(
        output / "metrics.json",
        {
            **metrics,
            "average_precision": float(metrics["ap"]),
            "protocol_id": matrix["protocol_id"],
            "family": family["id"],
            "task": job["task"],
            "fold": job["fold"],
            "shot": job["shot"],
            "seed": job["seed"],
            "probe_seed": probe_seed,
            "probe": {"head": "conv3x3_64_128_64", **FROZEN_PROBE},
            "threshold": threshold,
            "validation_threshold_selection": selection,
            "matrix_input_lock": str(lock_path.resolve()),
            "matrix_input_lock_sha256": sha256_file(lock_path),
            "shot_schedule": str(schedule_path.resolve()),
            "shot_schedule_sha256": sha256_file(schedule_path),
            "shot_schedule_record": schedule,
            "normalization": {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()},
            "train_seconds": time.perf_counter() - start,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "per_patch_confusion": {
                patch_id: _confusion(test_targets[index], test_probs[index], threshold)
                for index, patch_id in enumerate(test_ids)
            },
            "per_patch_target_support_sha256": {
                patch_id: target_support_sha256(test_targets[index])
                for index, patch_id in enumerate(test_ids)
            },
        },
    )
    print(f"[done] {job}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--devices", nargs="+", default=["npu:0", "npu:1", "npu:2", "npu:3", "npu:4", "npu:5"]
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Verify all three sealed exports and freeze schedules without starting NPU jobs.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.device is None:
            raise ValueError("worker requires --device")
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        lock_path = args.output_root / "matrix_input_lock.json"
        if not lock_path.is_file():
            raise FileNotFoundError("matrix worker requires the parent-created input lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("matrix_sha256") != sha256_file(args.matrix):
            raise ValueError("matrix worker input lock does not match its matrix")
        jobs = build_jobs(matrix)
        for index, job in enumerate(jobs):
            if index % args.worker_count == args.worker_index:
                run_job(matrix, args.output_root, lock_path, job, args.device)
        return
    matrix, lock_path = prepare_matrix(args.matrix, args.output_root)
    jobs = build_jobs(matrix)
    if args.prepare_only:
        print(f"[prepared] {len(jobs)} strict matrix cells; input lock: {lock_path}", flush=True)
        return
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--matrix",
            str(args.matrix),
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
        raise RuntimeError(f"strict matrix worker failures: {exit_codes}")


if __name__ == "__main__":
    main()
