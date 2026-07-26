#!/usr/bin/env python3
"""Run one fail-closed AEF annual-2025 contextual downstream probe."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.eval.export_aef_v5_embeddings import build_output_index, build_validity_index
from scripts.eval.registered_aef_contextual import (
    load_contextual_matrix,
    sha256_file,
    verify_frozen_shot_schedules,
    verify_runtime_source_hashes,
)
from scripts.eval.run_registered_paper_downstream import (
    FROZEN_PROBE,
    _canonical_sha256,
    _confusion,
    _fit_train_standardizer,
    _label_tree_hash,
    _logits_for_items,
    _normalize_items,
    _positive_background_pixel_counts,
    _set_seed,
    _write_json,
    build_artifact_manifest,
    build_registered_mixed_shot_schedule,
    build_validation_threshold_selection,
    compute_registered_metrics,
    load_registered_shot_ids,
    prepare_result_output,
    sigmoid_probabilities,
    verify_artifact_registry_binding,
    write_prediction_archive,
)
from scripts.eval.run_strong_downstream_benchmark import (
    load_manifest,
    load_patch_list,
    make_device,
    make_model,
    pos_weight_for,
    task_spec,
    train_epoch,
)


def verify_aef_embedding_export(
    embedding_root: Path, matrix: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    """Verify that the label-free annual AEF export matches the contextual matrix."""
    meta_path = embedding_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing AEF export metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("protocol_id") != "aef_annual_2025_contextual":
        raise ValueError("AEF embedding export has an invalid protocol")
    if meta.get("scientific_status") != "contextual_comparator_preliminary":
        raise ValueError("AEF embedding export has an invalid scientific status")
    if meta.get("output_identifier") != matrix.get("baseline_output_identifier"):
        raise ValueError("AEF embedding export output identifier does not match the matrix")
    if meta.get("global_label_free") is not True:
        raise ValueError("AEF contextual export must remain label-free")
    for key in ("source_manifest", "embedding_file_index", "validity_index"):
        record = meta.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"AEF export metadata lacks {key}")
        path = embedding_root / record["path"]
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"AEF export {key} does not match its metadata hash")
    repo_root = Path(__file__).resolve().parents[2]
    official_inputs = matrix.get("aef_official_inputs")
    if not isinstance(official_inputs, dict):
        raise ValueError("AEF contextual matrix lacks official input locks")
    coverage_lock = official_inputs.get("coverage_inventory")
    cog_lock = official_inputs.get("cog_lock")
    if not isinstance(coverage_lock, dict) or not isinstance(cog_lock, dict):
        raise ValueError("AEF contextual matrix has invalid official input locks")
    coverage = meta.get("coverage_inventory")
    if not isinstance(coverage, dict) or not isinstance(coverage.get("path"), str):
        raise ValueError("AEF export metadata lacks coverage inventory")
    coverage_path = Path(coverage["path"])
    expected_coverage_path = repo_root / str(coverage_lock.get("path"))
    if (
        coverage_path.resolve() != expected_coverage_path.resolve()
        or not coverage_path.is_file()
        or sha256_file(coverage_path) != coverage_lock.get("sha256")
        or coverage.get("sha256") != coverage_lock.get("sha256")
    ):
        raise ValueError("AEF export coverage inventory does not match metadata")
    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage_payload.get("v5_manifest_sha256") != matrix.get("manifest_sha256"):
        raise ValueError("AEF coverage inventory does not match the comparison manifest")
    expected_patch_ids = {
        str(record["patch_id"])
        for record in coverage_payload.get("records", [])
        if isinstance(record, dict) and "patch_id" in record
    }
    if len(expected_patch_ids) != 320:
        raise ValueError("AEF coverage inventory must bind exactly 320 patch IDs")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, list):
        manifest_records = manifest_payload
    elif isinstance(manifest_payload, dict):
        manifest_records = manifest_payload.get("records", [])
        if not manifest_records and isinstance(manifest_payload.get("patch_metadata"), str):
            metadata_path = Path(__file__).resolve().parents[2] / manifest_payload["patch_metadata"]
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest_records = (
                metadata_payload
                if isinstance(metadata_payload, list)
                else (
                    metadata_payload.get("records", [])
                    if isinstance(metadata_payload, dict)
                    else []
                )
            )
    else:
        manifest_records = []
    manifest_patch_ids = {
        str(record["patch_id"])
        for record in manifest_records
        if isinstance(record, dict) and "patch_id" in record
    }
    if manifest_patch_ids != expected_patch_ids:
        raise ValueError("AEF coverage inventory patch IDs differ from the probe manifest")
    meta_cog_lock = meta.get("cog_lock")
    expected_cog_path = repo_root / str(cog_lock.get("path"))
    if (
        not isinstance(meta_cog_lock, dict)
        or Path(str(meta_cog_lock.get("path", ""))).resolve() != expected_cog_path.resolve()
        or not expected_cog_path.is_file()
        or sha256_file(expected_cog_path) != cog_lock.get("sha256")
        or meta_cog_lock.get("sha256") != cog_lock.get("sha256")
    ):
        raise ValueError("AEF export COG lock does not match the contextual matrix")
    source_manifest_path = embedding_root / meta["source_manifest"]["path"]
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("release") != "aef/v1/annual"
        or source_manifest.get("year") != 2025
        or source_manifest.get("global_label_free") is not True
        or {str(record.get("patch_id")) for record in source_manifest.get("records", [])}
        != expected_patch_ids
    ):
        raise ValueError("AEF source manifest does not bind the official 2025 320-patch export")
    cog_payload = json.loads(expected_cog_path.read_text(encoding="utf-8"))
    locked_assets = {
        str(item.get("filename")): str(item.get("sha256"))
        for item in cog_payload.get("assets", [])
        if isinstance(item, dict)
    }
    snapshot = source_manifest.get("index_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("uri") != cog_payload.get("index_uri")
        or snapshot.get("sha256") != cog_payload.get("index_sha256")
    ):
        raise ValueError("AEF source manifest index snapshot does not match the official lock")
    for source_record in source_manifest["records"]:
        assets = source_record.get("source_assets") if isinstance(source_record, dict) else None
        if not isinstance(assets, list) or not assets:
            raise ValueError("AEF source manifest has no locked source assets")
        for asset in assets:
            filename = Path(str(asset.get("uri", ""))).name if isinstance(asset, dict) else ""
            if not isinstance(asset, dict) or locked_assets.get(filename) != asset.get("sha256"):
                raise ValueError("AEF source asset is outside the official COG lock")
    output_index_path = embedding_root / meta["embedding_file_index"]["path"]
    validity_index_path = embedding_root / meta["validity_index"]["path"]
    stored_output_index = json.loads(output_index_path.read_text(encoding="utf-8"))
    stored_validity_index = json.loads(validity_index_path.read_text(encoding="utf-8"))
    if build_output_index(embedding_root, "haidian", expected_patch_ids) != stored_output_index:
        raise ValueError("AEF embedding maps do not match the sealed output index")
    if build_validity_index(embedding_root, "haidian", expected_patch_ids) != stored_validity_index:
        raise ValueError("AEF validity masks do not match the sealed validity index")
    return meta


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run every provenance gate before loading a model, embedding, or label."""
    matrix = load_contextual_matrix(args.matrix)
    if args.task not in matrix["tasks"] or args.shot not in matrix["shots"]:
        raise ValueError("AEF probe cell is outside the registered matrix")
    if args.fold not in matrix["folds"] or args.shot_seed not in matrix["seeds"]:
        raise ValueError("AEF probe fold or seed is outside the registered matrix")
    if args.spatial_split != Path(matrix["spatial_split"]):
        raise ValueError("AEF probe split path does not match the registered matrix")
    if sha256_file(args.spatial_split) != matrix["spatial_split_sha256"]:
        raise ValueError("AEF probe split hash does not match the registered matrix")
    if sha256_file(args.manifest) != matrix["manifest_sha256"]:
        raise ValueError("AEF probe manifest hash does not match the registered matrix")
    if args.label_root != Path(matrix["label_root"]):
        raise ValueError("AEF probe label root does not match the registered matrix")
    statistics_path = Path(__file__).resolve().parents[2] / matrix["statistics_registry_path"]
    if (
        not statistics_path.is_file()
        or sha256_file(statistics_path) != matrix["statistics_registry_sha256"]
    ):
        raise ValueError("AEF probe statistics registry does not match the registered matrix")
    verify_frozen_shot_schedules(matrix)
    verify_runtime_source_hashes(matrix, Path(__file__).resolve().parents[2])
    verify_aef_embedding_export(args.embedding_root, matrix, args.manifest)
    return matrix


def append_verified_registry_record(
    registry: Path, record: dict[str, Any], artifact_path: Path
) -> None:
    """Append and verify under one exclusive lock, restoring only this failed write."""
    registry.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry.parent / f".{registry.name}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        original = registry.read_bytes() if registry.exists() else b""
        try:
            with registry.open("ab") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True).encode() + b"\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            verify_artifact_registry_binding(artifact_path, registry)
        except Exception:
            registry.write_bytes(original)
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/eval/rse_aef_annual_2025_contextual_comparison_matrix.json"),
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spatial-split", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--task", choices=["building", "road", "water"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--shot", choices=["5", "10"], required=True)
    parser.add_argument("--shot-seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = preflight(args)
    split = json.loads(args.spatial_split.read_text(encoding="utf-8"))["folds"][args.fold]
    records = load_manifest(args.manifest, args.manifest.parent)
    task = task_spec(args.task, args.label_root)
    label_sha256 = _label_tree_hash(task.label_roots)
    if label_sha256 != matrix["labels"][args.task]["tree_sha256"]:
        raise ValueError("AEF probe labels do not match the registered matrix")
    schedule_path = Path(matrix["shot_schedule"]["root"]) / (
        f"{args.task}_fold{args.fold}_seed{args.shot_seed}.json"
    )
    expected_schedule = build_registered_mixed_shot_schedule(
        args.task,
        list(split["train"]),
        args.fold,
        args.shot_seed,
        label_sha256,
        matrix["spatial_split_sha256"],
        lambda patch_id: _positive_background_pixel_counts(task, patch_id),
        budgets=(5, 10, 50),
        protocol_id="v5_osm_assisted",
    )
    train_ids, shot_manifest = load_registered_shot_ids(
        schedule_path,
        expected_schedule=expected_schedule,
        shot=args.shot,
        protocol_id="v5_osm_assisted",
    )
    probe_seed = int.from_bytes(
        hashlib.sha256(f"{args.fold}|{args.task}|{args.shot_seed}".encode()).digest()[:4],
        "little",
    )
    _set_seed(probe_seed)
    train_items = load_patch_list(
        records, task, train_ids, "xuannv_embedding", args.embedding_root, "haidian", "annual_2025"
    )
    val_items = load_patch_list(
        records,
        task,
        list(split["val"]),
        "xuannv_embedding",
        args.embedding_root,
        "haidian",
        "annual_2025",
    )
    test_items = load_patch_list(
        records,
        task,
        list(split["test"]),
        "xuannv_embedding",
        args.embedding_root,
        "haidian",
        "annual_2025",
    )
    mean, std = _fit_train_standardizer(train_items)
    train_items = _normalize_items(train_items, mean, std)
    val_items = _normalize_items(val_items, mean, std)
    test_items = _normalize_items(test_items, mean, std)
    prepare_result_output(args.output_root)
    device = make_device(args.device)
    model = make_model("conv3x3", 64, 64).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_PROBE["lr"], weight_decay=FROZEN_PROBE["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_PROBE["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_items, device))
    for _ in range(FROZEN_PROBE["epochs"]):
        train_epoch(model, train_items, device, optimizer, loss_fn, FROZEN_PROBE["batch_size"])
        scheduler.step()
    _, _, val_by_patch = _logits_for_items(model, val_items, device, FROZEN_PROBE["batch_size"])
    _, _, test_by_patch = _logits_for_items(model, test_items, device, FROZEN_PROBE["batch_size"])
    val_ids = sorted(val_by_patch)
    val_probs = np.stack([sigmoid_probabilities(val_by_patch[key][0]) for key in val_ids])
    val_targets = np.stack([val_by_patch[key][1] for key in val_ids])
    selection = build_validation_threshold_selection(val_probs.reshape(-1), val_targets.reshape(-1))
    threshold = float(selection["selected_threshold"])
    output = args.output_root
    validation_predictions = output / "predictions_validation.npz"
    write_prediction_archive(
        validation_predictions,
        patch_ids=val_ids,
        probability_maps=val_probs,
        target_maps=val_targets,
    )
    test_ids = sorted(test_by_patch)
    test_probs = np.stack([sigmoid_probabilities(test_by_patch[key][0]) for key in test_ids])
    test_targets = np.stack([test_by_patch[key][1] for key in test_ids])
    predictions = output / "predictions_test.npz"
    write_prediction_archive(
        predictions, patch_ids=test_ids, probability_maps=test_probs, target_maps=test_targets
    )
    metrics = compute_registered_metrics(
        test_probs.reshape(-1), test_targets.reshape(-1), threshold
    )
    metrics.update(
        {
            "average_precision": float(metrics["ap"]),
            "val_f1_at_selected_threshold": float(selection["selected_f1"]),
        }
    )
    torch.save(model.state_dict(), output / "final_probe.pt")
    export_meta = verify_aef_embedding_export(args.embedding_root, matrix, args.manifest)
    payload: dict[str, Any] = {
        **metrics,
        "paper_eligible": False,
        "admission_status": "contextual_aef_pending_paired_analysis",
        "protocol_id": "aef_annual_2025_contextual",
        "baseline_id": matrix["baseline_id"],
        "baseline_output_identifier": matrix["baseline_output_identifier"],
        "evidence_scope": matrix["evidence_scope"],
        "evidence_class": "osm_assisted_spatial_readout",
        "label_independence_status": "contextual_time_inequivalent_not_independent",
        "comparison_matrix_sha256": sha256_file(args.matrix),
        "time_inequivalent_contextual": True,
        "prohibited_claims": matrix["prohibited_claims"],
        "task": args.task,
        "fold": args.fold,
        "shot": args.shot,
        "shot_seed": args.shot_seed,
        "probe_seed": probe_seed,
        "probe": {"head": "conv3x3_64_128_64", **FROZEN_PROBE, "final_epoch_only": True},
        "threshold": threshold,
        "validation_threshold_selection": selection,
        "embedding_export": export_meta,
        "spatial_split_sha256": matrix["spatial_split_sha256"],
        "manifest_sha256": matrix["manifest_sha256"],
        "statistics_registry_sha256": matrix["statistics_registry_sha256"],
        "label_sha256": label_sha256,
        "runtime_sources": matrix["probe"]["reader_sources"],
        "normalization": {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()},
        "shot_manifest": shot_manifest,
        "shot_manifest_path": str(schedule_path.resolve()),
        "shot_manifest_sha256": sha256_file(schedule_path),
        "prediction_file": str(predictions.resolve()),
        "validation_prediction_file": str(validation_predictions.resolve()),
        "validation_prediction_sha256": sha256_file(validation_predictions),
        "test_patch_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "per_patch_confusion": {
            patch_id: _confusion(test_targets[index], test_probs[index], threshold)
            for index, patch_id in enumerate(test_ids)
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    _write_json(output / "metrics.json", payload)
    result_id = hashlib.sha256(
        (
            f"aef_annual_2025_contextual|{export_meta['source_manifest']['sha256']}|"
            f"{export_meta['embedding_file_index']['sha256']}|{args.task}|{args.fold}|"
            f"{args.shot}|{args.shot_seed}"
        ).encode()
    ).hexdigest()
    artifact = build_artifact_manifest(
        metric_payload=payload,
        output=output,
        predictions=predictions,
        validation_predictions=validation_predictions,
        result_id=result_id,
        registry_path=args.registry,
        label_sha256=label_sha256,
        patch_count=len(test_ids),
    )
    _write_json(output / "artifact_manifest.json", artifact)
    record = {
        "result_id": result_id,
        "artifact_sha256": sha256_file(output / "artifact_manifest.json"),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    record["registry_entry_sha256"] = _canonical_sha256(record)
    append_verified_registry_record(args.registry, record, output / "artifact_manifest.json")


if __name__ == "__main__":
    main()
