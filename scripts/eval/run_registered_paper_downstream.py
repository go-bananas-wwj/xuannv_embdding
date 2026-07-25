#!/usr/bin/env python3
"""Fail-closed helpers for registered paper downstream probes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.eval.run_strong_downstream_benchmark import (
    PatchTensor,
    compute_metrics,
    load_manifest,
    load_patch_list,
    make_device,
    make_model,
    pos_weight_for,
    task_spec,
    train_epoch,
)

FROZEN_SPLIT = Path("configs/eval/haidian_spatial_5fold_buffer1_seed42.json")
FROZEN_SPLIT_SHA256 = "57989b0f1e6944f94e74721de7f56fae5040ef7622550cf6fb4160acd080ba4f"
FROZEN_EVAL_MANIFEST_SHA256 = "9bd55c663616322c13804b7c8c0d2e32860ca1fda1d2e5d59dfca8404b901ab3"
FROZEN_PROBE = {"epochs": 80, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4}


def sha256_file(path: Path) -> str:
    """Return the content hash used by result provenance records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_embedding_file_index(embedding_root: Path, region: str, month: str) -> dict[str, Any]:
    """Hash every exported map consumed by one registered probe."""
    embedding_root = embedding_root.resolve()
    files = sorted((embedding_root / region).glob(f"*/{month}_embedding_map.pt"))
    if not files:
        raise FileNotFoundError(f"No {month} embedding maps under {embedding_root / region}")
    entries = [
        {
            "path": str(path.relative_to(embedding_root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    payload = {"schema_version": 1, "region": region, "month": str(month), "files": entries}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {**payload, "index_sha256": hashlib.sha256(encoded).hexdigest()}


def verify_embedding_file_index(
    embedding_root: Path,
    index: dict[str, Any],
    region: str,
    month: str,
    expected_patch_ids: set[str] | None = None,
) -> None:
    """Fail closed when an exported embedding map differs from its frozen index."""
    if index.get("region") != region or str(index.get("month")) != str(month):
        raise ValueError("Embedding index region or month does not match this evaluation")
    expected = index.get("files")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Embedding index has no file entries")
    actual = build_embedding_file_index(embedding_root, region, month)
    expected_paths = [entry.get("path") for entry in expected]
    actual_paths = [entry["path"] for entry in actual["files"]]
    if expected_paths != actual_paths:
        raise ValueError("Embedding index file set differs from exported embedding maps")
    if expected_patch_ids is not None:
        actual_patch_ids = {Path(path).parent.name for path in actual_paths}
        if actual_patch_ids != expected_patch_ids:
            raise ValueError("Embedding index patch IDs differ from the frozen manifest")
    for stored, observed in zip(expected, actual["files"], strict=True):
        if stored.get("sha256") != observed["sha256"]:
            raise ValueError(f"Embedding map hash mismatch: {observed['path']}")
        if stored.get("bytes") != observed["bytes"]:
            raise ValueError(f"Embedding map byte size mismatch: {observed['path']}")


def load_sealed_embedding_file_index(
    embedding_root: Path,
    export_meta: dict[str, Any],
    region: str,
    month: str,
    expected_patch_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Load an index whose content hash is sealed into the export metadata."""
    index_path = embedding_root / "embedding_file_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing embedding file index: {index_path}")
    if export_meta.get("embedding_file_index_sha256") != sha256_file(index_path):
        raise ValueError("Embedding file index does not match export metadata seal")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    description = export_meta.get("embedding_file_index")
    if not isinstance(description, dict):
        raise ValueError("Embedding export metadata lacks an index description")
    if description.get("path") != index_path.name:
        raise ValueError("Embedding metadata index path is inconsistent")
    if description.get("region") != region or str(description.get("month")) != str(month):
        raise ValueError("Embedding metadata region or month is inconsistent")
    if description.get("file_count") != len(index.get("files", [])):
        raise ValueError("Embedding metadata file_count is inconsistent")
    expected_index = index.get("index_sha256")
    actual_index = build_embedding_file_index(embedding_root, region, month).get("index_sha256")
    if expected_index != actual_index:
        raise ValueError("Embedding file index content hash is invalid")
    verify_embedding_file_index(embedding_root, index, region, month, expected_patch_ids)
    return index


def _patch_id_set_sha256(patch_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(patch_ids)).encode("utf-8")).hexdigest()


def _load_export_shard_records(
    embedding_root: Path,
    region: str,
    month: str,
    expected_patch_ids: set[str],
    shard_commands: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate each completed export shard before a whole-export provenance seal."""
    region_root = embedding_root / region
    expected_shards = set(shard_commands)
    records: list[dict[str, Any]] = []
    seen_patch_ids: set[str] = set()
    for shard_id in sorted(expected_shards):
        path = region_root / f"produced_patch_ids_shard_{shard_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing produced patch list for shard {shard_id}: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError(f"Shard {shard_id} patch list must be a JSON string list")
        patch_ids = set(raw)
        if len(patch_ids) != len(raw):
            raise ValueError(f"Shard {shard_id} patch list contains duplicate patch IDs")
        overlap = seen_patch_ids & patch_ids
        if overlap:
            raise ValueError(f"Shard patch union overlaps at {sorted(overlap)[:3]}")
        missing_maps = [
            patch_id
            for patch_id in sorted(patch_ids)
            if not (region_root / patch_id / f"{month}_embedding_map.pt").is_file()
        ]
        if missing_maps:
            raise ValueError(f"Shard {shard_id} is missing embedding maps: {missing_maps[:3]}")
        seen_patch_ids.update(patch_ids)
        records.append(
            {
                "shard_id": shard_id,
                "command": shard_commands[shard_id],
                "produced_patch_ids_path": str(path.relative_to(embedding_root)),
                "produced_patch_ids_sha256": sha256_file(path),
                "patch_count": len(patch_ids),
                "patch_ids_sha256": _patch_id_set_sha256(patch_ids),
            }
        )
    if seen_patch_ids != expected_patch_ids:
        raise ValueError("Shard patch union does not match the expected export patch IDs")
    map_patch_ids = {path.parent.name for path in region_root.glob(f"*/{month}_embedding_map.pt")}
    if map_patch_ids != expected_patch_ids:
        raise ValueError("Exported embedding map set does not match the expected patch IDs")
    return records, {
        "expected_patch_count": len(expected_patch_ids),
        "expected_patch_ids_sha256": _patch_id_set_sha256(expected_patch_ids),
        "map_index_sha256": build_embedding_file_index(embedding_root, region, month)[
            "index_sha256"
        ],
    }


def _read_shard_patch_ids(region_root: Path, shard_id: int) -> set[str]:
    path = region_root / f"produced_patch_ids_shard_{shard_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"Shard {shard_id} patch list must be a JSON string list")
    patch_ids = set(raw)
    if len(patch_ids) != len(raw):
        raise ValueError(f"Shard {shard_id} patch list contains duplicate patch IDs")
    return patch_ids


def _verify_imported_shards(
    embedding_root: Path,
    target_meta: dict[str, Any],
    region: str,
    month: str,
    shard_records: list[dict[str, Any]],
    imported_shards: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Verify a copied shard against its immutable source rather than trusting a note."""
    required = {
        "source_embedding_root",
        "source_meta_sha256",
        "source_patch_list_sha256",
        "source_embedding_map_index_sha256",
        "source_map_count",
        "copy_reason",
        "compatibility",
    }
    record_by_shard = {int(record["shard_id"]): record for record in shard_records}
    if set(imported_shards) - set(record_by_shard):
        raise ValueError("Import records refer to unknown export shards")
    if not imported_shards:
        return {}
    config_path = Path(str(target_meta.get("config_path", "")))
    if not config_path.is_file():
        raise FileNotFoundError(f"Target export configuration is unavailable: {config_path}")
    target_expected = {
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": target_meta.get("checkpoint_sha256"),
        "manifest_sha256": target_meta.get("manifest", {}).get("manifest_sha256"),
    }
    normalized: dict[str, dict[str, Any]] = {}
    for shard_id, entry in sorted(imported_shards.items()):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(
                f"Imported shard {shard_id} does not match the required provenance schema"
            )
        if not isinstance(entry["copy_reason"], str) or not entry["copy_reason"].strip():
            raise ValueError(f"Imported shard {shard_id} must include a nonempty copy_reason")
        source_root = Path(str(entry["source_embedding_root"])).resolve()
        source_meta_path = source_root / "meta.json"
        source_region = source_root / region
        source_list_path = source_region / f"produced_patch_ids_shard_{shard_id}.json"
        if not source_meta_path.is_file() or not source_list_path.is_file():
            raise FileNotFoundError(f"Imported shard {shard_id} source provenance is unavailable")
        if sha256_file(source_meta_path) != entry["source_meta_sha256"]:
            raise ValueError(f"Imported shard {shard_id} source metadata hash differs")
        if sha256_file(source_list_path) != entry["source_patch_list_sha256"]:
            raise ValueError(f"Imported shard {shard_id} source patch list hash differs")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        compatibility = entry["compatibility"]
        expected_compatibility = {
            **target_expected,
            "region": region,
            "month": str(month),
            "num_shards": len(record_by_shard),
            "shard_id": shard_id,
        }
        if compatibility != expected_compatibility:
            raise ValueError(f"Imported shard {shard_id} compatibility fields differ")
        source_config = Path(str(source_meta.get("config_path", "")))
        if (
            not source_config.is_file()
            or sha256_file(source_config) != compatibility["config_sha256"]
        ):
            raise ValueError(f"Imported shard {shard_id} source configuration differs")
        if source_meta.get("checkpoint_sha256") != compatibility["checkpoint_sha256"]:
            raise ValueError(f"Imported shard {shard_id} source checkpoint differs")
        if (
            source_meta.get("manifest", {}).get("manifest_sha256")
            != compatibility["manifest_sha256"]
        ):
            raise ValueError(f"Imported shard {shard_id} source manifest differs")
        source_ids = _read_shard_patch_ids(source_region, shard_id)
        target_ids = _read_shard_patch_ids(embedding_root / region, shard_id)
        if source_ids != target_ids:
            raise ValueError(f"Imported shard {shard_id} source patch IDs differ")
        source_index = build_embedding_file_index(source_root, region, month)
        if len(source_index["files"]) != entry["source_map_count"]:
            raise ValueError(f"Imported shard {shard_id} source map count differs")
        if source_index["index_sha256"] != entry["source_embedding_map_index_sha256"]:
            raise ValueError(f"Imported shard {shard_id} source embedding map index differs")
        if {Path(item["path"]).parent.name for item in source_index["files"]} != source_ids:
            raise ValueError(f"Imported shard {shard_id} source map patch IDs differ")
        for item in source_index["files"]:
            target_map = embedding_root / item["path"]
            if not target_map.is_file() or sha256_file(target_map) != item["sha256"]:
                raise ValueError(f"Imported shard {shard_id} target map differs from source")
        normalized[str(shard_id)] = entry
    return normalized


def canonicalize_embedding_export(
    embedding_root: Path,
    region: str,
    month: str,
    *,
    expected_patch_ids: set[str],
    shard_commands: dict[int, str],
    imported_shards: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one canonical, content-bound provenance record after sharded export."""
    embedding_root = embedding_root.resolve()
    meta_path = embedding_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing embedding export metadata: {meta_path}")
    records, validation = _load_export_shard_records(
        embedding_root, region, month, expected_patch_ids, shard_commands
    )
    imports = imported_shards or {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    normalized_imports = _verify_imported_shards(
        embedding_root, meta, region, month, records, imports
    )
    proof = {
        "schema_version": 1,
        "region": region,
        "month": str(month),
        "shard_count": len(shard_commands),
        "shards": records,
        "imported_shards": normalized_imports,
        "validation": validation,
    }
    proof_path = embedding_root / "canonical_export_provenance.json"
    _write_json(proof_path, proof)
    meta["canonical_export_provenance_sha256"] = sha256_file(proof_path)
    meta["canonical_export_provenance"] = {
        "path": proof_path.name,
        "schema_version": proof["schema_version"],
        "shard_count": proof["shard_count"],
    }
    _write_json(meta_path, meta)
    return proof


def verify_canonical_export_provenance(
    embedding_root: Path,
    export_meta: dict[str, Any],
    region: str,
    month: str,
    expected_patch_ids: set[str],
) -> dict[str, Any]:
    """Fail closed unless one whole-export provenance proof matches the shard outputs."""
    description = export_meta.get("canonical_export_provenance")
    if not isinstance(description, dict):
        raise ValueError("Embedding export lacks canonical shard provenance")
    proof_path = embedding_root / str(description.get("path", ""))
    if not proof_path.is_file():
        raise FileNotFoundError(f"Missing canonical export provenance: {proof_path}")
    if export_meta.get("canonical_export_provenance_sha256") != sha256_file(proof_path):
        raise ValueError("Canonical export provenance does not match export metadata seal")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != 1:
        raise ValueError("Unsupported canonical export provenance schema")
    if proof.get("region") != region or str(proof.get("month")) != str(month):
        raise ValueError("Canonical export provenance region or month is inconsistent")
    shards = proof.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Canonical export provenance has no shard records")
    shard_commands = {
        int(record["shard_id"]): str(record["command"])
        for record in shards
        if isinstance(record, dict) and isinstance(record.get("shard_id"), int)
    }
    if len(shard_commands) != len(shards) or len(shard_commands) != proof.get("shard_count"):
        raise ValueError("Canonical export provenance shard records are inconsistent")
    records, validation = _load_export_shard_records(
        embedding_root, region, month, expected_patch_ids, shard_commands
    )
    if records != shards or validation != proof.get("validation"):
        raise ValueError("Canonical export provenance differs from current shard outputs")
    imports = proof.get("imported_shards")
    if not isinstance(imports, dict):
        raise ValueError("Canonical export provenance imported_shards is invalid")
    normalized_imports = _verify_imported_shards(
        embedding_root,
        export_meta,
        region,
        month,
        records,
        {int(key): value for key, value in imports.items() if str(key).isdigit()},
    )
    if normalized_imports != imports:
        raise ValueError("Canonical export provenance imported shard records are inconsistent")
    return proof


def seal_embedding_file_index(embedding_root: Path, region: str, month: str) -> dict[str, Any]:
    """Write a content index and bind it into an already-complete export's metadata."""
    meta_path = embedding_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing embedding export metadata: {meta_path}")
    index = build_embedding_file_index(embedding_root, region, month)
    index_path = embedding_root / "embedding_file_index.json"
    _write_json(index_path, index)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["embedding_file_index_sha256"] = sha256_file(index_path)
    meta["embedding_file_index"] = {
        "path": index_path.name,
        "region": region,
        "month": str(month),
        "file_count": len(index["files"]),
    }
    _write_json(meta_path, meta)
    return index


def prepare_result_output(output: Path) -> None:
    """Atomically claim a fresh output directory for exactly one probe run."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Registered evaluation refusing to overwrite or reuse: {output}"
        ) from exc


def verify_embedding_registry(
    registry_path: Path,
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    index_sha256: str,
    canonical_provenance_sha256: str,
    region: str,
    month: str,
    patch_count: int,
) -> dict[str, Any]:
    """Require an external frozen record for the exact exported feature set."""
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing external embedding registry: {registry_path}")
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = raw.get("exports")
    if raw.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("Invalid external embedding registry schema")
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "embedding_file_index_sha256": index_sha256,
        "canonical_export_provenance_sha256": canonical_provenance_sha256,
        "region": region,
        "month": str(month),
        "patch_count": patch_count,
    }
    matches = [
        entry
        for entry in entries
        if all(entry.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise ValueError("External embedding registry does not contain the sealed export")
    return {
        "path": str(registry_path.resolve()),
        "sha256": sha256_file(registry_path),
        **matches[0],
    }


def verify_git_head_file(path: Path) -> None:
    """Require that an external registry is a clean, Git-tracked HEAD artifact."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("External registry must be inside the repository") from exc
    relative_text = str(relative)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_text],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("External registry must be Git tracked")
    head = subprocess.run(
        ["git", "show", f"HEAD:{relative_text}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or head.stdout != resolved.read_bytes():
        raise ValueError("External registry must exactly match the current Git HEAD")


def verify_encoder_provenance(
    config_path: Path,
    checkpoint_path: Path,
    expected_fold: int,
) -> dict[str, str | int]:
    """Reject an encoder checkpoint if its registered fold differs."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing encoder config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint_path}")
    if checkpoint_path.name != "best.pt":
        raise ValueError(
            "Registered evaluation requires the validation-selected encoder checkpoint "
            "named best.pt"
        )
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = raw.get("data", {})
    configured_fold = data.get("paper_fold")
    if configured_fold is None:
        raise ValueError(f"Encoder config does not declare data.paper_fold: {config_path}")
    if int(configured_fold) != expected_fold:
        raise ValueError(
            "Encoder paper_fold "
            f"{configured_fold} does not match requested evaluation fold {expected_fold}"
        )
    experiment_name = raw.get("experiment", {}).get("name")
    if experiment_name != checkpoint_path.parent.name:
        raise ValueError("Checkpoint parent directory does not match config experiment.name")
    split_path = Path(data.get("paper_spatial_split", ""))
    frozen_path = FROZEN_SPLIT.resolve()
    if split_path.resolve() != frozen_path:
        raise ValueError("Encoder config does not use the frozen registered spatial split")
    if sha256_file(split_path) != FROZEN_SPLIT_SHA256:
        raise ValueError("Encoder split hash differs from frozen registered spatial split")
    split = json.loads(frozen_path.read_text(encoding="utf-8"))["folds"][expected_fold]
    train_manifest = Path(data.get("train_manifest_path", ""))
    val_manifest = Path(data.get("val_manifest_path", ""))
    train_ids = _manifest_ids(train_manifest)
    val_ids = _manifest_ids(val_manifest)
    held_out = set(split["val"]) | set(split["test"]) | set(split.get("buffer", []))
    if not train_ids <= set(split["train"]) or train_ids & held_out:
        raise ValueError("Encoder training manifest overlaps validation, test, or buffer geography")
    if val_ids != set(split["val"]):
        raise ValueError("Encoder validation manifest differs from frozen fold validation set")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer_state = checkpoint.get("trainer_state", {})
    if checkpoint.get("epoch") != trainer_state.get("best_epoch"):
        raise ValueError("best.pt checkpoint epoch does not match recorded best_epoch")
    return {
        "encoder_fold": int(configured_fold),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _manifest_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered manifest: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(entry["patch_id"]) for entry in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate patch IDs in manifest: {path}")
    return set(ids)


def _verify_frozen_split(path: Path, fold: int) -> dict[str, list[str]]:
    if path.resolve() != FROZEN_SPLIT.resolve() or sha256_file(path) != FROZEN_SPLIT_SHA256:
        raise ValueError("Evaluation must use the frozen registered spatial split")
    folds = json.loads(path.read_text(encoding="utf-8"))["folds"]
    if fold < 0 or fold >= len(folds):
        raise ValueError(f"Fold {fold} is outside frozen split")
    result = {key: list(folds[fold].get(key, [])) for key in ("train", "val", "test", "buffer")}
    if set(result["train"]) & (set(result["val"]) | set(result["test"]) | set(result["buffer"])):
        raise ValueError("Frozen split has overlapping train and held-out patch IDs")
    if set(result["val"]) & set(result["test"]):
        raise ValueError("Frozen split has overlapping validation and test patch IDs")
    return result


def _verify_embedding_export(
    embedding_root: Path, manifest_path: Path, checkpoint_sha256: str, month: str
) -> dict[str, Any]:
    if sha256_file(manifest_path) != FROZEN_EVAL_MANIFEST_SHA256:
        raise ValueError("Evaluation must use the registered 320-patch manifest")
    meta_path = embedding_root / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing embedding export provenance: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    export_manifest = meta.get("manifest", {})
    if export_manifest.get("manifest_sha256") != FROZEN_EVAL_MANIFEST_SHA256:
        raise ValueError("Embedding export manifest is not the registered 320-patch manifest")
    if export_manifest.get("patch_count") != 320:
        raise ValueError("Embedding export provenance does not contain 320 patches")
    if meta.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Embedding export checkpoint hash does not match the evaluated encoder")
    manifest_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_patch_ids = {str(record["patch_id"]) for record in manifest_records}
    if len(expected_patch_ids) != 320:
        raise ValueError("Registered evaluation manifest must contain 320 unique patch IDs")
    verify_canonical_export_provenance(embedding_root, meta, "haidian", month, expected_patch_ids)
    load_sealed_embedding_file_index(embedding_root, meta, "haidian", month, expected_patch_ids)
    return meta


def sigmoid_probabilities(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable sigmoid without per-image scaling."""
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


def binary_f1(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """Compute pooled binary F1 at one frozen threshold."""
    predicted = np.asarray(probabilities) >= threshold
    truth = np.asarray(targets).astype(bool)
    true_positive = int(np.logical_and(predicted, truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def select_validation_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Select F1 threshold on the all-binary registered validation grid."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have identical flattened shapes")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Validation targets must be binary for the registered V2 evaluator")
    targets = targets.astype(np.uint8)
    if not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities must be finite")
    if targets.size == 0 or targets.sum() in {0, targets.size}:
        raise ValueError("Validation pixels must contain both positive and negative labels")
    best_threshold = 0.001
    best_f1 = -1.0
    for threshold in np.arange(0.001, 1.0, 0.001):
        score = binary_f1(probabilities, targets, float(threshold))
        if score >= best_f1:
            best_threshold = float(threshold)
            best_f1 = score
    return best_threshold, best_f1


def build_shot_manifest(
    task_name: str,
    train_ids: list[str],
    fold: int,
    seed: int,
    label_sha256: str,
    split_sha256: str,
    pixel_count: Callable[[str], int],
    budgets: tuple[int, ...] = (5, 10, 50),
    min_positive_pixels: int = 64,
) -> dict[str, Any]:
    """Create one immutable, nested positive/negative patch schedule.

    Patch ordering is sampled once at the largest budget.  Smaller shot levels
    are prefixes of that ordering, so all representations consume identical
    examples and the 5/10/50-shot sets are genuinely nested.
    """
    if not budgets or any(type(budget) is not int or budget <= 0 for budget in budgets):
        raise ValueError("Shot budgets must be non-empty positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Shot budgets must be strictly increasing without duplicates")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training patch IDs must be unique for sampling without replacement")
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    for patch_id in sorted(train_ids):
        count = int(pixel_count(patch_id))
        if count >= min_positive_pixels:
            positive_ids.append(patch_id)
        elif count == 0:
            negative_ids.append(patch_id)
    rng = random.Random(seed + fold * 1009)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)
    largest_budget = budgets[-1]
    if len(positive_ids) < largest_budget or len(negative_ids) < largest_budget:
        raise RuntimeError(
            f"Exact {largest_budget}+{largest_budget} shot budget infeasible for {task_name}: "
            f"positive={len(positive_ids)} negative={len(negative_ids)}"
        )
    sets: dict[str, dict[str, list[str]]] = {}
    for budget in budgets:
        selected_pos = positive_ids[:budget]
        selected_neg = negative_ids[:budget]
        combined = selected_pos + selected_neg
        level_rng = random.Random(seed + fold * 1009 + budget)
        level_rng.shuffle(combined)
        sets[str(budget)] = {
            "positive_patch_ids": selected_pos,
            "negative_patch_ids": selected_neg,
            "train_patch_ids": combined,
        }
    return {
        "schema_version": 1,
        "task": task_name,
        "fold": fold,
        "seed": seed,
        "rule": {
            "min_positive_pixels": min_positive_pixels,
            "negative_rule": "exactly_zero_positive_pixels",
            "selection": "deterministic_nested_prefix",
        },
        "label_sha256": label_sha256,
        "split_sha256": split_sha256,
        "eligible_positive_count": len(positive_ids),
        "eligible_negative_count": len(negative_ids),
        "sets": sets,
    }


def build_registered_shot_schedule(
    task_name: str,
    train_ids: list[str],
    fold: int,
    seed: int,
    label_sha256: str,
    split_sha256: str,
    pixel_count: Callable[[str], int],
    budgets: tuple[int, ...] = (5, 10, 50),
    min_positive_pixels: int = 64,
) -> dict[str, Any]:
    """Freeze all feasible exact budgets and explicitly record infeasible cells as NA."""
    if not budgets or any(type(budget) is not int or budget <= 0 for budget in budgets):
        raise ValueError("Shot budgets must be non-empty positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Shot budgets must be strictly increasing without duplicates")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training patch IDs must be unique for sampling without replacement")
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    for patch_id in sorted(train_ids):
        count = int(pixel_count(patch_id))
        if count >= min_positive_pixels:
            positive_ids.append(patch_id)
        elif count == 0:
            negative_ids.append(patch_id)
    rng = random.Random(seed + fold * 1009)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)
    sets: dict[str, dict[str, list[str]]] = {}
    unavailable: dict[str, dict[str, int | str]] = {}
    for budget in budgets:
        if len(positive_ids) < budget or len(negative_ids) < budget:
            unavailable[str(budget)] = {
                "status": "NA",
                "reason": "insufficient_exact_positive_or_negative_patches",
                "positive_available": len(positive_ids),
                "negative_available": len(negative_ids),
            }
            continue
        selected_pos = positive_ids[:budget]
        selected_neg = negative_ids[:budget]
        combined = selected_pos + selected_neg
        level_rng = random.Random(seed + fold * 1009 + budget)
        level_rng.shuffle(combined)
        sets[str(budget)] = {
            "positive_patch_ids": selected_pos,
            "negative_patch_ids": selected_neg,
            "train_patch_ids": combined,
        }
    rule = {
        "min_positive_pixels": min_positive_pixels,
        "negative_rule": "exactly_zero_positive_pixels",
        "selection": "deterministic_nested_prefix",
    }
    rule_sha256 = hashlib.sha256(
        json.dumps(rule, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "task": task_name,
        "fold": fold,
        "seed": seed,
        "rule": rule,
        "rule_sha256": rule_sha256,
        "label_sha256": label_sha256,
        "split_sha256": split_sha256,
        "eligible_positive_count": len(positive_ids),
        "eligible_negative_count": len(negative_ids),
        "sets": sets,
        "unavailable": unavailable,
    }


def build_registered_mixed_shot_schedule(
    task_name: str,
    train_ids: list[str],
    fold: int,
    seed: int,
    label_sha256: str,
    split_sha256: str,
    pixel_counts: Callable[[str], tuple[int, int]],
    budgets: tuple[int, ...] = (5, 10, 50),
    min_positive_pixels: int = 64,
    min_background_pixels: int = 64,
) -> dict[str, Any]:
    """Freeze a nested schedule of pixelwise mixed-class support patches.

    A dense segmentation support image supplies both classes when it contains
    sufficient foreground and background pixels.  This V2 rule avoids the
    inappropriate requirement that a road-support image be entirely road-free.
    """
    if not budgets or any(type(budget) is not int or budget <= 0 for budget in budgets):
        raise ValueError("Shot budgets must be non-empty positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Shot budgets must be strictly increasing without duplicates")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training patch IDs must be unique for sampling without replacement")

    eligible: list[tuple[str, int, int]] = []
    for patch_id in sorted(train_ids):
        positive_pixels, background_pixels = pixel_counts(patch_id)
        if positive_pixels < 0 or background_pixels < 0:
            raise ValueError(f"Negative pixel count for {task_name}/{patch_id}")
        if positive_pixels >= min_positive_pixels and background_pixels >= min_background_pixels:
            eligible.append((patch_id, int(positive_pixels), int(background_pixels)))

    rng = random.Random(seed + fold * 1009)
    rng.shuffle(eligible)
    sets: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, dict[str, int | str]] = {}
    for budget in budgets:
        if len(eligible) < budget:
            unavailable[str(budget)] = {
                "status": "NA",
                "reason": "insufficient_mixed_class_support_patches",
                "eligible_available": len(eligible),
            }
            continue
        selected = eligible[:budget]
        sets[str(budget)] = {
            "train_patch_ids": [patch_id for patch_id, _, _ in selected],
            "support_pixel_counts": [
                {
                    "patch_id": patch_id,
                    "positive_pixels": positive_pixels,
                    "background_pixels": background_pixels,
                }
                for patch_id, positive_pixels, background_pixels in selected
            ],
        }
    rule = {
        "support_unit": "mixed_class_labeled_patch",
        "min_positive_pixels": min_positive_pixels,
        "min_background_pixels": min_background_pixels,
        "selection": "deterministic_nested_prefix",
    }
    rule_sha256 = hashlib.sha256(
        json.dumps(rule, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 3,
        "task": task_name,
        "fold": fold,
        "seed": seed,
        "rule": rule,
        "rule_sha256": rule_sha256,
        "label_sha256": label_sha256,
        "split_sha256": split_sha256,
        "eligible_mixed_patch_count": len(eligible),
        "sets": sets,
        "unavailable": unavailable,
    }


def load_registered_shot_ids(
    path: Path, *, expected_schedule: dict[str, Any], shot: str
) -> tuple[list[str], dict[str, Any]]:
    """Load a shared schedule only when it exactly matches the registered selection rule."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared shot schedule: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded != expected_schedule:
        raise ValueError("Shared shot schedule does not match the registered selection rule")
    if shot in loaded.get("unavailable", {}):
        raise RuntimeError(f"Shot {shot} is registered NA: {loaded['unavailable'][shot]}")
    try:
        train_ids = loaded["sets"][shot]["train_patch_ids"]
    except KeyError as exc:
        raise ValueError(f"Shared shot schedule has no exact {shot}-shot set") from exc
    return list(train_ids), loaded


def _label_tree_hash(label_roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in sorted(label_roots):
        for path in sorted((root / "masks").glob("*.tif")):
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _positive_pixel_count(task: Any, patch_id: str) -> int:
    from scripts.eval.run_traditional_ml_benchmark import load_binary_mask

    values = load_binary_mask(task, patch_id)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Non-binary label encoding in {task.name}/{patch_id}")
    return int((values == 1).sum())


def _positive_background_pixel_counts(task: Any, patch_id: str) -> tuple[int, int]:
    """Return valid foreground/background pixel counts for a binary task mask."""
    from scripts.eval.run_traditional_ml_benchmark import load_binary_mask

    values = load_binary_mask(task, patch_id)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Non-binary label encoding in {task.name}/{patch_id}")
    return int((values == 1).sum()), int((values == 0).sum())


def _assert_binary_label_roots(label_roots: list[Path]) -> None:
    for root in label_roots:
        paths = sorted((root / "masks").glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No label masks under {root}")
        for path in paths:
            with rasterio.open(path) as src:
                values = src.read(1)
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"Non-binary or ignore label values found: {path}")


def _confusion(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, int]:
    truth = target.astype(bool)
    pred = probability >= threshold
    return {
        "tp": int(np.logical_and(pred, truth).sum()),
        "fp": int(np.logical_and(pred, ~truth).sum()),
        "fn": int(np.logical_and(~pred, truth).sum()),
        "tn": int(np.logical_and(~pred, ~truth).sum()),
    }


def _append_registry(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result_id = record["result_id"]
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    existing = json.loads(line)
                    if existing.get("result_id") == result_id:
                        if json.dumps(existing, ensure_ascii=False, sort_keys=True) != encoded:
                            raise ValueError(f"Result registry collision: {result_id}")
                        return
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def _normalize_items(
    items: list[PatchTensor], mean: torch.Tensor, std: torch.Tensor
) -> list[PatchTensor]:
    return [PatchTensor(item.patch_id, (item.x - mean) / std, item.y) for item in items]


def _fit_train_standardizer(items: list[PatchTensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("Cannot fit feature normalization without training patches")
    stacked = torch.stack([item.x for item in items])
    mean = stacked.mean(dim=(0, 2, 3), keepdim=False).view(-1, 1, 1)
    std = stacked.std(dim=(0, 2, 3), keepdim=False).clamp_min(1e-6).view(-1, 1, 1)
    return mean, std


def _logits_for_items(
    model: torch.nn.Module,
    items: list[PatchTensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_patch: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            x = torch.stack([item.x for item in batch]).to(device, non_blocking=True)
            output = model(x).detach().cpu().numpy().astype(np.float32)
            for item, patch_logits in zip(batch, output, strict=True):
                logits.append(patch_logits.reshape(-1))
                targets.append(item.y.numpy().reshape(-1))
                per_patch[item.patch_id] = (patch_logits, item.y.numpy())
    return np.concatenate(logits), np.concatenate(targets), per_patch


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def registry_entry_core_sha256(result_id: str, artifact: dict[str, Any]) -> str:
    """Hash the registry fields that can be known before the sidecar hash exists."""
    return _canonical_sha256(
        {
            "result_id": result_id,
            "paper_eligible": artifact["paper_eligible"],
            "admission_status": artifact["admission_status"],
            "metrics_sha256": artifact["metrics_sha256"],
            "metric_provenance_sha256": _canonical_sha256(artifact["metric_provenance"]),
        }
    )


def build_artifact_manifest(
    *,
    metric_payload: dict[str, Any],
    output: Path,
    predictions: Path,
    result_id: str,
    registry_path: Path,
    label_sha256: str,
    patch_count: int,
) -> dict[str, Any]:
    """Bind binary probe artifacts to their metric record and registry identity."""
    artifact = {
        "schema_version": 1,
        "paper_eligible": metric_payload["paper_eligible"],
        "admission_status": metric_payload["admission_status"],
        "result_id": result_id,
        "registry_path": str(registry_path.resolve()),
        "metrics_path": str((output / "metrics.json").resolve()),
        "metrics_sha256": sha256_file(output / "metrics.json"),
        "predictions_sha256": sha256_file(predictions),
        "probe_sha256": sha256_file(output / "final_probe.pt"),
        # The complete metric payload is the authoritative provenance record.
        "metric_provenance": metric_payload,
        "label_sha256": label_sha256,
        "patch_count": patch_count,
    }
    artifact["registry_entry_core_sha256"] = registry_entry_core_sha256(result_id, artifact)
    return artifact


def verify_artifact_registry_binding(artifact_path: Path, registry_path: Path) -> None:
    """Reject a sidecar unless a unique registry record cryptographically binds it."""
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    output = artifact_path.parent
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions_test.npz"
    probe_path = output / "final_probe.pt"
    if artifact.get("metrics_path") != str(metrics_path.resolve()):
        raise ValueError("Artifact metrics path differs from its sidecar directory")
    if artifact.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("Artifact metrics hash does not match metrics.json")
    if artifact.get("predictions_sha256") != sha256_file(predictions_path):
        raise ValueError("Artifact predictions hash does not match predictions_test.npz")
    if artifact.get("probe_sha256") != sha256_file(probe_path):
        raise ValueError("Artifact probe hash does not match final_probe.pt")
    if json.loads(metrics_path.read_text(encoding="utf-8")) != artifact.get("metric_provenance"):
        raise ValueError("Artifact metric provenance does not match metrics.json")
    if artifact.get("registry_path") != str(registry_path.resolve()):
        raise ValueError("Artifact registry path differs from the requested result registry")
    result_id = artifact.get("result_id")
    expected_core = registry_entry_core_sha256(str(result_id), artifact)
    if artifact.get("registry_entry_core_sha256") != expected_core:
        raise ValueError("Artifact registry-entry core hash is invalid")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Missing result registry: {registry_path}")
    matches = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("result_id") == result_id
    ]
    if len(matches) != 1:
        raise ValueError("Artifact must bind exactly one registry entry")
    record = matches[0]
    if record.get("artifact_sha256") != sha256_file(artifact_path):
        raise ValueError("Registry artifact hash does not match the sidecar")
    if record.get("registry_entry_core_sha256") != expected_core:
        raise ValueError("Registry entry core hash does not match the sidecar")
    sidecar_fields = {key: record.get(key) for key in artifact}
    if sidecar_fields != artifact:
        raise ValueError("Registry entry fields do not match the sidecar")
    record_without_hash = dict(record)
    entry_sha256 = record_without_hash.pop("registry_entry_sha256", None)
    if entry_sha256 != _canonical_sha256(record_without_hash):
        raise ValueError("Registry entry hash does not match its content")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one registered Xuannv Conv3x3 probe.")
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--embedding-registry",
        type=Path,
        required=True,
        help="Git-tracked external registry sealing the embedding file index.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--spatial-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--task", choices=["building", "road", "water"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument("--shot", choices=["5", "10", "50", "full"], required=True)
    parser.add_argument(
        "--shot-manifest",
        type=Path,
        default=None,
        help="Shared frozen schedule for a task/fold/seed; required unless --shot full.",
    )
    parser.add_argument("--shot-seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = verify_encoder_provenance(args.encoder_config, args.encoder_checkpoint, args.fold)
    split = _verify_frozen_split(args.spatial_split, args.fold)
    export_meta = _verify_embedding_export(
        args.embedding_root, args.manifest, str(provenance["checkpoint_sha256"]), args.month
    )
    verify_git_head_file(args.embedding_registry)
    embedding_registry = verify_embedding_registry(
        args.embedding_registry,
        checkpoint_sha256=str(provenance["checkpoint_sha256"]),
        config_sha256=str(provenance["config_sha256"]),
        manifest_sha256=FROZEN_EVAL_MANIFEST_SHA256,
        index_sha256=str(export_meta["embedding_file_index_sha256"]),
        canonical_provenance_sha256=str(export_meta["canonical_export_provenance_sha256"]),
        region="haidian",
        month=args.month,
        patch_count=320,
    )
    probe_seed = int.from_bytes(
        hashlib.sha256(f"{args.fold}|{args.task}|{args.shot_seed}".encode()).digest()[:4],
        "little",
    )
    _set_seed(probe_seed)
    device = make_device(args.device)
    split_sha = sha256_file(args.spatial_split)
    records = load_manifest(args.manifest, args.manifest.parent)
    task = task_spec(args.task, args.label_root)
    _assert_binary_label_roots(task.label_roots)
    if args.shot == "full":
        train_ids = split["train"]
        shot_manifest: dict[str, Any] | None = None
        shot_manifest_path: Path | None = None
    else:
        if args.shot_manifest is None:
            raise ValueError("--shot-manifest is required for registered few-shot evaluation")
        label_hash = _label_tree_hash(task.label_roots)
        expected = build_registered_mixed_shot_schedule(
            args.task,
            split["train"],
            args.fold,
            args.shot_seed,
            label_hash,
            split_sha,
            lambda patch_id: _positive_background_pixel_counts(task, patch_id),
            budgets=(5, 10, 50),
        )
        train_ids, shot_manifest = load_registered_shot_ids(
            args.shot_manifest, expected_schedule=expected, shot=args.shot
        )
        shot_manifest_path = args.shot_manifest
    train_items = load_patch_list(
        records,
        task,
        train_ids,
        "xuannv_embedding",
        args.embedding_root,
        "haidian",
        args.month,
    )
    val_items = load_patch_list(
        records, task, split["val"], "xuannv_embedding", args.embedding_root, "haidian", args.month
    )
    test_items = load_patch_list(
        records, task, split["test"], "xuannv_embedding", args.embedding_root, "haidian", args.month
    )
    mean, std = _fit_train_standardizer(train_items)
    train_items = _normalize_items(train_items, mean, std)
    val_items = _normalize_items(val_items, mean, std)
    test_items = _normalize_items(test_items, mean, std)
    prepare_result_output(args.output_root)
    model = make_model("conv3x3", 64, 64).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_PROBE["lr"],
        weight_decay=FROZEN_PROBE["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FROZEN_PROBE["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(train_items, device))
    start_time = time.perf_counter()
    for _ in range(FROZEN_PROBE["epochs"]):
        train_epoch(model, train_items, device, optimizer, loss_fn, FROZEN_PROBE["batch_size"])
        scheduler.step()
    val_logits, val_targets, _ = _logits_for_items(
        model, val_items, device, FROZEN_PROBE["batch_size"]
    )
    threshold, val_f1 = select_validation_threshold(sigmoid_probabilities(val_logits), val_targets)
    test_logits, test_targets, test_by_patch = _logits_for_items(
        model, test_items, device, FROZEN_PROBE["batch_size"]
    )
    metrics = compute_metrics(test_logits, test_targets, threshold=threshold)
    metrics["oracle_test_f1"] = metrics.pop("f1_best")
    metrics["oracle_test_threshold"] = metrics.pop("best_threshold")
    metrics["val_f1_at_selected_threshold"] = val_f1
    metrics["average_precision"] = float(
        average_precision_score(test_targets, sigmoid_probabilities(test_logits))
    )
    metrics["roc_auc"] = float(roc_auc_score(test_targets, sigmoid_probabilities(test_logits)))
    output = args.output_root
    predictions = output / "predictions_test.npz"
    ordered_patch_ids = sorted(test_by_patch)
    probability_maps = np.stack(
        [sigmoid_probabilities(test_by_patch[patch_id][0]) for patch_id in ordered_patch_ids]
    ).astype(np.float32)
    target_maps = np.stack([test_by_patch[patch_id][1] for patch_id in ordered_patch_ids]).astype(
        np.uint8
    )
    valid_maps = np.ones_like(target_maps, dtype=bool)
    np.savez_compressed(
        predictions,
        patch_ids=np.array(ordered_patch_ids, dtype=object),
        probabilities=probability_maps,
        targets=target_maps,
        valid_masks=valid_maps,
    )
    torch.save(model.state_dict(), output / "final_probe.pt")
    per_patch = {
        patch_id: _confusion(target_maps[index], probability_maps[index], threshold)
        for index, patch_id in enumerate(ordered_patch_ids)
    }
    metric_payload = {
        **metrics,
        "paper_eligible": False,
        "admission_status": "registered_preliminary_pending_external_gates",
        "task": args.task,
        "fold": args.fold,
        "shot": args.shot,
        "shot_seed": args.shot_seed,
        "probe_seed": probe_seed,
        "probe": {"head": "conv3x3_64_128_64", **FROZEN_PROBE, "final_epoch_only": True},
        "threshold": threshold,
        "train_seconds": time.perf_counter() - start_time,
        "provenance": provenance,
        "spatial_split_sha256": split_sha,
        "manifest_sha256": sha256_file(args.manifest),
        "embedding_export": export_meta,
        "embedding_registry": embedding_registry,
        "normalization": {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()},
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "prediction_file": str(predictions),
        "shot_manifest": shot_manifest,
        "shot_manifest_path": str(shot_manifest_path) if shot_manifest_path else None,
        "shot_manifest_sha256": sha256_file(shot_manifest_path) if shot_manifest_path else None,
        "per_patch_confusion": per_patch,
    }
    _write_json(output / "metrics.json", metric_payload)
    result_id = hashlib.sha256(
        f"{provenance['checkpoint_sha256']}|{args.task}|{args.fold}|{args.shot}|{args.shot_seed}".encode()
    ).hexdigest()
    artifact = build_artifact_manifest(
        metric_payload=metric_payload,
        output=output,
        predictions=predictions,
        result_id=result_id,
        registry_path=args.registry,
        label_sha256=_label_tree_hash(task.label_roots),
        patch_count=len(ordered_patch_ids),
    )
    _write_json(output / "artifact_manifest.json", artifact)
    registry_record = {
        "result_id": result_id,
        "artifact_sha256": sha256_file(output / "artifact_manifest.json"),
        "registry_entry_core_sha256": artifact["registry_entry_core_sha256"],
        **artifact,
    }
    registry_record["registry_entry_sha256"] = _canonical_sha256(registry_record)
    _append_registry(args.registry, registry_record)
    verify_artifact_registry_binding(output / "artifact_manifest.json", args.registry)


if __name__ == "__main__":
    main()
