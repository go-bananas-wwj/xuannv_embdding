#!/usr/bin/env python3
"""Run a P10C-only, high-resolution-availability-masked Harbin diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_FAMILY = "p10c_haidian_frozen_harbin_highres_masked"
PRIMARY_PROTOCOL = "harbin_frozen_transfer_multihead_20260729"
DISABLED_SOURCES = ("highres_optical_haidian", "highres_sar_haidian")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(
    "harbin_frozen_transfer_multihead",
    REPO_ROOT / "scripts/eval/run_harbin_frozen_transfer_multihead.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_masked_provenance(provenance: dict[str, Any]) -> None:
    """Validate the sealed 380-map export before any diagnostic reader is trained."""
    verification = provenance.get("embedding_verification")
    mask = provenance.get("highres_mask_verification")
    if not isinstance(verification, dict) or verification.get("patch_count") != 380:
        raise ValueError("masked export does not seal exactly 380 embedding maps")
    if verification.get("shape") != [64, 128, 128] or not isinstance(
        verification.get("files"), dict
    ):
        raise ValueError("masked export embedding shape or hashes are invalid")
    if len(verification["files"]) != 380:
        raise ValueError("masked export does not record every embedding hash")
    if not isinstance(mask, dict) or mask.get("all_mask_sums_zero") is not True:
        raise ValueError("masked export must record all_mask_sums_zero=true")
    if tuple(mask.get("disabled_highres_sources", ())) != DISABLED_SOURCES:
        raise ValueError("masked export does not disable optical and SAR together")
    if provenance.get("unmasked_embedding_as_masked_result") is not False:
        raise ValueError("unmasked embedding cannot be used as masked evidence")


def validate_masked_export(root: Path) -> dict[str, Any]:
    provenance_path = root / "highres_masked_export_provenance.json"
    audit_path = root / "highres_availability_audit.json"
    meta_path = root / "meta.json"
    for path in (provenance_path, audit_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f"unsealed masked export artifact: {path}")
    provenance = read_json(provenance_path)
    require_masked_provenance(provenance)
    if provenance.get("highres_availability_audit_sha256") != sha256_file(audit_path):
        raise ValueError("masked availability audit hash differs from provenance")
    files = provenance["embedding_verification"]["files"]
    for patch_id, expected_hash in files.items():
        path = root / "harbin" / patch_id / "202604_embedding_map.pt"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"masked embedding hash differs: {patch_id}")
    return {
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "metadata_sha256": sha256_file(meta_path),
        "availability_audit_sha256": sha256_file(audit_path),
        "embedding_files": files,
    }


def load_diagnostic_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("protocol_id") != "harbin_p10c_highres_mask_diagnostic_20260729":
        raise ValueError("unexpected masked diagnostic protocol")
    matrix = config.get("downstream_matrix")
    if not isinstance(matrix, dict) or tuple(matrix.get("families", ())) != (DIAGNOSTIC_FAMILY,):
        raise ValueError("masked diagnostic must remain P10C-only")
    primary_config = REPO_ROOT / str(matrix.get("primary_config", ""))
    primary_lock = Path(str(matrix.get("primary_output_root", ""))) / "matrix_input_lock.json"
    if not primary_config.is_file() or matrix.get("primary_config_sha256") != sha256_file(
        primary_config
    ):
        raise ValueError("locked primary diagnostic configuration differs")
    if not primary_lock.is_file() or matrix.get("primary_matrix_lock_sha256") != sha256_file(
        primary_lock
    ):
        raise ValueError("locked primary diagnostic matrix differs")
    return config


def copy_locked_schedules(
    source_root: Path, destination_root: Path, schedules: dict[str, Any]
) -> None:
    for relative in schedules:
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(
                    f"diagnostic schedule differs from locked primary schedule: {relative}"
                )
        else:
            shutil.copyfile(source, destination)


def prepare_diagnostic(config_path: Path, output_root: Path) -> Any:
    config = load_diagnostic_config(config_path)
    matrix_config_path = REPO_ROOT / config["downstream_matrix"]["primary_config"]
    primary_root = Path(config["downstream_matrix"]["primary_output_root"])
    primary = runner._load_worker_prepared(matrix_config_path, primary_root)
    if primary.config["protocol_id"] != PRIMARY_PROTOCOL:
        raise ValueError("diagnostic must reuse the locked primary schedule protocol")
    masked_root = Path(config["downstream_matrix"]["masked_embedding_root"])
    seal = validate_masked_export(masked_root)
    copy_locked_schedules(primary_root, output_root, primary.schedules)
    diagnostic_config = {
        "protocol_id": config["protocol_id"],
        "shot_schedule_protocol_id": primary.config["protocol_id"],
        "tasks": primary.config["tasks"],
        "folds": primary.config["folds"],
        "shots": primary.config["shots"],
        "seeds": primary.config["seeds"],
        "heads": primary.config["heads"],
        "reader": primary.config["reader"],
        "threshold_rule": primary.config["threshold_rule"],
    }
    family = {
        "id": DIAGNOSTIC_FAMILY,
        "embedding_root": str(masked_root),
        "month": "202604",
        "seal_file": "highres_masked_export_provenance.json",
        "encoder_status": "frozen_haidian_p10c_highres_optical_and_sar_masked",
    }
    prepared = replace(
        primary,
        config=diagnostic_config,
        families=(family,),
        lock_path=output_root / "matrix_input_lock.json",
    )
    lock = {
        "schema_version": 1,
        "protocol_id": diagnostic_config["protocol_id"],
        "table_scope": "p10c_modality_diagnostic_only_not_aef_primary_table",
        "config_path": str(config_path.resolve()),
        "config_sha256": runner.canonical_sha256(config),
        "primary_matrix_lock": str(primary.lock_path.resolve()),
        "primary_matrix_lock_sha256": sha256_file(primary.lock_path),
        "shared_schedule_protocol_id": PRIMARY_PROTOCOL,
        "shared_schedules": primary.schedules,
        "masked_embedding_seal": seal,
        "family": family,
        "job_count": len(runner.build_jobs(prepared)),
        "reader": diagnostic_config["reader"],
        "threshold_rule": diagnostic_config["threshold_rule"],
    }
    if prepared.lock_path.exists() and read_json(prepared.lock_path) != lock:
        raise ValueError("diagnostic input lock differs; refuse mixed masked inputs")
    runner.write_json_atomically(prepared.lock_path, lock)
    return prepared


def load_worker(config_path: Path, output_root: Path) -> Any:
    prepared = prepare_diagnostic(config_path, output_root)
    lock = read_json(prepared.lock_path)
    if (
        lock.get("job_count") != 540
        or lock.get("table_scope") != "p10c_modality_diagnostic_only_not_aef_primary_table"
    ):
        raise ValueError("diagnostic worker lock is invalid")
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/eval/harbin_p10c_highres_mask_diagnostic_20260729.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--devices", nargs="+", default=[f"npu:{index}" for index in range(6)])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if args.device is None:
            raise ValueError("worker requires --device")
        prepared = load_worker(args.config, args.output_root)
        for index, job in enumerate(runner.build_jobs(prepared)):
            if index % args.worker_count == args.worker_index:
                runner.run_reader_cell(prepared, device=args.device, **job)
        return
    prepared = prepare_diagnostic(args.config, args.output_root)
    print(f"[prepared] {len(runner.build_jobs(prepared))} P10C-only masked diagnostic cells")
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
        raise RuntimeError(f"masked diagnostic worker failures: {exit_codes}")


if __name__ == "__main__":
    main()
