#!/usr/bin/env python3
"""Resolve a registered V5 encoder checkpoint from its queue-sealed pointer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_pointer_sha256(payload: dict[str, object]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve_registered_v5_checkpoint(output_root: Path, *, family: str, fold: int) -> Path:
    """Return only the immutable checkpoint selected by the registered queue.

    The queue stores successful checkpoints within a specific attempt and seals a
    ``canonical_attempt.json`` pointer at the experiment root.  Consumers must
    validate that pointer instead of assuming a mutable top-level ``best.pt``.
    """
    if family != "full_150" or fold not in range(5):
        raise ValueError("unsupported registered V5 encoder family or fold")
    job_root = (
        output_root.resolve() / f"paper_registered_v5_{family}_fold{fold}_20260726"
    )
    pointer_path = job_root / "canonical_attempt.json"
    return validate_registered_v5_checkpoint_path(
        _checkpoint_from_pointer(pointer_path), expected_job_name=job_root.name
    )


def _checkpoint_from_pointer(pointer_path: Path) -> Path:
    """Read a canonical pointer and return its declared checkpoint path."""
    if not pointer_path.is_file():
        raise FileNotFoundError(f"registered V5 canonical attempt is missing: {pointer_path}")
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    return Path(str(payload.get("checkpoint_path", ""))).resolve()


def validate_registered_v5_checkpoint_path(checkpoint: Path, *, expected_job_name: str) -> Path:
    """Validate that a checkpoint is exactly the queue-selected V5 snapshot."""
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "verified_checkpoints":
        raise ValueError("registered V5 checkpoint is not a verified queue snapshot")
    attempt = checkpoint.parent.parent
    job_root = attempt.parent
    pointer_path = job_root / "canonical_attempt.json"
    if job_root.name != expected_job_name or not pointer_path.is_file():
        raise ValueError("registered V5 checkpoint is not bound to the expected encoder job")
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("registered V5 canonical attempt has an invalid schema")
    if payload.get("sha256") != _canonical_pointer_sha256(payload):
        raise ValueError("registered V5 canonical attempt self-hash changed")
    declared_attempt = Path(str(payload.get("attempt_dir", ""))).resolve()
    declared_checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    if declared_attempt != attempt or attempt.parent != job_root or not attempt.name.startswith("attempt_"):
        raise ValueError("registered V5 canonical attempt points outside its job")
    if declared_checkpoint != checkpoint:
        raise ValueError("registered V5 canonical attempt points to a different checkpoint")
    manifest = attempt / "attempt_manifest.json"
    if not manifest.is_file() or _sha256(manifest) != payload.get("attempt_manifest_sha256"):
        raise ValueError("registered V5 canonical attempt manifest hash changed")
    try:
        checkpoint.relative_to(attempt)
    except ValueError as exc:
        raise ValueError("registered V5 checkpoint points outside its canonical attempt") from exc
    if not checkpoint.is_file() or _sha256(checkpoint) != payload.get("checkpoint_sha256"):
        raise ValueError("registered V5 canonical checkpoint hash changed")
    verification = attempt / "checkpoint_verifications" / f"{checkpoint.name}.json"
    if not verification.is_file():
        raise ValueError("registered V5 checkpoint verification record is missing")
    verified = json.loads(verification.read_text(encoding="utf-8"))
    if (
        verified.get("schema_version") != 1
        or Path(str(verified.get("attempt_dir", ""))).resolve() != attempt
        or Path(str(verified.get("checkpoint_path", ""))).resolve() != checkpoint
        or verified.get("checkpoint_sha256") != payload.get("checkpoint_sha256")
    ):
        raise ValueError("registered V5 checkpoint verification record does not match the pointer")
    return checkpoint


def validate_registered_v5_checkpoint_config_binding(
    checkpoint: Path, *, expected_config_path: Path
) -> Path:
    """Require the queue-sealed checkpoint to originate from the requested config bytes."""
    checkpoint = checkpoint.resolve()
    attempt = checkpoint.parent.parent
    expected_config_path = expected_config_path.resolve()
    manifest_path = attempt / "attempt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_config")
    if not isinstance(source, dict):
        raise ValueError("registered V5 attempt manifest lacks source config provenance")
    source_path = Path(str(source.get("path", ""))).resolve()
    source_sha256 = source.get("sha256")
    if source_path != expected_config_path:
        raise ValueError("registered V5 checkpoint source config does not match requested config")
    if not expected_config_path.is_file() or source_sha256 != _sha256(expected_config_path):
        raise ValueError("registered V5 checkpoint source config hash does not match requested config")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--job-root", type=Path)
    args = parser.parse_args()
    if args.job_root is not None:
        job_root = args.job_root.resolve()
        pointer = job_root / "canonical_attempt.json"
        checkpoint = _checkpoint_from_pointer(pointer)
        print(validate_registered_v5_checkpoint_path(checkpoint, expected_job_name=job_root.name))
        return
    if args.output_root is None or args.family is None or args.fold is None:
        parser.error("either --job-root or --output-root, --family, and --fold are required")
    print(resolve_registered_v5_checkpoint(args.output_root, family=args.family, fold=args.fold))


if __name__ == "__main__":
    main()
