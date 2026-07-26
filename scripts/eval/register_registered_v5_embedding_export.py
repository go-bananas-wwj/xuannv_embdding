#!/usr/bin/env python3
"""Append one verified pending v5 export entry to a clean Git-tracked registry.

This command intentionally never commits.  The exporter writes a pending entry
next to the sealed embedding maps; this registrar is the explicit, reviewable
step that updates the Git-tracked registry and leaves the resulting diff for a
human commit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "downstreams")]

from scripts.eval import run_registered_paper_downstream as registered  # noqa: E402

REQUIRED_ENTRY_FIELDS = {
    "id",
    "family",
    "encoder_fold",
    "protocol_id",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "embedding_file_index_sha256",
    "canonical_export_provenance_sha256",
    "region",
    "month",
    "patch_count",
    "embedding_root",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pending-entry", type=Path, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pending_entry(path: Path, registry_path: Path) -> tuple[dict[str, Any], str]:
    """Load only a hash-sealed pending entry targeting this exact registry version."""
    pending = json.loads(path.read_text(encoding="utf-8"))
    if pending.get("schema_version") != 1:
        raise ValueError("Pending v5 registry entry has an invalid schema")
    if pending.get("registry_path") != str(registry_path.resolve()):
        raise ValueError("Pending v5 registry entry targets a different registry")
    if pending.get("registry_sha256") != registered.sha256_file(registry_path):
        raise ValueError("Pending v5 registry entry was built from a different registry version")
    entry = pending.get("entry")
    if not isinstance(entry, dict) or set(entry) != REQUIRED_ENTRY_FIELDS:
        raise ValueError("Pending v5 registry entry has an invalid entry schema")
    if pending.get("entry_sha256") != registered._canonical_sha256(entry):
        raise ValueError("Pending v5 registry entry hash is invalid")
    if entry["protocol_id"] != "v5_osm_assisted" or entry["patch_count"] != 320:
        raise ValueError("Pending entry does not satisfy the v5 export protocol")
    return entry, str(pending["registry_sha256"])


def rebuild_entry_from_export(pending_entry: dict[str, Any]) -> dict[str, Any]:
    """Recompute the entry from the sealed export rather than trusting pending JSON."""
    embedding_root = Path(str(pending_entry["embedding_root"])).resolve()
    if not embedding_root.is_dir():
        raise FileNotFoundError(f"Pending v5 export root does not exist: {embedding_root}")
    rebuilt = registered.build_v5_embedding_registry_entry(
        embedding_root,
        family=str(pending_entry["family"]),
        fold=int(pending_entry["encoder_fold"]),
        region=str(pending_entry["region"]),
        month=str(pending_entry["month"]),
    )
    entry = {**rebuilt, "embedding_root": str(embedding_root)}
    if entry != pending_entry:
        raise ValueError("Pending v5 registry entry does not match the sealed export")
    return entry


def append_pending_entries(
    registry_path: Path,
    entries: list[dict[str, Any]],
    *,
    expected_registry_sha256: str,
    dry_run: bool,
) -> None:
    """Append one registry-snapshot batch atomically after rebuilding every entry."""
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registered.sha256_file(registry_path) != expected_registry_sha256:
            raise ValueError("Pending V5 registry batch was superseded by another registrar")
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        registered.validate_v5_embedding_registry(registry_path, raw)
        exports = raw["exports"]
        identifiers = [str(entry["id"]) for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Pending v5 registry entries contain duplicate IDs")
        expected_folds = set(range(5))
        folds = {int(entry["encoder_fold"]) for entry in entries}
        if len(entries) != 5 or folds != expected_folds:
            raise ValueError("Pending V5 registry batch must contain exactly folds 0 through 4")
        if {
            (entry["family"], entry["protocol_id"], entry["region"], entry["month"])
            for entry in entries
        } != {("full_150", "v5_osm_assisted", "haidian", "202604")}:
            raise ValueError("Pending V5 registry batch does not share the registered export identity")
        if any(item.get("id") in identifiers for item in exports):
            raise ValueError("Registry already contains a pending v5 export entry ID")
        if dry_run:
            print(f"registry verified; commit required after appending {len(entries)} v5 export entries")
            return
        raw["exports"] = [*exports, *entries]
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{registry_path.name}.", dir=registry_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, registry_path)
            directory = os.open(registry_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"registry updated with {len(entries)} v5 export entries; commit required before any v5 probe")


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    pending_records = [load_pending_entry(path, registry_path) for path in args.pending_entry]
    pending_entries = [entry for entry, _ in pending_records]
    snapshots = {snapshot for _, snapshot in pending_records}
    if len(snapshots) != 1:
        raise ValueError("Pending V5 registry batch must share the same registry snapshot")
    entries = [rebuild_entry_from_export(entry) for entry in pending_entries]
    append_pending_entries(
        registry_path,
        entries,
        expected_registry_sha256=snapshots.pop(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
