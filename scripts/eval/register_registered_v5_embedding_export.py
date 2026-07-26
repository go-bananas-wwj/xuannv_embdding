#!/usr/bin/env python3
"""Append one verified pending v5 export entry to a clean Git-tracked registry.

This command intentionally never commits.  The exporter writes a pending entry
next to the sealed embedding maps; this registrar is the explicit, reviewable
step that updates the Git-tracked registry and leaves the resulting diff for a
human commit.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--pending-entry", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pending_entry(path: Path, registry_path: Path) -> dict[str, Any]:
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
    return entry


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


def append_pending_entry(registry_path: Path, entry: dict[str, Any], *, dry_run: bool) -> None:
    """Append exactly once after verifying the current registry is committed and unchanged."""
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    registered.validate_v5_embedding_registry(registry_path, raw)
    exports = raw["exports"]
    if any(item.get("id") == entry["id"] for item in exports):
        raise ValueError("Registry already contains this v5 export entry ID")
    if dry_run:
        print(f"registry verified; commit required after appending {entry['id']}")
        return
    raw["exports"] = [*exports, entry]
    registry_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"registry updated with {entry['id']}; commit required before any v5 probe")


def main() -> None:
    args = parse_args()
    pending_entry = load_pending_entry(args.pending_entry, args.registry)
    entry = rebuild_entry_from_export(pending_entry)
    append_pending_entry(args.registry, entry, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
