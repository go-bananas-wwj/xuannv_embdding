#!/usr/bin/env python3
"""Export and seal the two 2026-04 Harbin P10C paper embedding families.

The Harbin AEF inventory, not a label or a downstream split, defines the common
patch universe.  Both Xuannv families are exported from that exact universe so
the later Conv3x3 benchmark cannot silently compare different locations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("p10c_harbin_scratch", "p10c_haidian_frozen_harbin")
EXPECTED_SHAPE = (64, 128, 128)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for one immutable file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_export_registry(path: str | Path) -> dict[str, Any]:
    """Load the two-family export contract without accepting a relaxed transfer."""
    registry_path = Path(path)
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Harbin export registry has an invalid schema")
    if raw.get("kind") != "p10c_harbin_paper_embedding_exports":
        raise ValueError("Harbin export registry has an invalid kind")
    if raw.get("region") != "harbin" or raw.get("month") != "202604":
        raise ValueError("Harbin export registry must target harbin 202604")
    if not isinstance(raw.get("coverage_inventory"), str):
        raise ValueError("Harbin export registry lacks coverage_inventory")
    exports = raw.get("exports")
    if not isinstance(exports, dict) or set(exports) != set(FAMILIES):
        raise ValueError("Harbin export registry must contain exactly the two paper families")
    devices: set[str] = set()
    for family in FAMILIES:
        entry = exports[family]
        if not isinstance(entry, dict) or set(entry) != {
            "config",
            "checkpoint",
            "source_manifest",
            "device",
            "encoder_frozen",
        }:
            raise ValueError(f"Harbin export registry entry is invalid: {family}")
        path_fields = (key for key in entry if key != "encoder_frozen")
        if not all(isinstance(entry[key], str) and entry[key] for key in path_fields):
            raise ValueError(f"Harbin export registry has an empty path/device: {family}")
        devices.add(str(entry["device"]))
    if len(devices) != len(FAMILIES):
        raise ValueError("Harbin paper exports require distinct devices for parallel execution")
    if exports["p10c_harbin_scratch"]["encoder_frozen"] is not False:
        raise ValueError("Harbin scratch export cannot claim a frozen encoder")
    if exports["p10c_haidian_frozen_harbin"]["encoder_frozen"] is not True:
        raise ValueError("Haidian-to-Harbin export must keep the encoder frozen")
    return raw


def _coverage_patch_ids(path: Path) -> list[str]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    records = inventory.get("records") if isinstance(inventory, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("AEF coverage inventory must contain non-empty records")
    patch_ids = [str(record.get("patch_id", "")) for record in records if isinstance(record, dict)]
    patch_ids_are_unique = len(set(patch_ids)) == len(patch_ids)
    if not all(patch_ids) or len(patch_ids) != len(records) or not patch_ids_are_unique:
        raise ValueError("AEF coverage inventory has invalid patch IDs")
    return sorted(patch_ids)


def materialize_aef_covered_manifest(
    source_manifest: str | Path, coverage_inventory: str | Path, destination: str | Path
) -> list[str]:
    """Copy only the AEF-covered Harbin records, preserving relative imagery paths."""
    source_path = Path(source_manifest)
    destination_path = Path(destination)
    if source_path.parent.resolve() != destination_path.parent.resolve():
        raise ValueError("covered manifest must share the source manifest parent directory")
    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("source manifest must be a list")
    wanted = _coverage_patch_ids(Path(coverage_inventory))
    by_id = {
        str(record.get("patch_id", "")): record for record in records if isinstance(record, dict)
    }
    if len(by_id) != len(records) or any(patch_id not in by_id for patch_id in wanted):
        raise ValueError("AEF coverage patch set is not a subset of the source manifest")
    selected = [by_id[patch_id] for patch_id in wanted]
    if {str(record.get("region")) for record in selected} != {"harbin"}:
        raise ValueError("covered manifest must contain only Harbin records")
    destination_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return wanted


def verify_export_maps(export_root: str | Path, patch_ids: list[str], month: str) -> dict[str, Any]:
    """Reject incomplete, non-finite, wrong-shape, or collapsed embedding maps."""
    region_root = Path(export_root) / "harbin"
    actual_ids = sorted(path.name for path in region_root.iterdir() if path.is_dir())
    if actual_ids != sorted(patch_ids):
        raise ValueError("exported patch IDs do not exactly match the AEF-covered manifest")
    files: dict[str, str] = {}
    for patch_id in patch_ids:
        path = region_root / patch_id / f"{month}_embedding_map.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing embedding map: {path}")
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor) or tuple(embedding.shape) != EXPECTED_SHAPE:
            raise ValueError(f"embedding map has invalid shape: {path}")
        if not bool(torch.isfinite(embedding).all()):
            raise ValueError(f"embedding map contains NaN or Inf: {path}")
        if bool((embedding.var(dim=(1, 2), unbiased=False) == 0).any()):
            raise ValueError(f"embedding map has zero-variance channel: {path}")
        files[patch_id] = sha256_file(path)
    return {"patch_count": len(patch_ids), "shape": list(EXPECTED_SHAPE), "files": files}


def build_export_command(
    entry: dict[str, Any], output_root: Path, manifest_path: Path, month: str
) -> list[str]:
    """Build one precompute command; caller starts the two independent commands together."""
    return [
        sys.executable,
        "downstreams/scripts/precompute_embeddings.py",
        "--config",
        str(entry["config"]),
        "--regions",
        "harbin",
        "--output-root",
        str(output_root),
        "--export-name",
        str(entry["id"]),
        "--manifest-path",
        str(manifest_path),
        "--checkpoint",
        str(entry["checkpoint"]),
        "--months",
        month,
        "--device",
        str(entry["device"]),
    ]


def write_export_provenance(
    export_root: Path,
    registry_path: Path,
    source_manifest: Path,
    coverage_inventory: Path,
    verification: dict[str, Any],
) -> None:
    """Seal the completed maps with exactly the inputs that governed their export."""
    payload = {
        "schema_version": 1,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "coverage_inventory": str(coverage_inventory.resolve()),
        "coverage_inventory_sha256": sha256_file(coverage_inventory),
        "verification": verification,
    }
    (export_root / "paper_export_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "configs/eval/paper_harbin_embedding_exports_20260728.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/paper_p10c_harbin_20260728"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = args.registry.resolve()
    registry = load_export_registry(registry_path)
    coverage_inventory = (REPO_ROOT / registry["coverage_inventory"]).resolve()
    if not coverage_inventory.is_file():
        raise FileNotFoundError(f"missing AEF coverage inventory: {coverage_inventory}")
    jobs: list[tuple[str, subprocess.Popen[str], Path, Path]] = []
    for family in FAMILIES:
        entry = {**registry["exports"][family], "id": family}
        source_manifest = Path(entry["source_manifest"])
        if not source_manifest.is_file():
            raise FileNotFoundError(f"missing source manifest: {source_manifest}")
        covered_manifest = source_manifest.with_name(
            f"{source_manifest.stem}_{family}_aef_covered.json"
        )
        patch_ids = materialize_aef_covered_manifest(
            source_manifest, coverage_inventory, covered_manifest
        )
        export_root = args.output_root / family
        command = build_export_command(entry, args.output_root, covered_manifest, registry["month"])
        if args.dry_run:
            print(" ".join(command))
            continue
        jobs.append(
            (
                family,
                subprocess.Popen(command, cwd=REPO_ROOT, text=True),
                export_root,
                covered_manifest,
            )
        )
        print(f"started {family} on {entry['device']} for {len(patch_ids)} AEF-covered patches")
    for family, process, export_root, covered_manifest in jobs:
        if process.wait() != 0:
            raise RuntimeError(f"Harbin embedding export failed: {family}")
        patch_ids = _coverage_patch_ids(coverage_inventory)
        verification = verify_export_maps(export_root, patch_ids, registry["month"])
        write_export_provenance(
            export_root, registry_path, covered_manifest, coverage_inventory, verification
        )
        print(f"sealed {family}: {verification['patch_count']} maps")


if __name__ == "__main__":
    main()
