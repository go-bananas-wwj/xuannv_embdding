#!/usr/bin/env python3
"""Build the label-free reference-grid inventory for the AEF contextual comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import rasterio

from scripts.eval.registered_v5_matrix import V5_EVAL_MANIFEST_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def reference_record(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        grid = {
            "crs": str(dataset.crs),
            "transform": list(dataset.transform)[:6],
            "bounds": [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ],
            "shape": [dataset.height, dataset.width],
        }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        **grid,
        "grid_sha256": canonical_sha256(grid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed-region-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.manifest) != V5_EVAL_MANIFEST_SHA256:
        raise ValueError("Coverage inventory requires the registered V5 320-patch manifest")
    manifest_records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_by_patch = {str(record["patch_id"]): record for record in manifest_records}
    patch_ids = sorted(records_by_patch)
    if len(manifest_records) != 320 or len(patch_ids) != 320:
        raise ValueError("Coverage inventory requires exactly 320 unique patch IDs")

    def reference_from_manifest(patch_id: str) -> Path:
        candidates = records_by_patch[patch_id].get("s2")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"V5 manifest has no Sentinel-2 reference for {patch_id}")
        path = args.processed_region_root / str(candidates[0])
        if not path.is_file():
            raise FileNotFoundError(f"V5 manifest reference raster is missing: {path}")
        return path

    payload = {
        "schema_version": 1,
        "kind": "aef_v5_coverage_inventory",
        "v5_manifest_sha256": V5_EVAL_MANIFEST_SHA256,
        "records": [
            {
                "patch_id": patch_id,
                "reference_grid": reference_record(reference_from_manifest(patch_id)),
            }
            for patch_id in patch_ids
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
