#!/usr/bin/env python3
"""Export official annual AEF maps onto the frozen V5 patch grid.

This is intentionally a label-free, fail-closed data preparation boundary.  It
does not know about downstream tasks, spatial folds, shots, or probe outputs.
The resulting directory is a *contextual* AEF comparator input, never an AEF
checkpoint entry in the Xuannv encoder registry.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch

from scripts.data.prepare_aef_2025_embeddings import (
    AEF_YEAR,
    cache_cog,
    patch_wgs84_geometry,
    read_aef_patch_with_validity,
)
from scripts.eval import registered_v5_matrix
from scripts.eval.registered_v5_matrix import (
    V5_EVAL_MANIFEST,
    V5_EVAL_MANIFEST_SHA256,
)

LOGGER = logging.getLogger(__name__)
OUTPUT_MONTH = "annual_2025"
OUTPUT_NAME = f"{OUTPUT_MONTH}_embedding_map.pt"
EXPECTED_CHANNELS = 64
LOCKED_AEF_INDEX_URI = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def load_committed_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse exactly the bytes currently committed at HEAD, never a later rewrite."""
    resolved = path.resolve()
    root = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    repo_root = Path(root.stdout.strip())
    relative = resolved.relative_to(repo_root)
    head_bytes = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repo_root, capture_output=True, check=True
    ).stdout
    if resolved.read_bytes() != head_bytes:
        raise ValueError(f"AEF protocol input is not identical to current Git HEAD: {relative}")
    payload = json.loads(head_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AEF protocol input must be a JSON object: {relative}")
    return payload, hashlib.sha256(head_bytes).hexdigest()


def load_committed_json(path: Path) -> dict[str, Any]:
    return load_committed_json_snapshot(path)[0]


def committed_json_sha256(path: Path) -> str:
    resolved = path.resolve()
    root = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    relative = resolved.relative_to(Path(root.stdout.strip()))
    head = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root.stdout.strip(),
        capture_output=True,
        check=True,
    ).stdout
    return hashlib.sha256(head).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--aef-index-path", required=True)
    parser.add_argument("--aef-index-uri", default=LOCKED_AEF_INDEX_URI)
    parser.add_argument("--cog-cache-dir", type=Path, required=True)
    parser.add_argument("--cog-lock", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_coverage_inventory(path: Path) -> tuple[list[str], dict[str, dict[str, Any]], str]:
    payload, digest = load_committed_json_snapshot(path)
    if payload.get("schema_version") != 1 or payload.get("kind") != "aef_v5_coverage_inventory":
        raise ValueError("AEF coverage inventory has an invalid schema")
    if payload.get("v5_manifest_sha256") != V5_EVAL_MANIFEST_SHA256:
        raise ValueError("AEF coverage inventory is not bound to the registered V5 manifest")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 320:
        raise ValueError("AEF coverage inventory must contain exactly 320 patch records")
    patch_ids = [str(record.get("patch_id", "")) for record in records]
    if not patch_ids or any(not patch_id for patch_id in patch_ids):
        raise ValueError("AEF coverage inventory has empty patch IDs")
    if len(set(patch_ids)) != len(patch_ids):
        raise ValueError("AEF coverage inventory has duplicate patch IDs")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        reference = record.get("reference_grid")
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            raise ValueError("AEF coverage inventory lacks a reference-grid record")
        by_id[str(record["patch_id"])] = dict(reference)
    if not V5_EVAL_MANIFEST.is_file() or sha256_file(V5_EVAL_MANIFEST) != V5_EVAL_MANIFEST_SHA256:
        raise ValueError("Registered V5 evaluation manifest is missing or has changed")
    v5_patch_ids = {
        str(record["patch_id"])
        for record in json.loads(V5_EVAL_MANIFEST.read_text(encoding="utf-8"))
    }
    if set(patch_ids) != v5_patch_ids:
        raise ValueError("AEF coverage inventory patch IDs differ from registered V5 manifest")
    return sorted(patch_ids), by_id, digest


def load_cog_lock(path: Path) -> tuple[dict[str, str], dict[str, Any], str]:
    payload, digest = load_committed_json_snapshot(path)
    entries = payload.get("assets")
    if payload.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("AEF COG lock has an invalid schema")
    lock = {str(item.get("filename")): str(item.get("sha256")) for item in entries}
    if not lock or any(len(digest) != 64 for digest in lock.values()):
        raise ValueError("AEF COG lock has invalid URI/SHA-256 entries")
    return lock, payload, digest


def snapshot_verified_file(
    source: Path, expected_sha256: str, staging_root: Path, kind: str
) -> Path:
    """Freeze a verified source object before raster I/O to close replacement races."""
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"AEF {kind} bytes do not match frozen hash: {source}")
    snapshot_dir = staging_root / f"locked_{kind}s"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"{expected_sha256}{source.suffix}"
    if not snapshot.exists():
        shutil.copy2(source, snapshot)
    if sha256_file(snapshot) != expected_sha256:
        raise ValueError(f"AEF {kind} snapshot hash mismatch: {source}")
    snapshot.chmod(0o444)
    return snapshot


def snapshot_locked_cog(local_cog: Path, expected_sha256: str, staging_root: Path) -> Path:
    return snapshot_verified_file(local_cog, expected_sha256, staging_root, "cog")


def snapshot_reference_grid(reference: Path, inventory: dict[str, Any], staging_root: Path) -> Path:
    frozen = snapshot_verified_file(reference, str(inventory["sha256"]), staging_root, "reference")
    observed = _reference_record(frozen)
    for key in ("sha256", "grid_sha256", "crs", "transform", "bounds", "shape"):
        if observed.get(key) != inventory.get(key):
            raise ValueError(
                f"AEF reference snapshot disagrees with coverage inventory: {reference}"
            )
    return frozen


def _row_cog_uri(row: Any) -> str:
    assets = row.get("assets")
    if isinstance(assets, dict) and isinstance(assets.get("data"), dict):
        href = assets["data"].get("href")
        if href:
            return str(href)
    path = row.get("path")
    if path:
        return str(path)
    raise KeyError("AEF index row has neither assets.data.href nor path")


def candidate_cogs(index: gpd.GeoDataFrame, ref_path: Path) -> list[str]:
    matches = index[index.intersects(patch_wgs84_geometry(ref_path))]
    return sorted({_row_cog_uri(row) for _, row in matches.iterrows()})


def select_single_cog(candidates: list[str], ref_path: Path) -> str:
    """Legacy single-source selector retained for direct unit checks."""
    if not candidates:
        raise FileNotFoundError(f"No official AEF COG intersects {ref_path}")
    if len(candidates) != 1:
        raise ValueError(
            f"multiple official AEF COGs intersect {ref_path}; no mosaic rule is registered"
        )
    return candidates[0]


def read_complete_aef_patch(cog_path: Path, ref_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Return only a fully covered 64-channel official AEF map."""
    embedding, valid = read_aef_patch_with_validity(str(cog_path), ref_path)
    if embedding.ndim != 3 or embedding.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Official AEF product must contain {EXPECTED_CHANNELS} channels, "
            f"got {tuple(embedding.shape)}"
        )
    if embedding.shape[-2:] != (128, 128):
        raise ValueError(f"Official AEF output must be 128x128, got {tuple(embedding.shape[-2:])}")
    if valid.shape != (128, 128) or not bool(valid.all()):
        raise ValueError(
            "Official AEF export requires complete valid coverage for every output pixel"
        )
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError(
            "Official AEF embedding contains non-finite values after complete-coverage check"
        )
    return embedding.contiguous(), valid


def read_complete_aef_mosaic(
    cog_paths: list[Path], ref_path: Path
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Use a declared URI-sorted, first-valid-pixel mosaic for boundary patches."""
    if not cog_paths:
        raise ValueError("AEF mosaic requires at least one frozen source COG")
    mosaic = torch.zeros((EXPECTED_CHANNELS, 128, 128), dtype=torch.float32)
    valid_out = np.zeros((128, 128), dtype=bool)
    selector = np.full((128, 128), 255, dtype=np.uint8)
    for source_index, cog_path in enumerate(cog_paths):
        embedding, valid = read_aef_patch_with_validity(str(cog_path), ref_path)
        if embedding.ndim != 3 or tuple(embedding.shape) != (EXPECTED_CHANNELS, 128, 128):
            raise ValueError(
                f"Official AEF product must be 64x128x128, got {tuple(embedding.shape)}"
            )
        if valid.shape != (128, 128):
            raise ValueError(f"Official AEF valid mask has invalid shape: {valid.shape}")
        if not bool(torch.isfinite(embedding[:, valid]).all()):
            raise ValueError(f"Official AEF source has non-finite valid values: {cog_path}")
        take = valid & ~valid_out
        mosaic[:, take] = embedding[:, take]
        selector[take] = source_index
        valid_out |= valid
    if not bool(valid_out.all()):
        raise ValueError(
            "Official AEF export requires complete valid coverage for every output pixel"
        )
    return mosaic.contiguous(), valid_out, selector


def build_output_index(
    output_root: Path, region: str, expected_patch_ids: set[str]
) -> dict[str, Any]:
    root = output_root.resolve()
    maps = sorted((root / region).glob(f"*/{OUTPUT_NAME}"))
    patch_ids = {path.parent.name for path in maps}
    if patch_ids != expected_patch_ids:
        raise ValueError("AEF exported embedding patch set differs from frozen manifest patch set")
    entries: list[dict[str, Any]] = []
    for path in maps:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != (EXPECTED_CHANNELS, 128, 128):
            raise ValueError(f"AEF output map has invalid shape: {path}: {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"AEF output map has invalid dtype: {path}: {tensor.dtype}")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        )
    payload = {
        "schema_version": 1,
        "region": region,
        "output_identifier": OUTPUT_MONTH,
        "file_count": len(entries),
        "files": entries,
    }
    return {**payload, "index_sha256": canonical_sha256(payload)}


def build_validity_index(
    output_root: Path, region: str, expected_patch_ids: set[str]
) -> dict[str, Any]:
    root = output_root.resolve()
    masks = sorted((root / region).glob(f"*/{OUTPUT_MONTH}_valid_mask.npy"))
    patch_ids = {path.parent.name for path in masks}
    if patch_ids != expected_patch_ids:
        raise ValueError("AEF valid-mask patch set differs from frozen manifest patch set")
    entries: list[dict[str, Any]] = []
    for path in masks:
        valid = np.load(path, allow_pickle=False)
        if valid.shape != (128, 128) or valid.dtype != np.bool_ or not bool(valid.all()):
            raise ValueError(f"AEF validity mask is not complete: {path}")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "valid_pixels": int(valid.sum()),
                "total_pixels": int(valid.size),
            }
        )
    payload = {
        "schema_version": 1,
        "region": region,
        "output_identifier": OUTPUT_MONTH,
        "file_count": len(entries),
        "files": entries,
    }
    return {**payload, "index_sha256": canonical_sha256(payload)}


def _reference_record(path: Path) -> dict[str, Any]:
    import rasterio

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
        "path": str(path),
        "sha256": sha256_file(path),
        **grid,
        "grid_sha256": canonical_sha256(grid),
    }


def verify_reference_record(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Coverage-inventory reference raster is missing: {path}")
    observed = _reference_record(path)
    for key in ("sha256", "grid_sha256", "crs", "transform", "bounds", "shape"):
        if observed.get(key) != record.get(key):
            raise ValueError(f"Coverage-inventory reference grid drifted for {path}: {key}")
    return path


def verify_runtime_sources() -> dict[str, Any]:
    """Require both numerical implementation files to be committed at the current Git HEAD."""
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    repo_root = Path(root.stdout.strip())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    records: list[dict[str, str]] = []
    for path in (
        Path(__file__).resolve(),
        Path(read_aef_patch_with_validity.__code__.co_filename),
        Path(str(registered_v5_matrix.__file__)),
    ):
        relative = path.relative_to(repo_root)
        head = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=repo_root, capture_output=True, check=True
        ).stdout
        if head != path.read_bytes():
            raise ValueError(f"AEF runtime source is not committed at current Git HEAD: {relative}")
        records.append({"path": str(relative), "sha256": sha256_file(path)})
    return {"git_revision": revision, "files": records}


def export(args: argparse.Namespace) -> dict[str, Any]:
    patch_ids, reference_inventory, coverage_inventory_sha256 = load_coverage_inventory(
        args.coverage_inventory
    )
    if args.overwrite:
        raise ValueError(
            "Paper AEF exporter does not support in-place overwrite; use a fresh output root"
        )
    if args.output_root.exists():
        raise FileExistsError(f"Paper AEF output root must not already exist: {args.output_root}")
    if str(args.aef_index_path).startswith("s3://"):
        raise ValueError("Paper exporter requires a local immutable AEF index snapshot path")
    source_index = Path(args.aef_index_path)
    if not source_index.is_file():
        raise FileNotFoundError(f"Missing local AEF index snapshot: {source_index}")
    cog_lock, cog_lock_meta, cog_lock_sha256 = load_cog_lock(args.cog_lock)
    if (
        cog_lock_meta.get("release") != "aef/v1/annual"
        or cog_lock_meta.get("year") != AEF_YEAR
        or cog_lock_meta.get("index_uri") != args.aef_index_uri
        or cog_lock_meta.get("index_sha256") != sha256_file(source_index)
    ):
        raise ValueError("Local AEF index does not match the Git-locked official source inventory")
    runtime_sources = verify_runtime_sources()
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path = args.output_root.parent / f".{args.output_root.name}.lease"
    lease_handle = lease_path.open("a+")
    try:
        try:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another AEF export owns destination lease: {lease_path}") from exc
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{args.output_root.name}.staging.", dir=args.output_root.parent
            )
        )
        snapshot = staging_root / "aef_index_snapshot.parquet"
        if snapshot.resolve() != source_index.resolve():
            shutil.copy2(source_index, snapshot)
        if sha256_file(snapshot) != cog_lock_meta["index_sha256"]:
            raise ValueError("AEF index snapshot changed before it could be parsed")
        snapshot.chmod(0o444)
        index = gpd.read_parquet(snapshot)
        if sha256_file(snapshot) != cog_lock_meta["index_sha256"]:
            raise ValueError("AEF index snapshot changed while it was parsed")
        if "datetime" in index.columns:
            index = index[index["datetime"].dt.year == AEF_YEAR].copy()
        elif "year" in index.columns:
            index = index[index["year"] == AEF_YEAR].copy()
        if index.empty:
            raise RuntimeError(f"No AEF records found for annual {AEF_YEAR}")
        if index.crs is None:
            index = index.set_crs("OGC:CRS84")
        source_records: list[dict[str, Any]] = []
        for position, patch_id in enumerate(patch_ids, start=1):
            reference = verify_reference_record(reference_inventory[patch_id])
            frozen_reference = snapshot_reference_grid(
                reference, reference_inventory[patch_id], staging_root
            )
            candidates = candidate_cogs(index, frozen_reference)
            frozen_cogs: list[Path] = []
            source_assets: list[dict[str, str]] = []
            for selected in candidates:
                local_cog = Path(cache_cog(selected, args.cog_cache_dir))
                if not local_cog.is_file():
                    raise FileNotFoundError(
                        f"AEF cache operation did not yield a local file: {local_cog}"
                    )
                expected_cog_sha256 = cog_lock.get(Path(selected).name)
                if expected_cog_sha256 is None:
                    raise ValueError(
                        f"AEF COG cache bytes do not match the frozen lock: {selected}"
                    )
                frozen_cog = snapshot_locked_cog(local_cog, expected_cog_sha256, staging_root)
                frozen_cogs.append(frozen_cog)
                source_assets.append(
                    {
                        "uri": selected,
                        "cache_path": str(local_cog),
                        "sha256": expected_cog_sha256,
                    }
                )
            embedding, valid, selector = read_complete_aef_mosaic(frozen_cogs, frozen_reference)
            for frozen_cog, source in zip(frozen_cogs, source_assets, strict=True):
                if sha256_file(frozen_cog) != source["sha256"]:
                    raise ValueError(f"AEF COG snapshot changed during read: {source['uri']}")
            if sha256_file(frozen_reference) != reference_inventory[patch_id]["sha256"]:
                raise ValueError(f"AEF reference snapshot changed during read: {reference}")
            patch_dir = staging_root / args.region / patch_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            map_path = patch_dir / OUTPUT_NAME
            valid_path = patch_dir / f"{OUTPUT_MONTH}_valid_mask.npy"
            selector_path = patch_dir / f"{OUTPUT_MONTH}_source_selector.npy"
            torch.save(embedding, map_path)
            np.save(valid_path, valid, allow_pickle=False)
            np.save(selector_path, selector, allow_pickle=False)
            reference_grid = _reference_record(frozen_reference)
            source_records.append(
                {
                    "patch_id": patch_id,
                    "ordered_candidate_cogs": candidates,
                    "selection_rule": "uri_sorted_first_valid_pixel_mosaic",
                    "source_assets": source_assets,
                    "reference_grid": reference_grid,
                    "reprojection_footprint": reference_grid["bounds"],
                    "valid_mask_path": str(valid_path.relative_to(staging_root)),
                    "valid_mask_sha256": sha256_file(valid_path),
                    "source_selector_path": str(selector_path.relative_to(staging_root)),
                    "source_selector_sha256": sha256_file(selector_path),
                    "valid_pixel_count": int(valid.sum()),
                    "output_path": str(map_path.relative_to(staging_root)),
                }
            )
            LOGGER.info("[%d/%d] exported %s", position, len(patch_ids), patch_id)

        # Copied source objects protect raster I/O but are not published as embedding artifacts.
        shutil.rmtree(staging_root / "locked_cogs", ignore_errors=True)
        shutil.rmtree(staging_root / "locked_references", ignore_errors=True)

        source_manifest = {
            "schema_version": 1,
            "release": "aef/v1/annual",
            "year": AEF_YEAR,
            "index_snapshot": {
                "uri": args.aef_index_uri,
                "path": snapshot.name,
                "sha256": sha256_file(snapshot),
            },
            "dequantization": "sign(x) * (x / 127.5)^2; int8 -128 is nodata",
            "resampling": "bilinear_reproject_to_reference_grid",
            "global_label_free": True,
            "records": source_records,
        }
        source_path = staging_root / "aef_source_manifest.json"
        source_path.write_text(
            json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_index = build_output_index(staging_root, args.region, set(patch_ids))
        output_index_path = staging_root / "embedding_file_index.json"
        output_index_path.write_text(
            json.dumps(output_index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        validity_index = build_validity_index(staging_root, args.region, set(patch_ids))
        validity_index_path = staging_root / "validity_index.json"
        validity_index_path.write_text(
            json.dumps(validity_index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta = {
            "schema_version": 1,
            "protocol_id": "aef_annual_2025_contextual",
            "scientific_status": "contextual_comparator_preliminary",
            "release": "aef/v1/annual",
            "output_identifier": OUTPUT_MONTH,
            "global_label_free": True,
            "coverage_inventory": {
                "path": str(args.coverage_inventory),
                "sha256": coverage_inventory_sha256,
                "patch_count": len(patch_ids),
            },
            "cog_lock": {
                "path": str(args.cog_lock),
                "sha256": cog_lock_sha256,
            },
            "source_manifest": {"path": source_path.name, "sha256": sha256_file(source_path)},
            "embedding_file_index": {
                "path": output_index_path.name,
                "sha256": sha256_file(output_index_path),
            },
            "validity_index": {
                "path": validity_index_path.name,
                "sha256": sha256_file(validity_index_path),
            },
            "input_allowlist": [
                str(args.coverage_inventory),
                str(source_index),
                str(args.cog_cache_dir),
                str(args.cog_lock),
            ],
            "runtime_sources": runtime_sources,
            "argv": sys.argv,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
        }
        (staging_root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging_root.replace(args.output_root)
        return meta
    except Exception:
        if "staging_root" in locals():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise
    finally:
        fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
        lease_handle.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    export(args)


if __name__ == "__main__":
    main()
