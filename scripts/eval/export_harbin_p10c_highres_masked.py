#!/usr/bin/env python3
"""Export a separately sealed Harbin P10C embedding with all high-res availability off."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_PAPER_EXPORT_PATH = REPO_ROOT / "scripts/eval/export_p10c_harbin_paper_embeddings.py"
_PAPER_EXPORT_SPEC = importlib.util.spec_from_file_location("paper_export", _PAPER_EXPORT_PATH)
if _PAPER_EXPORT_SPEC is None or _PAPER_EXPORT_SPEC.loader is None:
    raise ImportError(f"cannot load {_PAPER_EXPORT_PATH}")
paper_export = importlib.util.module_from_spec(_PAPER_EXPORT_SPEC)
sys.modules[_PAPER_EXPORT_SPEC.name] = paper_export
_PAPER_EXPORT_SPEC.loader.exec_module(paper_export)

DISABLED_HIGHRES_SOURCES = ("highres_optical_haidian", "highres_sar_haidian")
EXPECTED_SHAPE = (64, 128, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    required = {
        "schema_version": 1,
        "kind": "harbin_p10c_highres_mask_diagnostic",
        "protocol_id": "harbin_p10c_highres_mask_diagnostic_20260729",
        "region": "harbin",
        "month": "202604",
        "family": "p10c_haidian_frozen_harbin",
        "encoder_frozen": True,
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise ValueError(f"invalid masked-export contract field {field}")
    if tuple(contract.get("disabled_highres_sources", ())) != DISABLED_HIGHRES_SOURCES:
        raise ValueError("masked-export contract must disable optical and SAR together")
    if "unmasked_embedding_as_masked_result" not in contract.get("prohibited_evidence", []):
        raise ValueError("masked-export contract must prohibit unmasked evidence")
    return contract


def mask_highres_availability(
    highres_masks: dict[str, torch.Tensor], disabled_sources: tuple[str, ...]
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, bool | float]]]:
    """Set each requested high-res availability mask to zero and record the before/after state."""
    masked = {source: value.clone() for source, value in highres_masks.items()}
    audit: dict[str, dict[str, bool | float]] = {}
    for source in disabled_sources:
        original = highres_masks.get(source)
        if original is None:
            audit[source] = {
                "present_in_batch": False,
                "available_before": False,
                "available_after": False,
                "mask_sum_before": 0.0,
                "mask_sum_after": 0.0,
            }
            continue
        before = float(original.detach().sum().item())
        masked[source] = torch.zeros_like(original)
        audit[source] = {
            "present_in_batch": True,
            "available_before": before > 0.0,
            "available_after": False,
            "mask_sum_before": before,
            "mask_sum_after": 0.0,
        }
    return masked, audit


def verify_mask_audit(audit_path: Path, patch_ids: list[str]) -> dict[str, Any]:
    """Fail closed unless every exported patch records both requested sources as unavailable."""
    payload = read_json(audit_path)
    patches = payload.get("patches")
    if not isinstance(patches, dict) or sorted(patches) != sorted(patch_ids):
        raise ValueError("masked availability audit does not cover exactly the exported patch set")
    for patch_id in patch_ids:
        record = patches[patch_id]
        if not isinstance(record, dict) or set(record) != set(DISABLED_HIGHRES_SOURCES):
            raise ValueError(f"masked availability audit lacks required sources: {patch_id}")
        for source in DISABLED_HIGHRES_SOURCES:
            source_record = record[source]
            if not isinstance(source_record, dict):
                raise ValueError(f"invalid availability record: {patch_id}/{source}")
            if (
                bool(source_record.get("available_after"))
                or float(source_record.get("mask_sum_after", 1.0)) != 0.0
            ):
                raise ValueError(
                    f"high-resolution availability remains enabled: {patch_id}/{source}"
                )
    return {
        "patch_count": len(patch_ids),
        "disabled_highres_sources": list(DISABLED_HIGHRES_SOURCES),
        "all_mask_sums_zero": True,
    }


def export_masked_embeddings(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    output_root: Path,
    month: str,
) -> tuple[list[str], dict[str, dict[str, dict[str, bool | float]]]]:
    """Run inference while retaining frames but forcing high-res fusion availability to zero."""
    output_root.mkdir(parents=True, exist_ok=False)
    produced: list[str] = []
    audit: dict[str, dict[str, dict[str, bool | float]]] = {}
    with torch.no_grad():
        for batch in loader:
            patch_ids = [str(value) for value in batch["patch_ids"]]
            if len(patch_ids) != 1:
                raise ValueError(
                    "masked export requires batch_size=1 for per-patch availability audit"
                )
            masked_masks, record = mask_highres_availability(
                batch.get("highres_masks", {}), DISABLED_HIGHRES_SOURCES
            )
            source_frames = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["source_frames"].items()
            }
            source_masks = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["source_masks"].items()
            }
            highres_frames = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.get("highres_frames", {}).items()
            }
            highres_masks = {
                key: value.to(device, non_blocking=True) for key, value in masked_masks.items()
            }
            output = model(
                source_frames=source_frames,
                source_masks=source_masks,
                timestamps=batch["timestamps"].to(device),
                highres_frames=highres_frames or None,
                highres_masks=highres_masks or None,
            )
            month_indices = (batch["timestamps"][0] == int(month)).nonzero(as_tuple=False)
            if month_indices.numel() != 1:
                raise ValueError(f"expected exactly one {month} embedding month for {patch_ids[0]}")
            embedding = (
                output.embedding_map[0, int(month_indices.item())].detach().cpu().contiguous()
            )
            if tuple(embedding.shape) != EXPECTED_SHAPE or not bool(
                torch.isfinite(embedding).all()
            ):
                raise ValueError(f"invalid masked embedding map for {patch_ids[0]}")
            patch_dir = output_root / "harbin" / patch_ids[0]
            patch_dir.mkdir(parents=True, exist_ok=False)
            torch.save(embedding, patch_dir / f"{month}_embedding_map.pt")
            produced.append(patch_ids[0])
            audit[patch_ids[0]] = record
    return sorted(produced), audit


def selected_manifest(
    contract: dict[str, Any], destination: Path, max_patches: int | None
) -> list[str]:
    source_manifest = Path(contract["source_manifest"])
    inventory = REPO_ROOT / contract["coverage_inventory"]
    patch_ids = paper_export.materialize_aef_covered_manifest(
        source_manifest, inventory, destination
    )
    if max_patches is None:
        return patch_ids
    if max_patches < 1:
        raise ValueError("max_patches must be positive")
    selected = patch_ids[:max_patches]
    records = json.loads(destination.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(records[:max_patches], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "configs/eval/harbin_p10c_highres_mask_diagnostic_20260729.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main() -> None:
    from downstreams.inference import (
        build_inference_loader,
        load_model_for_inference,
        write_meta_json,
    )

    args = parse_args()
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse masked export root: {output_root}")
    source_manifest = Path(contract["source_manifest"])
    suffix = "preflight" if args.max_patches is not None else "full"
    manifest_path = source_manifest.with_name(
        f"{source_manifest.stem}_highres_masked_{suffix}.json"
    )
    patch_ids = selected_manifest(contract, manifest_path, args.max_patches)
    model, cfg, device = load_model_for_inference(
        contract["model_config"], contract["checkpoint"], device_preference=args.device
    )
    loader = build_inference_loader(
        cfg, contract["region"], manifest_path=manifest_path, split="all"
    )
    produced, audit = export_masked_embeddings(
        model, loader, device, output_root, contract["month"]
    )
    if produced != sorted(patch_ids):
        raise ValueError("masked export patch IDs differ from its independent manifest")
    audit_path = output_root / "highres_availability_audit.json"
    audit_path.write_text(json.dumps({"patches": audit}, indent=2) + "\n", encoding="utf-8")
    mask_verification = verify_mask_audit(audit_path, patch_ids)
    embedding_verification = paper_export.verify_export_maps(
        output_root, patch_ids, contract["month"]
    )
    write_meta_json(
        output_root, Path(contract["checkpoint"]), contract_path, " ".join(sys.argv), manifest_path
    )
    provenance = {
        "schema_version": 1,
        "protocol_id": contract["protocol_id"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "selected_manifest": str(manifest_path.resolve()),
        "selected_manifest_sha256": sha256_file(manifest_path),
        "embedding_verification": embedding_verification,
        "highres_mask_verification": mask_verification,
        "highres_availability_audit": str(audit_path.resolve()),
        "highres_availability_audit_sha256": sha256_file(audit_path),
        "unmasked_embedding_as_masked_result": False,
    }
    (output_root / "highres_masked_export_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance["highres_mask_verification"], sort_keys=True))


if __name__ == "__main__":
    main()
