from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/export_p10c_harbin_paper_embeddings.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("p10c_harbin_export", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path, *, frozen: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "p10c_harbin_paper_embedding_exports",
                "region": "harbin",
                "month": "202604",
                "coverage_inventory": "coverage.json",
                "exports": {
                    "p10c_harbin_scratch": {
                        "config": "scratch.yaml",
                        "checkpoint": "scratch.pt",
                        "source_manifest": "scratch_manifest.json",
                        "device": "npu:0",
                        "encoder_frozen": False,
                    },
                    "p10c_haidian_frozen_harbin": {
                        "config": "haidian.yaml",
                        "checkpoint": "haidian.pt",
                        "source_manifest": "haidian_manifest.json",
                        "device": "npu:1",
                        "encoder_frozen": frozen,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_requires_two_distinct_devices_and_a_frozen_transfer(tmp_path: Path) -> None:
    module = _load_module()
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)

    registry = module.load_export_registry(registry_path)

    assert registry["month"] == "202604"
    assert registry["exports"]["p10c_haidian_frozen_harbin"]["encoder_frozen"] is True
    broken_path = tmp_path / "broken.json"
    _write_registry(broken_path, frozen=False)
    with pytest.raises(ValueError, match="frozen"):
        module.load_export_registry(broken_path)


def test_coverage_manifest_uses_exactly_the_aef_covered_patch_set(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "harbin_manifest.json"
    source.write_text(
        json.dumps(
            [
                {"region": "harbin", "patch_id": "harbin_patch_000001", "s2": []},
                {"region": "harbin", "patch_id": "harbin_patch_000002", "s2": []},
                {"region": "harbin", "patch_id": "harbin_patch_000003", "s2": []},
            ]
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "coverage.json"
    inventory.write_text(
        json.dumps(
            {
                "records": [
                    {"patch_id": "harbin_patch_000003"},
                    {"patch_id": "harbin_patch_000001"},
                ]
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "covered.json"

    patch_ids = module.materialize_aef_covered_manifest(source, inventory, destination)

    assert patch_ids == ["harbin_patch_000001", "harbin_patch_000003"]
    assert [item["patch_id"] for item in json.loads(destination.read_text())] == patch_ids


def test_verify_export_rejects_partial_or_nonfinite_embedding_maps(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "export"
    patch_ids = ["harbin_patch_000001", "harbin_patch_000002"]
    for patch_id in patch_ids:
        path = root / "harbin" / patch_id
        path.mkdir(parents=True)
        torch.save(torch.ones((64, 128, 128)), path / "202604_embedding_map.pt")

    with pytest.raises(ValueError, match="zero-variance"):
        module.verify_export_maps(root, patch_ids, "202604")

    spatial_pattern = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    torch.save(
        torch.stack([spatial_pattern * float(index + 1) for index in range(64)]),
        root / "harbin" / patch_ids[0] / "202604_embedding_map.pt",
    )
    torch.save(
        torch.stack([spatial_pattern * float(index + 2) for index in range(64)]),
        root / "harbin" / patch_ids[1] / "202604_embedding_map.pt",
    )
    report = module.verify_export_maps(root, patch_ids, "202604")
    assert report["patch_count"] == 2
    assert report["shape"] == [64, 128, 128]
