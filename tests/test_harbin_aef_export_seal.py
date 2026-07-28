from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/export_harbin_aef_annual_embeddings.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_aef_export", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seal_export_hashes_every_complete_annual_map(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "aef"
    patch_ids = ["harbin_patch_000001", "harbin_patch_000002"]
    for index, patch_id in enumerate(patch_ids):
        destination = root / "harbin" / patch_id
        destination.mkdir(parents=True)
        torch.save(
            torch.full((64, 128, 128), float(index + 1)),
            destination / "annual_2025_embedding_map.pt",
        )
    coverage = tmp_path / "coverage.json"
    cog_lock = tmp_path / "cog_lock.json"
    coverage.write_text(json.dumps({"records": [{"patch_id": item} for item in patch_ids]}))
    cog_lock.write_text(json.dumps({"assets": []}))

    meta = module.seal_export(root, coverage, cog_lock, patch_ids)

    index = json.loads((root / "embedding_file_index.json").read_text())
    assert meta["patch_count"] == 2
    assert index["patch_count"] == 2
    assert [entry["patch_id"] for entry in index["files"]] == patch_ids
    assert meta["embedding_file_index"]["sha256"] == module.sha256_file(
        root / "embedding_file_index.json"
    )


def test_seal_export_rejects_partial_map_set(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "aef"
    destination = root / "harbin" / "harbin_patch_000001"
    destination.mkdir(parents=True)
    torch.save(torch.ones((64, 128, 128)), destination / "annual_2025_embedding_map.pt")
    coverage = tmp_path / "coverage.json"
    cog_lock = tmp_path / "cog_lock.json"
    coverage.write_text("{}")
    cog_lock.write_text("{}")

    with pytest.raises(ValueError, match="map set"):
        module.seal_export(
            root,
            coverage,
            cog_lock,
            ["harbin_patch_000001", "harbin_patch_000002"],
        )
