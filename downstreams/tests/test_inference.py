from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from downstreams.inference import precompute_embeddings, write_meta_json


class _SimpleOutput:
    def __init__(self, embedding_map: torch.Tensor, embedding: torch.Tensor) -> None:
        self.embedding_map = embedding_map
        self.embedding = embedding


class _FakeModel:
    def __init__(self, embedding_map: torch.Tensor, embedding: torch.Tensor) -> None:
        self._embedding_map = embedding_map
        self._embedding = embedding

    def __call__(self, **kwargs: Any) -> _SimpleOutput:
        return _SimpleOutput(self._embedding_map, self._embedding)


class _OneBatchLoader:
    def __init__(self, batch: dict[str, Any]) -> None:
        self._batch = batch

    def __iter__(self):
        yield self._batch


def test_write_meta_json_random_init(tmp_path: Path) -> None:
    out_dir = tmp_path / "embeddings"
    out_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: config\n")

    write_meta_json(
        output_dir=out_dir,
        checkpoint_path=None,
        config_path=config_path,
        command_line="python precompute_embeddings.py --random-init",
    )

    meta_path = out_dir / "meta.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["checkpoint_path"] is None
    assert meta["checkpoint_sha256"] is None
    assert meta["config_path"] == str(config_path)


def test_write_meta_json_with_checkpoint(tmp_path: Path) -> None:
    out_dir = tmp_path / "embeddings"
    out_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: config\n")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"not a real checkpoint")

    write_meta_json(
        output_dir=out_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        command_line="python precompute_embeddings.py --checkpoint best.pt",
    )

    meta_path = out_dir / "meta.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["checkpoint_path"] == str(checkpoint_path)
    assert meta["checkpoint_sha256"] is not None
    assert len(meta["checkpoint_sha256"]) == 16


def test_precompute_embeddings_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    loader = _OneBatchLoader(
        {
            "patch_ids": ["patch_000001"],
            "timestamps": torch.tensor([[202604]]),
            "source_frames": {"s2": torch.randn(1, 1, 4, 8, 8)},
            "source_masks": {"s2": torch.ones(1, 1, 8, 8)},
        }
    )
    emb_map = torch.randn(1, 1, 8, 8, 8)
    scene_emb = torch.randn(1, 1, 8)
    model = _FakeModel(emb_map, scene_emb)

    precompute_embeddings(model, loader, torch.device("cpu"), output_dir)

    patch_dir = output_dir / "patch_000001"
    assert (patch_dir / "202604_embedding_map.pt").exists()
    assert (patch_dir / "202604_scene_embedding.pt").exists()
