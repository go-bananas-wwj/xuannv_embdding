from __future__ import annotations

import json
from pathlib import Path

from downstreams.inference import write_meta_json


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
