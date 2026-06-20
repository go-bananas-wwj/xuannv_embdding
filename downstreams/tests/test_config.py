from __future__ import annotations

from pathlib import Path

from downstreams.utils.config import load_config


def test_load_config_deep_merge(tmp_path: Path) -> None:
    base = tmp_path / "_base_.yaml"
    base.write_text(
        "experiment:\n  name: base\n  seed: 42\n\ntraining:\n  epochs: 100\n  lr: 1.0e-3\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "_base_: _base_.yaml\n\nexperiment:\n  name: child\n\ntraining:\n  lr: 1.0e-4\n",
        encoding="utf-8",
    )

    cfg = load_config(child)
    assert cfg["experiment"]["name"] == "child"
    assert cfg["experiment"]["seed"] == 42  # 从 base 继承
    assert cfg["training"]["epochs"] == 100  # 从 base 继承
    assert cfg["training"]["lr"] == 1.0e-4  # 被 child 覆盖
