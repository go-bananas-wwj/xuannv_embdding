from __future__ import annotations

"""冒烟测试：验证包导入、配置加载和设备选择。"""

from pathlib import Path

import pytest
import torch

import xuannv_embedding
from xuannv_embedding.config import Config
from xuannv_embedding.utils.device import get_device


def test_import_package() -> None:
    """包版本号应符合预期。"""
    assert xuannv_embedding.__version__ == "0.1.0"


def test_config_load(tmp_path: Path) -> None:
    """基于 ``_base_`` 的 YAML 继承与字段覆盖应正常工作。"""
    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"

    base.write_text(
        """
experiment:
  name: base_experiment
  seed: 42
data:
  root: /data/xuannv_embedding/processed/base
  region: base_region
  manifest_path: /data/xuannv_embedding/processed/base/manifest.json
  num_samples: 100
  batch_size: 4
model:
  embed_dim: 128
  sensor_channels:
    s2: 13
    s1: 2
  target_heads:
    rgb:
      bands: [2, 1, 0]
training:
  epochs: 10
  lr: 0.001
  weight_decay: 0.01
  warmup_epochs: 1
  gradient_accumulation_steps: 1
  save_every: 2
  eval_every: 2
""",
        encoding="utf-8",
    )

    derived.write_text(
        """
_base_: base.yaml
experiment:
  name: derived_experiment
data:
  region: harbin
  num_samples: 424
  sources:
    - s2
    - s1
    - landsat
""",
        encoding="utf-8",
    )

    cfg = Config.from_yaml(derived)

    assert cfg.experiment.name == "derived_experiment"
    assert cfg.experiment.seed == 42
    assert cfg.data.root == Path("/data/xuannv_embedding/processed/base")
    assert cfg.data.region == "harbin"
    assert cfg.data.num_samples == 424
    assert cfg.data.batch_size == 4
    assert cfg.data.sources == ["s2", "s1", "landsat"]
    assert cfg.model.embed_dim == 128
    assert cfg.model.sensor_channels["s1"] == 2


def test_get_device() -> None:
    """``preference`` 参数应被直接尊重。"""
    device = get_device("cpu")
    assert device == torch.device("cpu")
