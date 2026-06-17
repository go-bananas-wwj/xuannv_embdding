from __future__ import annotations

# 冒烟测试：验证包导入、配置加载、设备选择、端到端模型前向与训练入口。
import importlib.util
import sys
import types
from pathlib import Path

import torch

import xuannv_embedding
from xuannv_embedding.config import Config
from xuannv_embedding.models.model import AEFModel, AEFOutput
from xuannv_embedding.utils.device import get_device


def _import_train_module() -> types.ModuleType:
    """通过 importlib 动态导入 scripts/train/train.py，避免包路径依赖。"""
    train_path = Path(__file__).parent.parent / "scripts" / "train" / "train.py"
    spec = importlib.util.spec_from_file_location("scripts.train.train", str(train_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载训练脚本: {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts.train.train"] = module
    spec.loader.exec_module(module)
    return module


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
  statistics_dir: /data/xuannv_embedding/statistics/base_region
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


def test_get_device_with_preference() -> None:
    """``preference`` 参数应被直接尊重。"""
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("npu:0") == torch.device("npu:0")
    assert get_device("cuda:0") == torch.device("cuda:0")


def test_get_device_fallback_no_npu(monkeypatch) -> None:
    """无 preference 且 NPU 不可用时，应正确回退到 CUDA 或 CPU。"""
    # 模拟 torch_npu 未安装
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    device = get_device()
    if torch.cuda.is_available():
        assert device == torch.device("cuda:0")
    else:
        assert device == torch.device("cpu")


def test_get_device_invalid_preference() -> None:
    """非法 preference 应抛出 RuntimeError。"""
    try:
        get_device("not_a_device")
    except RuntimeError:
        return
    raise AssertionError("非法设备字符串应抛出 RuntimeError")


def test_model_forward_from_config() -> None:
    """从真实配置文件构造 AEFModel 并完成一次前向传播。"""
    cfg = Config.from_yaml(Path(__file__).parent.parent / "configs" / "smoke.yaml")

    aef_target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=aef_target_heads,
    )

    batch_size = 2
    time_steps = 2
    height, width = 16, 16

    source_frames: dict[str, torch.Tensor] = {}
    source_masks: dict[str, torch.Tensor] = {}
    for source in cfg.data.sources:
        channels = cfg.model.sensor_channels[source]
        source_frames[source] = torch.randn(batch_size, time_steps, channels, height, width)
        source_masks[source] = torch.ones(batch_size, time_steps)

    timestamps = torch.arange(time_steps).float().unsqueeze(0).expand(batch_size, -1)

    highres_frame: torch.Tensor | None = None
    highres_mask: torch.Tensor | None = None
    if "highres" in cfg.model.sensor_channels:
        highres_frame = torch.randn(batch_size, cfg.model.sensor_channels["highres"], height, width)
        highres_mask = torch.ones(batch_size, 1, height, width)

    output = model(
        source_frames=source_frames,
        source_masks=source_masks,
        timestamps=timestamps,
        highres_frame=highres_frame,
        highres_mask=highres_mask,
    )

    assert isinstance(output, AEFOutput)
    assert output.embedding.shape == (batch_size, cfg.model.embed_dim)
    assert output.embedding_map.shape == (batch_size, cfg.model.embed_dim, height, width)
    assert set(output.reconstructions.keys()) == set(cfg.model.target_heads.keys())


def test_train_entry_argparse() -> None:
    """训练入口参数解析应能被测试导入并正确解析。"""
    train_module = _import_train_module()
    parse_args = train_module.parse_args

    args = parse_args(["--config", "configs/smoke.yaml"])
    assert args.config == "configs/smoke.yaml"
    assert args.resume is None
    assert args.device is None

    args = parse_args(["--config", "configs/smoke.yaml", "--resume", "foo.pt"])
    assert args.config == "configs/smoke.yaml"
    assert args.resume == "foo.pt"
