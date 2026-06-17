from __future__ import annotations

"""基于 dataclass 的 YAML 配置加载，支持 ``_base_`` 继承。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """数据相关配置。"""

    root: Path
    region: str
    manifest_path: Path
    num_samples: int
    max_patches: int | None = None
    batch_size: int = 4
    num_workers: int = 4
    sources: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sources is None:
            self.sources = ["s2", "s1", "landsat"]


@dataclass
class ExperimentConfig:
    """实验元信息配置。"""

    name: str
    seed: int = 42


@dataclass
class ModelConfig:
    """模型结构配置。"""

    embed_dim: int
    sensor_channels: dict[str, int]
    target_heads: dict[str, dict]


@dataclass
class TrainingConfig:
    """训练超参数配置。"""

    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    gradient_accumulation_steps: int
    save_every: int
    eval_every: int


@dataclass
class Config:
    """顶层配置，聚合数据、实验、模型、训练配置。"""

    data: DataConfig
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """从 YAML 文件加载配置，支持 ``_base_`` 继承。

        参数:
            path: YAML 文件路径，相对路径或绝对路径均可。

        返回:
            解析后的 ``Config`` 实例。
        """
        raw = _load_yaml_with_base(Path(path))
        return cls(
            data=DataConfig(
                root=Path(raw["data"]["root"]),
                region=raw["data"]["region"],
                manifest_path=Path(raw["data"]["manifest_path"]),
                num_samples=raw["data"]["num_samples"],
                max_patches=raw["data"].get("max_patches"),
                batch_size=raw["data"].get("batch_size", 4),
                num_workers=raw["data"].get("num_workers", 4),
                sources=raw["data"].get("sources", ["s2", "s1", "landsat"]),
            ),
            experiment=ExperimentConfig(
                name=raw["experiment"]["name"],
                seed=raw["experiment"].get("seed", 42),
            ),
            model=ModelConfig(
                embed_dim=raw["model"]["embed_dim"],
                sensor_channels=raw["model"]["sensor_channels"],
                target_heads=raw["model"]["target_heads"],
            ),
            training=TrainingConfig(
                epochs=raw["training"]["epochs"],
                lr=raw["training"]["lr"],
                weight_decay=raw["training"]["weight_decay"],
                warmup_epochs=raw["training"]["warmup_epochs"],
                gradient_accumulation_steps=raw["training"]["gradient_accumulation_steps"],
                save_every=raw["training"]["save_every"],
                eval_every=raw["training"]["eval_every"],
            ),
        )


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """加载 YAML，并在存在 ``_base_`` 字段时递归合并父配置。"""
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    base_name = config.pop("_base_", None)
    if base_name is None:
        return config

    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parent / base_path

    base_config = _load_yaml_with_base(base_path)
    merged = _deep_merge(base_config, config)
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，``override`` 优先级更高。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
