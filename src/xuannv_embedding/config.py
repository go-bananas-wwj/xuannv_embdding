from __future__ import annotations

# 基于 dataclass 的 YAML 配置加载，支持 ``_base_`` 继承。
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置解析或校验失败的自定义异常。"""


@dataclass
class DataConfig:
    """数据相关配置。"""

    root: Path
    region: str
    manifest_path: Path
    num_samples: int
    statistics_dir: Path | None = None
    max_patches: int | None = None
    batch_size: int = 4
    num_workers: int = 4
    patch_size: int = 256
    sources: list[str] = field(default_factory=lambda: ["s2", "s1", "landsat"])


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
    target_heads: dict[str, dict[str, Any]]


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

        异常:
            ConfigError: YAML 解析结果非法或必填字段缺失。
        """
        raw = _load_yaml_with_base(Path(path))
        _validate_yaml_is_dict(raw, path)

        try:
            data_cfg = raw["data"]
            experiment_cfg = raw["experiment"]
            model_cfg = raw["model"]
            training_cfg = raw["training"]
        except KeyError as exc:
            key = exc.args[0]
            raise ConfigError(f"配置文件 {path} 缺少必填顶层字段: {key}") from exc

        _validate_required_keys(
            data_cfg, ["root", "region", "manifest_path", "num_samples"], path, "data"
        )
        _validate_required_keys(
            model_cfg, ["embed_dim", "sensor_channels", "target_heads"], path, "model"
        )
        _validate_required_keys(
            training_cfg,
            [
                "epochs",
                "lr",
                "weight_decay",
                "warmup_epochs",
                "gradient_accumulation_steps",
                "save_every",
                "eval_every",
            ],
            path,
            "training",
        )
        _validate_required_keys(experiment_cfg, ["name"], path, "experiment")

        root = Path(data_cfg["root"])
        region = data_cfg["region"]
        statistics_dir = data_cfg.get("statistics_dir")
        if statistics_dir is None:
            # 约定 root 为 processed/{region}/scenes，统计量位于数据根目录 statistics/{region}
            statistics_dir = root.parent.parent.parent / "statistics" / region
        else:
            statistics_dir = Path(statistics_dir)

        return cls(
            data=DataConfig(
                root=root,
                region=region,
                manifest_path=Path(data_cfg["manifest_path"]),
                num_samples=data_cfg["num_samples"],
                statistics_dir=statistics_dir,
                max_patches=data_cfg.get("max_patches"),
                batch_size=data_cfg.get("batch_size", 4),
                num_workers=data_cfg.get("num_workers", 4),
                patch_size=data_cfg.get("patch_size", 256),
                sources=data_cfg.get("sources", ["s2", "s1", "landsat"]),
            ),
            experiment=ExperimentConfig(
                name=experiment_cfg["name"],
                seed=experiment_cfg.get("seed", 42),
            ),
            model=ModelConfig(
                embed_dim=model_cfg["embed_dim"],
                sensor_channels=model_cfg["sensor_channels"],
                target_heads=model_cfg["target_heads"],
            ),
            training=TrainingConfig(
                epochs=training_cfg["epochs"],
                lr=training_cfg["lr"],
                weight_decay=training_cfg["weight_decay"],
                warmup_epochs=training_cfg["warmup_epochs"],
                gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
                save_every=training_cfg["save_every"],
                eval_every=training_cfg["eval_every"],
            ),
        )


def _validate_yaml_is_dict(raw: Any, path: str | Path) -> None:
    """校验 YAML 解析结果是否为字典。"""
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件 {path} 必须是 YAML mapping，实际得到 {type(raw).__name__}")


def _validate_required_keys(
    section: Any, keys: list[str], path: str | Path, section_name: str
) -> None:
    """校验配置段是否为字典并包含所有必填字段。"""
    if not isinstance(section, dict):
        actual_type = type(section).__name__
        raise ConfigError(
            f"配置文件 {path} 中的 `{section_name}` 必须是 mapping，实际得到 {actual_type}"
        )
    for key in keys:
        if key not in section:
            raise ConfigError(f"配置文件 {path} 的 `{section_name}` 缺少必填字段: {key}")


def _load_yaml_with_base(path: Path, loaded: set[str] | None = None) -> dict[str, Any]:
    """加载 YAML，并在存在 ``_base_`` 字段时递归合并父配置。

    参数:
        path: 当前待加载的 YAML 文件路径。
        loaded: 已加载文件路径集合，用于检测循环引用。

    返回:
        合并后的配置字典。

    异常:
        ConfigError: 文件不存在、YAML 非法或出现循环继承。
    """
    if loaded is None:
        loaded = set()

    abs_path = path.resolve()
    path_str = str(abs_path)
    if path_str in loaded:
        raise ConfigError(f"检测到 `_base_` 循环引用: {path_str}")
    loaded.add(path_str)

    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        config: Any = yaml.safe_load(f)

    _validate_yaml_is_dict(config, path)

    base_name = config.pop("_base_", None)
    if base_name is None:
        return config

    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parent / base_path

    base_config = _load_yaml_with_base(base_path, loaded)
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
