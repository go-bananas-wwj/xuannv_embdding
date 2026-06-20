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
    num_samples: int | None = None  # 仅作为元数据，实际样本数由 dataset 长度决定
    statistics_dir: Path | None = None
    max_patches: int | None = None
    batch_size: int = 4
    num_workers: int = 8
    patch_size: int = 256
    num_months: int = 17
    months: list[str] = field(default_factory=lambda: [])
    sources: list[str] = field(default_factory=lambda: ["s2", "s1", "landsat"])


@dataclass
class ExperimentConfig:
    """实验元信息配置。"""

    name: str
    seed: int = 42
    output_dir: Path | None = None
    use_wandb: bool = True
    wandb_project: str = "xuannv-embedding-stage1"
    wandb_run_name: str | None = None


@dataclass
class ModelConfig:
    """模型结构配置。"""

    embed_dim: int
    sensor_channels: dict[str, int]
    target_heads: dict[str, dict[str, Any]]
    stem_dim: int = 32
    num_months: int = 17
    stp: dict[str, Any] = field(default_factory=dict)


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
    amp: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 0


@dataclass
class Config:
    """顶层配置，聚合数据、实验、模型、训练配置。"""

    data: DataConfig
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        """将配置递归转换为字典，便于 WANDB 等外部工具记录。"""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        result: dict[str, Any] = {}
        for section_name in ("data", "experiment", "model", "training"):
            section = getattr(self, section_name)
            result[section_name] = {
                field.name: _convert(getattr(section, field.name))
                for field in section.__dataclass_fields__.values()
            }
        return result

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

        _validate_required_keys(data_cfg, ["root", "region", "manifest_path"], path, "data")
        _validate_required_keys(
            model_cfg, ["embed_dim", "sensor_channels", "target_heads"], path, "model"
        )

        stp_cfg = model_cfg.get("stp", {})
        stp_defaults = {
            "space_dim": 512,
            "time_dim": 256,
            "precision_dim": 128,
            "num_blocks": 6,
            "num_heads": 8,
        }
        for key, value in stp_defaults.items():
            stp_cfg.setdefault(key, value)
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

        data_num_months = data_cfg.get("num_months", 17)
        model_num_months = model_cfg.get("num_months", 17)
        if data_num_months != model_num_months:
            raise ConfigError(
                f"配置 {path} 中 data.num_months ({data_num_months}) 与 "
                f"model.num_months ({model_num_months}) 不一致，必须相等"
            )

        root = Path(data_cfg["root"])
        region = data_cfg["region"]
        statistics_dir = data_cfg.get("statistics_dir")
        if statistics_dir is None:
            # 约定 root 为 processed/{region}，统计量位于数据根目录 statistics/{region}
            statistics_dir = root.parent / "statistics" / region
        else:
            statistics_dir = Path(statistics_dir)

        return cls(
            data=DataConfig(
                root=root,
                region=region,
                manifest_path=Path(data_cfg["manifest_path"]),
                num_samples=data_cfg.get("num_samples"),
                statistics_dir=statistics_dir,
                max_patches=data_cfg.get("max_patches"),
                batch_size=data_cfg.get("batch_size", 4),
                num_workers=data_cfg.get("num_workers", 8),
                patch_size=data_cfg.get("patch_size", 256),
                num_months=data_cfg.get("num_months", 17),
                months=data_cfg.get("months", []),
                sources=data_cfg.get("sources", ["s2", "s1", "landsat"]),
            ),
            experiment=ExperimentConfig(
                name=experiment_cfg["name"],
                seed=experiment_cfg.get("seed", 42),
                output_dir=(
                    Path(experiment_cfg["output_dir"]) if experiment_cfg.get("output_dir") else None
                ),
                use_wandb=experiment_cfg.get("use_wandb", True),
                wandb_project=experiment_cfg.get("wandb_project", "xuannv-embedding-stage1"),
                wandb_run_name=experiment_cfg.get("wandb_run_name"),
            ),
            model=ModelConfig(
                embed_dim=model_cfg["embed_dim"],
                sensor_channels=model_cfg["sensor_channels"],
                target_heads=model_cfg["target_heads"],
                stem_dim=model_cfg.get("stem_dim", 32),
                num_months=model_cfg.get("num_months", 17),
                stp=stp_cfg,
            ),
            training=TrainingConfig(
                epochs=training_cfg["epochs"],
                lr=training_cfg["lr"],
                weight_decay=training_cfg["weight_decay"],
                warmup_epochs=training_cfg["warmup_epochs"],
                gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
                save_every=training_cfg["save_every"],
                eval_every=training_cfg["eval_every"],
                amp=training_cfg.get("amp", True),
                gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
                log_every=training_cfg.get("log_every", 0),
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
