from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SANDBOX_ROOT = Path("/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815")
EXPECTED_SOURCE_ROOT = Path("/data2/china_xuannv_embedding/data")
EXPECTED_SECTIONS = {"experiment", "sandbox", "runtime", "data", "model", "export"}


@dataclass(frozen=True)
class SmokeConfig:
    sandbox_root: Path
    source_root: Path
    sentinel: str
    physical_npu: int
    logical_device: str
    max_patches: int
    years: tuple[int, int]
    max_optimizer_steps: int
    num_workers: int
    s2_scale: float
    s1_epsilon: float
    s1_source_order: tuple[str, str]
    s1_model_order: tuple[str, str]
    embed_dim: int
    aef_hidden_dim: int
    highres_hidden_dim: int
    gate_init: float
    gradient_smoke_gate: float
    export_dtype: str
    zarr_version: int
    periods: tuple[str, ...]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(section: dict[str, Any], name: str, keys: set[str]) -> None:
    actual = set(section)
    if actual != keys:
        raise ValueError(f"{name} keys must be exactly {sorted(keys)}; got {sorted(actual)}")


def _as_pair(value: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return value[0], value[1]


def load_smoke_config(path: Path) -> SmokeConfig:
    """加载并严格验证 China V1 隔离融合 smoke test 配置。"""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "config")
    if "_base_" in root:
        raise ValueError("_base_ inheritance is not allowed")
    if set(root) != EXPECTED_SECTIONS:
        raise ValueError(f"config sections must be exactly {sorted(EXPECTED_SECTIONS)}")

    experiment = _mapping(root["experiment"], "experiment")
    sandbox = _mapping(root["sandbox"], "sandbox")
    runtime = _mapping(root["runtime"], "runtime")
    data = _mapping(root["data"], "data")
    model = _mapping(root["model"], "model")
    export = _mapping(root["export"], "export")
    _require_exact_keys(experiment, "experiment", {"name", "seed", "use_wandb"})
    _require_exact_keys(sandbox, "sandbox", {"root", "source_root", "sentinel"})
    _require_exact_keys(
        runtime,
        "runtime",
        {
            "physical_npu",
            "logical_device",
            "max_patches",
            "years",
            "max_optimizer_steps",
            "num_workers",
        },
    )
    _require_exact_keys(
        data,
        "data",
        {"s2_scale", "s1_epsilon", "s1_source_order", "s1_model_order"},
    )
    _require_exact_keys(
        model,
        "model",
        {
            "embed_dim",
            "aef_hidden_dim",
            "highres_hidden_dim",
            "gate_init",
            "gradient_smoke_gate",
        },
    )
    _require_exact_keys(export, "export", {"dtype", "zarr_version", "periods"})

    years = _as_pair(runtime["years"], "runtime.years")
    source_order = _as_pair(data["s1_source_order"], "data.s1_source_order")
    model_order = _as_pair(data["s1_model_order"], "data.s1_model_order")
    config = SmokeConfig(
        sandbox_root=Path(sandbox["root"]),
        source_root=Path(sandbox["source_root"]),
        sentinel=sandbox["sentinel"],
        physical_npu=runtime["physical_npu"],
        logical_device=runtime["logical_device"],
        max_patches=runtime["max_patches"],
        years=(years[0], years[1]),
        max_optimizer_steps=runtime["max_optimizer_steps"],
        num_workers=runtime["num_workers"],
        s2_scale=data["s2_scale"],
        s1_epsilon=data["s1_epsilon"],
        s1_source_order=(source_order[0], source_order[1]),
        s1_model_order=(model_order[0], model_order[1]),
        embed_dim=model["embed_dim"],
        aef_hidden_dim=model["aef_hidden_dim"],
        highres_hidden_dim=model["highres_hidden_dim"],
        gate_init=model["gate_init"],
        gradient_smoke_gate=model["gradient_smoke_gate"],
        export_dtype=export["dtype"],
        zarr_version=export["zarr_version"],
        periods=tuple(export["periods"]),
    )
    validate_limits(config)
    return config


def validate_limits(config: SmokeConfig) -> None:
    """拒绝任何会扩大该 smoke test 资源或数据边界的配置。"""
    if config.sandbox_root != EXPECTED_SANDBOX_ROOT:
        raise ValueError("sandbox root must be the exact isolated smoke root")
    if config.source_root != EXPECTED_SOURCE_ROOT:
        raise ValueError("source root must be the exact China V1 source root")
    if config.sentinel != ".xuannv_isolated_smoke":
        raise ValueError("sandbox sentinel is invalid")
    if config.physical_npu != 2 or config.logical_device != "npu:0":
        raise ValueError("smoke test requires exactly two physical NPUs and npu:0")
    if config.max_patches != 4:
        raise ValueError("smoke test must use exactly four patches")
    if config.years != (2020, 2021):
        raise ValueError("smoke test must use years 2020 and 2021")
    if config.max_optimizer_steps != 2 or config.num_workers != 0:
        raise ValueError("smoke test runtime limits are invalid")
    if config.s2_scale != 0.0001 or config.s1_epsilon != 1.0e-6:
        raise ValueError("smoke test data scaling is invalid")
    if config.s1_source_order != ("vh", "vv") or config.s1_model_order != ("vv", "vh"):
        raise ValueError("smoke test S1 channel order is invalid")
    if config.export_dtype != "float16" or config.zarr_version != 2:
        raise ValueError("smoke export format is invalid")
