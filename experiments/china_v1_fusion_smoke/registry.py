from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RegistryError(ValueError):
    """registry 不符合隔离 smoke 或正式数据用途策略时抛出。"""


_SMOKE_REQUIREMENTS = {
    "synthetic": True,
    "allowed_use": "smoke_test_only",
    "formal_training_allowed": False,
    "formal_evaluation_allowed": False,
}
_HIGHRES_KIND = "annual_s2_rgb_5x_deterministic_texture"
_HIGHRES_REQUIREMENTS = {
    "claimed_native_gsd_m": None,
    "model_input_gsd_m": 2,
    "contains_real_2m_information": False,
}


def _has_exact_value(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _require_exact(raw: Mapping[str, Any], requirements: Mapping[str, Any]) -> None:
    for key, expected in requirements.items():
        if key not in raw or not _has_exact_value(raw[key], expected):
            raise RegistryError(f"registry field {key!r} must equal {expected!r}")


def validate_smoke_registry(raw: Mapping[str, Any]) -> None:
    """只允许具有不可放宽 synthetic 限制的 smoke registry。"""
    if not isinstance(raw, Mapping):
        raise RegistryError("registry must be a mapping")
    _require_exact(raw, _SMOKE_REQUIREMENTS)
    if raw.get("synthetic_kind") == _HIGHRES_KIND:
        _require_exact(raw, _HIGHRES_REQUIREMENTS)


def validate_formal_registry(raw: Mapping[str, Any]) -> None:
    """拒绝一切 synthetic registry，防止 smoke 产物进入正式流程。"""
    if not isinstance(raw, Mapping):
        raise RegistryError("registry must be a mapping")
    if "synthetic" not in raw or type(raw["synthetic"]) is not bool:
        raise RegistryError("formal registry requires a boolean synthetic field")
    if raw["synthetic"] is True:
        raise RegistryError("synthetic data is forbidden in formal registries")
