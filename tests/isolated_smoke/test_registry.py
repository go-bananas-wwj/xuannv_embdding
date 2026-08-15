from __future__ import annotations

import pytest

from experiments.china_v1_fusion_smoke.registry import (
    RegistryError,
    validate_formal_registry,
    validate_smoke_registry,
)


def test_formal_registry_rejects_synthetic_context() -> None:
    raw = {
        "synthetic": True,
        "synthetic_kind": "annual_s2_fixed_projection",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }

    validate_smoke_registry(raw)
    with pytest.raises(RegistryError, match="synthetic data is forbidden"):
        validate_formal_registry(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"synthetic": 1},
        {"synthetic": False},
        {
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": 0,
            "formal_evaluation_allowed": False,
        },
    ],
)
def test_smoke_registry_rejects_nonexact_synthetic_restrictions(raw: dict[str, object]) -> None:
    """注册表策略不能通过真值转换放宽 synthetic 限制。"""
    with pytest.raises(RegistryError):
        validate_smoke_registry(raw)


def test_highres_smoke_registry_requires_explicit_no_real_2m_claim() -> None:
    raw = {
        "synthetic": True,
        "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
        "claimed_native_gsd_m": None,
        "model_input_gsd_m": 2,
        "contains_real_2m_information": False,
    }

    validate_smoke_registry(raw)
    for key, incorrect in (
        ("claimed_native_gsd_m", 2),
        ("contains_real_2m_information", True),
    ):
        malformed = dict(raw)
        malformed[key] = incorrect
        with pytest.raises(RegistryError):
            validate_smoke_registry(malformed)


@pytest.mark.parametrize(
    "raw",
    [
        {
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
        {
            "synthetic": True,
            "synthetic_kind": "unrecognized_side_input",
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
        {
            "synthetic": True,
            "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "model_input_gsd_m": 2,
            "contains_real_2m_information": False,
        },
    ],
    ids=("missing-kind", "unknown-kind", "highres-missing-native-gsd-disclaimer"),
)
def test_smoke_registry_requires_a_recognized_kind_and_all_disclaimers(
    raw: dict[str, object],
) -> None:
    with pytest.raises(RegistryError):
        validate_smoke_registry(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"synthetic": False, "allowed_use": "smoke_test_only"},
        {"synthetic": False, "formal_training_allowed": False},
        {"synthetic": False, "formal_evaluation_allowed": False},
        {
            "synthetic": False,
            "synthetic_kind": "annual_s2_fixed_projection",
        },
    ],
    ids=("smoke-use", "training-forbidden", "evaluation-forbidden", "synthetic-kind"),
)
def test_formal_registry_rejects_smoke_only_declarations_with_false_synthetic_bit(
    raw: dict[str, object],
) -> None:
    with pytest.raises(RegistryError, match="smoke-only"):
        validate_formal_registry(raw)
