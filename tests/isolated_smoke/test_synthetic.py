from __future__ import annotations

import torch

from experiments.china_v1_fusion_smoke.synthetic import generate_synthetic_context


def test_synthetic_context_is_deterministic_and_aligned() -> None:
    """同一 patch-year 和 seed 必须生成可复现、网格对齐的合成上下文。"""
    s2 = torch.linspace(0.0, 1.0, steps=4 * 3 * 10 * 128 * 128, dtype=torch.float32).reshape(
        4, 3, 10, 128, 128
    )
    valid_s2 = torch.ones((4, 3, 1, 128, 128), dtype=torch.bool)

    first = generate_synthetic_context(s2, valid_s2, "patch-a", 2020, seed=20260815)
    second = generate_synthetic_context(s2, valid_s2, "patch-a", 2020, seed=20260815)

    torch.testing.assert_close(first.aef, second.aef)
    torch.testing.assert_close(first.highres, second.highres)
    assert first.aef.shape == (64, 128, 128)
    assert first.aef_valid.shape == (1, 128, 128)
    assert first.highres.shape == (3, 640, 640)
    assert first.highres_valid.shape == (1, 640, 640)
    norms = torch.linalg.vector_norm(first.aef[:, first.aef_valid[0]], dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_synthetic_context_metadata_disclaims_official_aef_and_real_2m() -> None:
    """合成产物元数据必须明确限制为 smoke，且不得声称真实 AEF/2 m 信息。"""
    s2 = torch.full((4, 3, 10, 128, 128), 0.25, dtype=torch.float32)
    valid_s2 = torch.ones((4, 3, 1, 128, 128), dtype=torch.bool)

    context = generate_synthetic_context(s2, valid_s2, "patch-a", 2020, seed=20260815)

    assert context.metadata["aef"] == {
        "synthetic": True,
        "synthetic_kind": "annual_s2_fixed_projection",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    assert context.metadata["highres"] == {
        "synthetic": True,
        "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
        "claimed_native_gsd_m": None,
        "model_input_gsd_m": 2,
        "contains_real_2m_information": False,
    }
