from __future__ import annotations

import pytest
import torch

from experiments.china_v1_fusion_smoke import model as model_module
from experiments.china_v1_fusion_smoke.model import IsolatedFusionSmokeModel


def _small_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260815)
    return {
        "s2": torch.randn((2, 4, 3, 10, 16, 16), generator=generator),
        "s1": torch.randn((2, 4, 3, 2, 16, 16), generator=generator),
        "valid_s2": torch.ones((2, 4, 3, 1, 16, 16), dtype=torch.bool),
        "valid_s1": torch.ones((2, 4, 3, 1, 16, 16), dtype=torch.bool),
        "aef": torch.randn((2, 64, 16, 16), generator=generator),
        "aef_valid": torch.ones((2, 1, 16, 16), dtype=torch.bool),
        "highres": torch.randn((2, 3, 80, 80), generator=generator),
        "highres_valid": torch.ones((2, 1, 80, 80), dtype=torch.bool),
    }


def _run_small_model(*, use_aef: bool, use_highres: bool):
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()
    return model(
        **inputs,
        use_aef=use_aef,
        use_highres=use_highres,
    )


@pytest.mark.parametrize(
    ("use_aef", "use_highres"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_all_branches_keep_the_output_contract(use_aef: bool, use_highres: bool) -> None:
    """任一旁路组合都必须保留季度布局和逐像素 vMF 单位范数。"""
    output = _run_small_model(use_aef=use_aef, use_highres=use_highres)

    assert output.embedding.shape == (2, 4, 64, 16, 16)
    norms = torch.linalg.vector_norm(output.embedding, dim=2)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_zero_initialized_full_matches_base() -> None:
    """零 gate 必须使完整模型可精确回退到同一基座。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()

    base = model(**inputs, use_aef=False, use_highres=False)
    full = model(**inputs, use_aef=True, use_highres=True)

    torch.testing.assert_close(full.pre_vmf, base.pre_vmf, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full.embedding, base.embedding, atol=1e-6, rtol=1e-6)
    assert full.gates["aef"].item() == 0.0
    assert full.gates["highres"].item() == 0.0


def test_bfloat16_autocast_zero_gate_full_matches_base_dtype_and_values() -> None:
    """AMP 下零 gate 旁路不得把 BF16 基座提升到 FP32 或改变归一化结果。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        base = model(**inputs, use_aef=False, use_highres=False)
        full = model(**inputs, use_aef=True, use_highres=True)

    assert base.pre_vmf.dtype == torch.bfloat16
    assert full.pre_vmf.dtype == base.pre_vmf.dtype
    assert full.embedding.dtype == base.embedding.dtype
    torch.testing.assert_close(full.pre_vmf, base.pre_vmf, atol=0.0, rtol=0.0)
    torch.testing.assert_close(full.embedding, base.embedding, atol=0.0, rtol=0.0)


def test_disabled_branches_need_no_placeholder_tensors() -> None:
    """显式关闭的旁路不得访问 AEF 或高分辨率占位输入。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()

    output = model(
        s2=inputs["s2"],
        s1=inputs["s1"],
        valid_s2=inputs["valid_s2"],
        valid_s1=inputs["valid_s1"],
        aef=None,
        aef_valid=None,
        highres=None,
        highres_valid=None,
        use_aef=False,
        use_highres=False,
    )

    assert bool(output.embedding.isfinite().all())


def _assert_finite_nonzero_gradients(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    )
    assert sum(gradient.abs().sum().item() for gradient in gradients if gradient is not None) > 0.0


def test_gate_override_reaches_both_side_adapters_and_output_projection() -> None:
    """临时非零 gate 必须让两个旁路和输出投影都得到有效梯度。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).train()
    model.s2_stem.requires_grad_(False)
    model.s1_stem.requires_grad_(False)

    output = model(
        **inputs,
        use_aef=True,
        use_highres=True,
        gate_override=0.1,
    )
    output.pre_vmf.square().mean().backward()

    _assert_finite_nonzero_gradients(model.aef_adapter)
    _assert_finite_nonzero_gradients(model.highres_stem)
    _assert_finite_nonzero_gradients(model.highres_adapter)
    _assert_finite_nonzero_gradients(model.output_projection)
    assert output.gates["aef"].item() == pytest.approx(0.1)
    assert output.gates["highres"].item() == pytest.approx(0.1)
    assert model.aef_gate.item() == 0.0
    assert model.highres_gate.item() == 0.0


def test_gate_override_keeps_exact_forward_value_and_gate_parameter_gradients() -> None:
    """override 只能替换前向值；切断 tanh(gate) 梯度是无效 gradient smoke。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).train()

    output = model(
        **inputs,
        use_aef=True,
        use_highres=True,
        gate_override=0.1,
    )
    expected = torch.as_tensor(0.1, dtype=output.pre_vmf.dtype)
    torch.testing.assert_close(output.gates["aef"], expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(output.gates["highres"], expected, atol=0.0, rtol=0.0)

    output.pre_vmf.square().mean().backward()

    assert model.aef_gate.grad is not None
    assert model.highres_gate.grad is not None
    assert bool(torch.isfinite(model.aef_gate.grad)) and model.aef_gate.grad.abs().item() > 0.0
    assert (
        bool(torch.isfinite(model.highres_gate.grad)) and model.highres_gate.grad.abs().item() > 0.0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "quarters",
        "months",
        "s2-channels",
        "mask-channels",
        "batch",
        "spatial",
        "aef-channels",
        "highres-grid",
        "highres-mask-grid",
    ],
)
def test_forward_rejects_every_nonexact_input_contract(mutation: str) -> None:
    """季度/月数/通道/mask/batch/空间/5H 网格均须在卷积前显式 fail closed。"""
    inputs = _small_inputs()
    if mutation == "quarters":
        inputs["s2"] = inputs["s2"][:, :3]
        inputs["valid_s2"] = inputs["valid_s2"][:, :3]
    elif mutation == "months":
        inputs["s2"] = inputs["s2"][:, :, :2]
        inputs["valid_s2"] = inputs["valid_s2"][:, :, :2]
    elif mutation == "s2-channels":
        inputs["s2"] = inputs["s2"][:, :, :, :9]
    elif mutation == "mask-channels":
        inputs["valid_s2"] = inputs["valid_s2"].expand(-1, -1, -1, 2, -1, -1)
    elif mutation == "batch":
        inputs["s1"] = inputs["s1"][:1]
        inputs["valid_s1"] = inputs["valid_s1"][:1]
    elif mutation == "spatial":
        inputs["s1"] = inputs["s1"][..., :15, :]
        inputs["valid_s1"] = inputs["valid_s1"][..., :15, :]
    elif mutation == "aef-channels":
        inputs["aef"] = inputs["aef"][:, :63]
    elif mutation == "highres-grid":
        inputs["highres"] = inputs["highres"][..., :79, :]
    elif mutation == "highres-mask-grid":
        inputs["highres_valid"] = inputs["highres_valid"][..., :79, :]
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    model = IsolatedFusionSmokeModel(embed_dim=64).eval()
    with pytest.raises(ValueError, match="input contract"):
        model(**inputs, use_aef=True, use_highres=True)


def test_downsampled_highres_mask_keeps_only_completely_valid_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全有效窗口为真，含任一无效像素的窗口为假，且不再依赖平均池化。"""
    valid = torch.ones((1, 1, 10, 10), dtype=torch.bool)
    valid[:, :, 0, 0] = False

    def forbidden_avg_pool2d(*_args, **_kwargs):
        raise AssertionError("all-valid mask must not depend on average-pool rounding")

    monkeypatch.setattr(model_module.functional, "avg_pool2d", forbidden_avg_pool2d)
    downsampled = model_module._downsample_all_valid_mask(valid, dtype=torch.float32)

    expected = torch.tensor([[[[False, True], [True, True]]]])
    torch.testing.assert_close(downsampled, expected)


def test_highres_branch_downsamples_the_real_five_times_grid() -> None:
    """高分旁路必须让 5H×5W 张量直接进入 5×5/stride-5 卷积。"""
    inputs = _small_inputs()
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()
    downsampler = model.highres_stem[0]
    assert isinstance(downsampler, torch.nn.Conv2d)
    assert downsampler.kernel_size == (5, 5)
    assert downsampler.stride == (5, 5)
    observed_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record_shapes(
        _module: torch.nn.Module,
        module_inputs: tuple[torch.Tensor, ...],
        module_output: torch.Tensor,
    ) -> None:
        observed_shapes.append((tuple(module_inputs[0].shape), tuple(module_output.shape)))

    handle = downsampler.register_forward_hook(record_shapes)
    try:
        model(**inputs, use_aef=False, use_highres=True)
    finally:
        handle.remove()

    assert observed_shapes == [((2, 3, 80, 80), (2, 32, 16, 16))]
