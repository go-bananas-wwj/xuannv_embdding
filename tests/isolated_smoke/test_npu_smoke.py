from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.nn import functional as functional

from experiments.china_v1_fusion_smoke.model import IsolatedFusionSmokeModel
from experiments.china_v1_fusion_smoke.registry import validate_smoke_registry

SANDBOX = Path("/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815")


@dataclass(frozen=True)
class SmallNpuResult:
    embedding: torch.Tensor
    aef_grad_norm: float
    highres_grad_norm: float


def _load_prepared_side_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    manifests = SANDBOX / "manifests"
    aef_registry = json.loads((manifests / "aef_registry.json").read_text(encoding="utf-8"))
    highres_registry = json.loads(
        (manifests / "highres_2m_registry.json").read_text(encoding="utf-8")
    )
    validate_smoke_registry(aef_registry)
    validate_smoke_registry(highres_registry)
    aef_entry = aef_registry["entries"][0]
    highres_entry = highres_registry["entries"][0]
    assert (aef_entry["patch_id"], aef_entry["year"]) == (
        highres_entry["patch_id"],
        highres_entry["year"],
    )
    aef_payload = torch.load(SANDBOX / aef_entry["path"], map_location="cpu", weights_only=True)
    highres_payload = torch.load(
        SANDBOX / highres_entry["path"], map_location="cpu", weights_only=True
    )
    aef = functional.interpolate(
        aef_payload["aef"].unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False
    )
    aef_valid = functional.interpolate(
        aef_payload["aef_valid"].unsqueeze(0).float(), size=(16, 16), mode="nearest"
    ).bool()
    highres = functional.interpolate(
        highres_payload["highres"].unsqueeze(0),
        size=(80, 80),
        mode="bilinear",
        align_corners=False,
    )
    highres_valid = functional.interpolate(
        highres_payload["highres_valid"].unsqueeze(0).float(),
        size=(80, 80),
        mode="nearest",
    ).bool()
    return aef, aef_valid, highres, highres_valid


def _gradient_l1(module: torch.nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    )
    return float(sum(gradient.abs().sum().item() for gradient in gradients if gradient is not None))


def run_small_npu_full_branch(*, device: str) -> SmallNpuResult:
    """用已准备旁路缓存执行一个小尺寸 Full 前向和反向。"""
    aef, aef_valid, highres, highres_valid = _load_prepared_side_inputs()
    generator = torch.Generator(device="cpu").manual_seed(20260815)
    inputs = {
        "s2": torch.randn((1, 4, 3, 10, 16, 16), generator=generator).to(device),
        "s1": torch.randn((1, 4, 3, 2, 16, 16), generator=generator).to(device),
        "valid_s2": torch.ones((1, 4, 3, 1, 16, 16), dtype=torch.bool, device=device),
        "valid_s1": torch.ones((1, 4, 3, 1, 16, 16), dtype=torch.bool, device=device),
        "aef": aef.to(device),
        "aef_valid": aef_valid.to(device),
        "highres": highres.to(device),
        "highres_valid": highres_valid.to(device),
    }
    torch.manual_seed(20260815)
    model = IsolatedFusionSmokeModel(embed_dim=64).to(device).train()
    model.s2_stem.requires_grad_(False)
    model.s1_stem.requires_grad_(False)
    output = model(
        **inputs,
        use_aef=True,
        use_highres=True,
        gate_override=0.1,
    )
    output.pre_vmf.square().mean().backward()
    torch.npu.synchronize()
    result = SmallNpuResult(
        embedding=output.embedding.detach().cpu(),
        aef_grad_norm=_gradient_l1(model.aef_adapter),
        highres_grad_norm=(_gradient_l1(model.highres_stem) + _gradient_l1(model.highres_adapter)),
    )
    del output, model, inputs
    torch.npu.empty_cache()
    return result


def test_one_visible_npu_runs_full_branch_backward() -> None:
    """显式 opt-in 时，唯一逻辑 NPU 必须完成有限且双旁路非零梯度的 Full 反向。"""
    if os.environ.get("RUN_XUANNV_NPU_SMOKE") != "1":
        pytest.skip("explicit NPU smoke opt-in required")

    import torch_npu  # noqa: F401

    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "2"
    assert torch.npu.device_count() == 1
    torch.npu.set_device("npu:0")
    result = run_small_npu_full_branch(device="npu:0")
    assert result.embedding.shape == (1, 4, 64, 16, 16)
    assert result.embedding.isfinite().all().item()
    assert result.aef_grad_norm > 0.0
    assert result.highres_grad_norm > 0.0
