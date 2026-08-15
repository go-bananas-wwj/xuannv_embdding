from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import torch
from torch.nn import functional as functional

_S2_SHAPE = (4, 3, 10, 128, 128)
_VALID_SHAPE = (4, 3, 1, 128, 128)


@dataclass(frozen=True)
class SyntheticAnnualContext:
    """仅供隔离 smoke test 使用的年度 S2 衍生旁路输入。"""

    aef: torch.Tensor
    aef_valid: torch.Tensor
    highres: torch.Tensor
    highres_valid: torch.Tensor
    metadata: Mapping[str, Mapping[str, object]]


def _generator_seed(seed: int, patch_id: str, year: int) -> int:
    """用跨进程稳定的 SHA-256 代替 Python ``hash`` 派生随机种子。"""
    digest = sha256(f"{seed}|{patch_id}|{year}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _validate_inputs(s2_year: torch.Tensor, valid_s2_year: torch.Tensor) -> None:
    if s2_year.shape != _S2_SHAPE:
        raise ValueError(f"s2_year must have shape {_S2_SHAPE}; got {tuple(s2_year.shape)}")
    if valid_s2_year.shape != _VALID_SHAPE:
        raise ValueError(
            f"valid_s2_year must have shape {_VALID_SHAPE}; got {tuple(valid_s2_year.shape)}"
        )
    if s2_year.dtype != torch.float32:
        raise ValueError(f"s2_year must be float32; got {s2_year.dtype}")
    if valid_s2_year.dtype != torch.bool:
        raise ValueError(f"valid_s2_year must be bool; got {valid_s2_year.dtype}")
    if s2_year.device.type != "cpu" or valid_s2_year.device.type != "cpu":
        raise ValueError("synthetic context generation requires CPU inputs")


def _annual_s2(
    s2_year: torch.Tensor, valid_s2_year: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = s2_year.reshape(12, 10, 128, 128)
    valid = valid_s2_year.reshape(12, 1, 128, 128)
    counts = valid.sum(dim=0)
    annual_valid = counts > 0
    annual = (observations * valid).sum(dim=0) / counts.clamp_min(1).to(s2_year.dtype)
    return annual * annual_valid.to(s2_year.dtype), annual_valid


def generate_synthetic_context(
    s2_year: torch.Tensor,
    valid_s2_year: torch.Tensor,
    patch_id: str,
    year: int,
    seed: int,
) -> SyntheticAnnualContext:
    """从一年 S2 生成确定性的 synthetic AEF 与 5×上采样高分支输入。

    这不是官方 AEF，也不包含真实 2 m 信息；此函数只能服务隔离 smoke test。
    """
    _validate_inputs(s2_year, valid_s2_year)
    annual_s2, aef_valid = _annual_s2(s2_year, valid_s2_year)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_generator_seed(seed, patch_id, year))

    projection = torch.randn((64, 10), generator=generator, dtype=torch.float32)
    aef = torch.einsum("oc,chw->ohw", projection, annual_s2)
    aef = functional.normalize(aef, p=2, dim=0, eps=1.0e-6)

    rgb = annual_s2[[2, 1, 0]].unsqueeze(0)
    up = functional.interpolate(rgb, size=(640, 640), mode="bicubic", align_corners=False)
    blur = functional.avg_pool2d(up, kernel_size=5, stride=1, padding=2)
    texture = torch.randn(up.shape, generator=generator, dtype=torch.float32) * 0.002
    highres = (up + 0.15 * (up - blur) + texture).clamp(0.0, 1.5).squeeze(0)
    highres_valid = (
        functional.interpolate(
            aef_valid.to(torch.float32).unsqueeze(0), size=(640, 640), mode="nearest"
        )
        .to(torch.bool)
        .squeeze(0)
    )

    return SyntheticAnnualContext(
        aef=aef,
        aef_valid=aef_valid,
        highres=highres,
        highres_valid=highres_valid,
        metadata={
            "aef": {
                "synthetic": True,
                "synthetic_kind": "annual_s2_fixed_projection",
                "allowed_use": "smoke_test_only",
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            },
            "highres": {
                "synthetic": True,
                "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
                "allowed_use": "smoke_test_only",
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
                "claimed_native_gsd_m": None,
                "model_input_gsd_m": 2,
                "contains_real_2m_information": False,
            },
        },
    )
