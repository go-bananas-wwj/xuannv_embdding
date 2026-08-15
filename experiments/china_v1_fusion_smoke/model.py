from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as functional


def _downsample_all_valid_mask(valid: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """仅保留 5×5 窗口内不存在无效像素的位置。"""
    invalid = (~valid).to(dtype=dtype)
    return functional.max_pool2d(invalid, kernel_size=5, stride=5).eq(0.0)


@dataclass(frozen=True)
class FusionOutput:
    """隔离融合模型的季度单位向量及门控前中间量。"""

    embedding: torch.Tensor
    pre_vmf: torch.Tensor
    gates: Mapping[str, torch.Tensor]


class IsolatedFusionSmokeModel(nn.Module):
    """仅用于 China V1 隔离 smoke 的可撤销季度融合模型。"""

    def __init__(self, embed_dim: int = 64, sensor_dim: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.s2_stem = nn.Sequential(
            nn.Conv2d(10, sensor_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(sensor_dim, sensor_dim, kernel_size=3, padding=1),
        )
        self.s1_stem = nn.Sequential(
            nn.Conv2d(2, sensor_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(sensor_dim, sensor_dim, kernel_size=3, padding=1),
        )
        self.output_projection = nn.Conv2d(sensor_dim * 2, embed_dim, kernel_size=1)

        self.aef_adapter = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(16, embed_dim, kernel_size=1),
        )
        self.highres_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=5),
            nn.GELU(),
        )
        self.highres_adapter = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.GELU(),
            nn.Conv2d(32, embed_dim, kernel_size=1),
        )
        self.aef_gate = nn.Parameter(torch.zeros(()))
        self.highres_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _quarterly_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
        stem: nn.Module,
    ) -> torch.Tensor:
        batch_size, quarters, months, channels, height, width = values.shape
        sanitized = torch.where(valid, values, torch.zeros_like(values))
        encoded = stem(sanitized.reshape(-1, channels, height, width))
        encoded = encoded.reshape(batch_size, quarters, months, -1, height, width)
        weights = valid.to(encoded.dtype)
        total = (encoded * weights).sum(dim=2)
        counts = weights.sum(dim=2).clamp_min(1.0)
        return total / counts

    def _gate_values(
        self, reference: torch.Tensor, gate_override: float | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gate_override is not None:
            override = torch.as_tensor(
                gate_override,
                device=reference.device,
                dtype=reference.dtype,
            )
            learned_aef = torch.tanh(self.aef_gate).to(reference)
            learned_highres = torch.tanh(self.highres_gate).to(reference)
            return (
                learned_aef + (override - learned_aef).detach(),
                learned_highres + (override - learned_highres).detach(),
            )
        return torch.tanh(self.aef_gate).to(reference), torch.tanh(self.highres_gate).to(reference)

    @staticmethod
    def _validate_input_contract(
        s2: torch.Tensor,
        s1: torch.Tensor,
        valid_s2: torch.Tensor,
        valid_s1: torch.Tensor,
        aef: torch.Tensor | None,
        aef_valid: torch.Tensor | None,
        highres: torch.Tensor | None,
        highres_valid: torch.Tensor | None,
        use_aef: bool,
        use_highres: bool,
    ) -> None:
        """在任何卷积前验证固定季度/月数/通道与共同空间轴。"""
        tensors = (s2, s1, valid_s2, valid_s1)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ValueError("input contract requires tensor sensor inputs and masks")
        if s2.ndim != 6:
            raise ValueError("input contract requires S2 rank 6")
        batch, quarters, months, channels, height, width = s2.shape
        if (quarters, months, channels) != (4, 3, 10) or height <= 0 or width <= 0:
            raise ValueError("input contract requires S2 [B,4,3,10,H,W]")
        if tuple(s1.shape) != (batch, 4, 3, 2, height, width):
            raise ValueError("input contract requires S1 [B,4,3,2,H,W] on the common grid")
        mask_shape = (batch, 4, 3, 1, height, width)
        if tuple(valid_s2.shape) != mask_shape or tuple(valid_s1.shape) != mask_shape:
            raise ValueError("input contract requires sensor masks [B,4,3,1,H,W]")
        if valid_s2.dtype is not torch.bool or valid_s1.dtype is not torch.bool:
            raise ValueError("input contract requires bool sensor masks")
        if use_aef:
            if not isinstance(aef, torch.Tensor) or not isinstance(aef_valid, torch.Tensor):
                raise ValueError("input contract requires enabled AEF tensors")
            if tuple(aef.shape) != (batch, 64, height, width):
                raise ValueError("input contract requires AEF [B,64,H,W]")
            if tuple(aef_valid.shape) != (batch, 1, height, width):
                raise ValueError("input contract requires AEF mask [B,1,H,W]")
            if aef_valid.dtype is not torch.bool:
                raise ValueError("input contract requires a bool AEF mask")
        if use_highres:
            if not isinstance(highres, torch.Tensor) or not isinstance(
                highres_valid, torch.Tensor
            ):
                raise ValueError("input contract requires enabled highres tensors")
            highres_shape = (batch, 3, height * 5, width * 5)
            highres_mask_shape = (batch, 1, height * 5, width * 5)
            if tuple(highres.shape) != highres_shape:
                raise ValueError("input contract requires highres [B,3,5H,5W]")
            if tuple(highres_valid.shape) != highres_mask_shape:
                raise ValueError("input contract requires highres mask [B,1,5H,5W]")
            if highres_valid.dtype is not torch.bool:
                raise ValueError("input contract requires a bool highres mask")

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        valid_s2: torch.Tensor,
        valid_s1: torch.Tensor,
        aef: torch.Tensor | None,
        aef_valid: torch.Tensor | None,
        highres: torch.Tensor | None,
        highres_valid: torch.Tensor | None,
        use_aef: bool,
        use_highres: bool,
        gate_override: float | None = None,
    ) -> FusionOutput:
        """融合显式启用的年度旁路；关闭旁路时不访问其输入。"""
        self._validate_input_contract(
            s2,
            s1,
            valid_s2,
            valid_s1,
            aef,
            aef_valid,
            highres,
            highres_valid,
            use_aef,
            use_highres,
        )
        s2_quarterly = self._quarterly_mean(s2, valid_s2, self.s2_stem)
        s1_quarterly = self._quarterly_mean(s1, valid_s1, self.s1_stem)
        batch_size, quarters, _, height, width = s2_quarterly.shape
        base_inputs = torch.cat((s2_quarterly, s1_quarterly), dim=2)
        base = self.output_projection(
            base_inputs.reshape(batch_size * quarters, -1, height, width)
        ).reshape(batch_size, quarters, self.embed_dim, height, width)

        aef_gate, highres_gate = self._gate_values(base, gate_override)
        z = base
        if use_aef:
            if aef is None or aef_valid is None:
                raise ValueError("enabled AEF branch requires aef and aef_valid")
            masked_aef = torch.where(aef_valid, aef, torch.zeros_like(aef))
            aef_delta = self.aef_adapter(masked_aef)
            aef_delta = aef_delta * aef_valid.to(device=aef_delta.device, dtype=aef_delta.dtype)
            z = z + aef_gate * aef_delta[:, None].expand(-1, quarters, -1, -1, -1)
        if use_highres:
            if highres is None or highres_valid is None:
                raise ValueError("enabled highres branch requires highres and highres_valid")
            masked_highres = torch.where(highres_valid, highres, torch.zeros_like(highres))
            highres_features = self.highres_stem(masked_highres)
            downsampled_valid = _downsample_all_valid_mask(
                highres_valid,
                dtype=highres.dtype,
            )
            highres_delta = self.highres_adapter(highres_features)
            highres_delta = highres_delta * downsampled_valid.to(
                device=highres_delta.device, dtype=highres_delta.dtype
            )
            z = z + highres_gate * highres_delta[:, None].expand(-1, quarters, -1, -1, -1)

        embedding = functional.normalize(z, p=2, dim=2, eps=1.0e-6)
        return FusionOutput(
            embedding=embedding,
            pre_vmf=z,
            gates={"aef": aef_gate, "highres": highres_gate},
        )
