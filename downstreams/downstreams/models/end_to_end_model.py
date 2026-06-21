from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from downstreams.inference import load_model_for_inference


class EndToEndModel(nn.Module):
    """端到端分割模型：将预训练 AEF encoder 与下游 head 包装为统一 nn.Module。

    输入为单张/双时相高分影像（已按 bitemporal/include_diff 通道拼接），
    内部构造 AEFModel 所需的 source_frames/source_masks/timestamps/highres_frames
    等字典，调用 encoder 得到 embedding_map，再喂给 segmentation head。
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        highres_source_name: str,
        sensor_channels: dict[str, int],
        months: list[str],
        output_size: tuple[int, int] = (128, 128),
        target_size: tuple[int, int] | None = None,
        include_diff: bool = True,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        ref_year: int = 2025,
        ref_month: int = 1,
        num_months: int = 17,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.highres_source_name = highres_source_name
        self.sensor_channels = sensor_channels
        self.months = months
        self.output_size = output_size
        self.target_size = target_size
        self.include_diff = include_diff
        self.highres_channels = sensor_channels[highres_source_name]
        self.temporal_sources = [s for s in sensor_channels if not s.startswith("highres")]

        mean_arr = torch.zeros(self.highres_channels, dtype=torch.float32)
        std_arr = torch.ones(self.highres_channels, dtype=torch.float32)
        if mean is not None:
            mean_arr = torch.tensor(mean, dtype=torch.float32)
        if std is not None:
            std_arr = torch.tensor(std, dtype=torch.float32)
            std_arr = torch.where(std_arr == 0.0, torch.ones_like(std_arr), std_arr)
        self.register_buffer("_mean", mean_arr.view(-1, 1, 1))
        self.register_buffer("_std", std_arr.view(-1, 1, 1))

        self.ref_year = ref_year
        self.ref_month = ref_month
        self.num_months = num_months

    def _yyyymm_to_index(self, yyyymm: int) -> int:
        year = yyyymm // 100
        month = yyyymm % 100
        idx = (year - self.ref_year) * 12 + (month - self.ref_month)
        if idx < 0 or idx >= self.num_months:
            raise ValueError(
                f"月份 {yyyymm} 超出模型月度 bin 范围 "
                f"[{self.ref_year}{self.ref_month:02d}, "
                f"{self.ref_year + (self.ref_month + self.num_months - 1) // 12}"
                f"{((self.ref_month + self.num_months - 1) % 12) + 1:02d}]"
            )
        return idx

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean.to(x.device, x.dtype)) / self._std.to(x.device, x.dtype)

    def set_backbone_frozen(self, freeze: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        M = len(self.months)

        # 拆分双时相影像；dataset 已按 [t1, t2, |t1-t2|] 拼接通道
        frames: list[torch.Tensor] = []
        for m in range(M):
            start = m * self.highres_channels
            frames.append(image[:, start : start + self.highres_channels])

        frames = [self._normalize(f) for f in frames]
        # 高分影像在 AEFModel 内部会被复制到所有月度 bin，因此传入多时相均值帧即可
        highres_frame = torch.stack(frames, dim=1).mean(dim=1)  # (B, C, H, W)

        out_h, out_w = self.output_size

        highres_frames = {self.highres_source_name: highres_frame}
        # AEFModel 期望 highres_masks 与低分辨率参考尺寸一致
        highres_masks = {
            self.highres_source_name: torch.ones(
                B, 1, out_h, out_w, dtype=torch.float32, device=highres_frame.device
            )
        }

        source_frames: dict[str, torch.Tensor] = {}
        source_masks: dict[str, torch.Tensor] = {}
        for source in self.temporal_sources:
            ch = self.sensor_channels[source]
            source_frames[source] = torch.zeros(
                B, M, ch, out_h, out_w, dtype=highres_frame.dtype, device=highres_frame.device
            )
            source_masks[source] = torch.ones(
                B, M, dtype=torch.float32, device=highres_frame.device
            )

        timestamps = torch.tensor(
            [[int(m) for m in self.months]] * B,
            dtype=torch.long,
            device=highres_frame.device,
        )

        output = self.encoder(
            source_frames=source_frames,
            source_masks=source_masks,
            timestamps=timestamps,
            highres_frames=highres_frames,
            highres_masks=highres_masks,
        )

        # AEFModel 输出 AEFOutput，也可能被简单包装为 tuple
        if isinstance(output, (tuple, list)):
            embedding_map = output[0]
        elif hasattr(output, "embedding_map"):
            embedding_map = output.embedding_map
        else:
            embedding_map = output

        # (B, T_month, D, out_h, out_w)
        month_indices = [self._yyyymm_to_index(int(m)) for m in self.months]
        embs: list[torch.Tensor] = []
        for bin_idx in month_indices:
            emb = embedding_map[:, bin_idx]  # (B, D, out_h, out_w)
            embs.append(emb)

        if M == 2:
            emb_t1, emb_t2 = embs
            emb = torch.cat([emb_t1, emb_t2], dim=1)
            if self.include_diff:
                emb = torch.cat([emb, torch.abs(emb_t1 - emb_t2)], dim=1)
        else:
            emb = embs[0]

        if self.target_size is not None and emb.shape[-2:] != self.target_size:
            emb = F.interpolate(
                emb, size=self.target_size, mode="bilinear", align_corners=False
            )

        return self.head(emb)


def _load_highres_statistics(statistics_dir: Path, source: str) -> dict[str, Any]:
    stat_path = statistics_dir / f"{source}_stats.json"
    if not stat_path.exists():
        return {}
    with open(stat_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_end_to_end_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    head: nn.Module,
    freeze_backbone_epochs: int = 0,
    output_size: tuple[int, int] = (128, 128),
    target_size: tuple[int, int] | None = None,
    months: list[str] | None = None,
    include_diff: bool = True,
    highres_source_name: str | None = None,
) -> tuple[EndToEndModel, torch.device]:
    """从 AEF 配置/检查点构建端到端模型。"""
    encoder, cfg, device = load_model_for_inference(
        config_path, checkpoint_path, random_init=False
    )

    sensor_channels = cfg.model.sensor_channels
    if highres_source_name is None:
        highres_sources = [s for s in sensor_channels if s.startswith("highres_optical_")]
        if not highres_sources:
            raise ValueError("encoder sensor_channels 中未找到 highres_optical 源")
        highres_source_name = highres_sources[0]

    if highres_source_name not in sensor_channels:
        raise ValueError(f"sensor_channels 中不存在 {highres_source_name}")

    statistics = _load_highres_statistics(cfg.data.statistics_dir, highres_source_name)

    model = EndToEndModel(
        encoder=encoder,
        head=head,
        highres_source_name=highres_source_name,
        sensor_channels=sensor_channels,
        months=months if months is not None else ["202605"],
        output_size=output_size,
        target_size=target_size,
        include_diff=include_diff,
        mean=statistics.get("mean"),
        std=statistics.get("std"),
        ref_year=2025,
        ref_month=1,
        num_months=cfg.model.num_months,
    )
    if freeze_backbone_epochs > 0:
        model.set_backbone_frozen(True)

    return model.to(device), device


__all__ = ["EndToEndModel", "build_end_to_end_model"]
