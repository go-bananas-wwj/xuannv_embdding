from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from xuannv_embedding.models.blocks import (
    EmbeddingUpsampleHead,
    STPEncoder,
    TemporalSummarizer,
)
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.sensor_encoders import SensorEncoderBank

# AEF 主模型：多源时序遥感数据 -> 多分辨率 STP 编码器 -> 时序摘要 ->
# 嵌入上采样 -> 可选高分辨率融合 -> vMF 瓶颈 -> 解码器。


@dataclass
class AEFOutput:
    """AEFModel 前向输出容器。"""

    embedding_map: torch.Tensor  # [B, D, H, W]，与输入相同空间分辨率
    embedding: torch.Tensor  # [B, D]
    reconstructions: dict[str, torch.Tensor]


class AEFModel(nn.Module):
    """AEF（Angular Embedding Field）月度地理嵌入模型。

    输入支持多源时序遥感数据（如 S2/S1/Landsat），以及可选的稀疏高分辨率
    数据。模型通过 per-sensor stem、多分辨率 Space-Time-Precision 编码器、
    时序摘要器、嵌入上采样头、vMF 瓶颈生成单位球面上的像素级与场景级嵌入，
    并解码回各目标模态。输出 ``embedding_map`` 与输入保持相同空间分辨率。
    """

    def __init__(
        self,
        sensor_channels: dict[str, int],
        embed_dim: int,
        target_heads: dict[str, tuple[str, int]],
        stem_dim: int = 32,
        stp: dict[str, Any] | None = None,
        num_space_heads: int = 8,
    ) -> None:
        """初始化 AEFModel。

        Args:
            sensor_channels: 数据源到输入通道数的映射，例如
                ``{"s2": 13, "s1": 2, "landsat": 11, "highres": 3, "highres_sar": 1}``。
                若使用高分辨率融合，需要为每个以 ``"highres"`` 开头的 source
                注册对应的通道数。
            embed_dim: 统一嵌入维度，也是最终 embedding_map 的通道数。
            target_heads: 解码器配置，格式为 ``name -> (kind, channels)``，
                其中 ``kind`` 为 ``"continuous"`` 或 ``"categorical"``。
            stem_dim: 各时序数据源经 stem encoder 后输出的统一通道数。
            stp: STP 编码器配置字典，可包含 ``space_dim``、``time_dim``、
                ``precision_dim``、``num_blocks``、``num_heads``。缺失项使用默认值。
            num_space_heads: ``stp["num_heads"]`` 的默认值。
        """
        super().__init__()
        self.sensor_channels = sensor_channels
        self.embed_dim = embed_dim
        self.target_heads = target_heads
        self.stem_dim = stem_dim

        stp_cfg = dict(stp) if stp is not None else {}
        stp_defaults = {
            "space_dim": 512,
            "time_dim": 256,
            "precision_dim": 128,
            "num_blocks": 6,
            "num_heads": num_space_heads,
        }
        for key, value in stp_defaults.items():
            stp_cfg.setdefault(key, value)
        self.stp_cfg = stp_cfg

        temporal_channels = {
            name: ch
            for name, ch in sensor_channels.items()
            if not name.startswith("highres")
        }
        highres_channels = {
            name: ch
            for name, ch in sensor_channels.items()
            if name.startswith("highres")
        }

        self.temporal_source_order = list(temporal_channels.keys())
        self.temporal_stem_bank = SensorEncoderBank(temporal_channels, stem_dim)
        self.highres_encoder_bank = SensorEncoderBank(highres_channels, embed_dim)

        total_stem_channels = stem_dim * len(temporal_channels)
        self.total_stem_channels = total_stem_channels
        self.stp_encoder = STPEncoder(
            input_channels=total_stem_channels,
            space_dim=stp_cfg["space_dim"],
            time_dim=stp_cfg["time_dim"],
            precision_dim=stp_cfg["precision_dim"],
            num_blocks=stp_cfg["num_blocks"],
            num_heads=stp_cfg["num_heads"],
        )

        self.temporal_summarizer = TemporalSummarizer(
            feature_dim=stp_cfg["precision_dim"],
            embed_dim=embed_dim,
            num_heads=stp_cfg["num_heads"],
        )
        self.upsample_head = EmbeddingUpsampleHead(embed_dim, embed_dim)
        self.highres_fusion = AvailabilityAwareFusion(embed_dim)
        self.bottleneck = VMFBottleneck(embed_dim, embed_dim)

        # 构建各目标模态的 decoder head。
        self.decoders = nn.ModuleDict()
        for name, (kind, ch) in target_heads.items():
            if kind == "continuous":
                self.decoders[name] = ContinuousDecoder(embed_dim, ch)
            elif kind == "categorical":
                self.decoders[name] = CategoricalDecoder(embed_dim, ch)
            else:
                raise ValueError(f"不支持的 decoder 类型: {kind!r}")

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frames: dict[str, torch.Tensor] | None = None,
        highres_masks: dict[str, torch.Tensor] | None = None,
    ) -> AEFOutput:
        """AEFModel 前向传播。

        Args:
            source_frames: 各时序数据源的输入，格式 ``{source: (B, T, C, H, W)}``。
            source_masks: 各时序数据源的时间有效掩码，格式 ``{source: (B, T)}``。
            timestamps: 时间戳，形状 ``(B, T)``。
            highres_frames: 可选的高分辨率单帧输入字典，格式 ``{source: (B, C, H, W)}``。
            highres_masks: 高分辨率可用性掩码字典，格式 ``{source: (B, 1, H, W)}``；
                提供 ``highres_frames`` 时也必须提供，且 key 需与 ``highres_frames`` 一致。

        Returns:
            AEFOutput，包含与输入同分辨率的 embedding_map、embedding 与 reconstructions。
        """
        if not source_frames:
            raise ValueError("source_frames 不能为空字典")

        timestamps = timestamps.float()

        # 1) 编码每个时序数据源到 stem 维度，并屏蔽缺失时间步。
        # 跳过时间维度为 0 的 source，避免后续 reshape 出错。
        temporal_sources = [
            s
            for s in source_frames
            if not s.startswith("highres") and source_frames[s].shape[1] > 0
        ]
        if not temporal_sources:
            raise ValueError("source_frames 中至少需要一个有效的时序数据源")

        # 第一个 source 用于确定 batch/time/space 维度。
        first_x = source_frames[temporal_sources[0]]
        B, T, _, H, W = first_x.shape
        input_size = (H, W)
        temporal_input = torch.zeros(
            B, T, H, W, self.total_stem_channels,
            device=first_x.device, dtype=first_x.dtype,
        )
        masks: list[torch.Tensor] = []

        for source in temporal_sources:
            x = source_frames[source]
            if source not in self.sensor_channels:
                raise KeyError(
                    f"未知数据源: {source!r}，可用数据源: {list(self.sensor_channels.keys())}"
                )
            mask = source_masks.get(source)
            if mask is None:
                raise KeyError(f"缺少 source mask: {source!r}")

            x_flat = x.reshape(B * T, x.shape[2], H, W)
            x_enc = self.temporal_stem_bank(x_flat, source)  # (B*T, stem_dim, H, W)
            x_enc = x_enc.view(B, T, self.stem_dim, H, W)
            mask_5d = mask[:, :, None, None, None]
            x_enc = x_enc * mask_5d

            source_idx = self.temporal_source_order.index(source)
            start = source_idx * self.stem_dim
            end = start + self.stem_dim
            temporal_input[..., start:end] = x_enc.permute(0, 1, 3, 4, 2)
            masks.append(mask)

        # 2) 合并时间有效性掩码。
        combined_mask = torch.stack(masks, dim=0).amax(dim=0)  # (B, T)

        # 3) STP 编码器输出 1/2L 精度特征，同时拿回原始输入尺寸。
        feats, _ = self.stp_encoder(
            temporal_input, timestamps, mask=combined_mask
        )  # (B, T, H//2, W//2, precision_dim)

        # 4) 时序摘要：每个像素生成 embed_dim 维单位向量。
        # 只使用有效时间步计算 min/max，避免 padding 0 污染 valid_period。
        masked_ts_min = timestamps.masked_fill(combined_mask == 0, float("inf"))
        masked_ts_max = timestamps.masked_fill(combined_mask == 0, float("-inf"))
        t_min = masked_ts_min.min(dim=1).values
        t_max = masked_ts_max.max(dim=1).values
        empty_period = combined_mask.sum(dim=1) == 0
        t_min = torch.where(empty_period, torch.zeros_like(t_min), t_min)
        t_max = torch.where(empty_period, torch.zeros_like(t_max), t_max)
        valid_period = torch.stack([t_min, t_max], dim=1)

        mu = self.temporal_summarizer(
            feats, timestamps, valid_period, mask=combined_mask
        )  # (B, H//2, W//2, embed_dim)

        # 5) 上采样回原始分辨率；奇数尺寸时使用 target_size 兜底。
        mu_up = self.upsample_head(mu, target_size=input_size)  # (B, H, W, embed_dim)
        base_feat = mu_up.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

        # 6) 可选高分辨率 availability-aware 融合。
        if highres_frames:
            if highres_masks is None:
                raise ValueError("提供 highres_frames 时必须同时提供 highres_masks")
            for source, highres_frame in highres_frames.items():
                if source not in self.sensor_channels:
                    raise KeyError(
                        f"sensor_channels 中未注册 {source!r}，无法融合高分辨率数据"
                    )
                highres_mask = highres_masks.get(source)
                if highres_mask is None:
                    raise KeyError(f"缺少 highres_mask: {source!r}")
                highres_feat = self.highres_encoder_bank(highres_frame, source)
                base_feat = self.highres_fusion(base_feat, highres_feat, highres_mask)

        # 7) vMF 瓶颈：输出位于单位球面。
        emb_map = self.bottleneck(base_feat)  # (B, embed_dim, H, W)

        # 8) 全局平均池化得到场景级嵌入。
        emb = emb_map.mean(dim=[2, 3])  # (B, embed_dim)

        # 9) 解码器重建目标模态。
        reconstructions = {name: decoder(emb_map) for name, decoder in self.decoders.items()}

        return AEFOutput(embedding_map=emb_map, embedding=emb, reconstructions=reconstructions)
