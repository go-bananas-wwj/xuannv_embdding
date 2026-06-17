from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xuannv_embedding.models.blocks import SpaceOperator, TimeOperator
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.sensor_encoders import SensorEncoderBank
from xuannv_embedding.models.time_encoding import TimeEncoding

# AEF 主模型：多源时序遥感数据 -> 空间/时间注意力 -> vMF 瓶颈 -> 解码器。


@dataclass
class AEFOutput:
    """AEFModel 前向输出容器。"""

    embedding_map: torch.Tensor  # [B, D, H, W]
    embedding: torch.Tensor  # [B, D]
    reconstructions: dict[str, torch.Tensor]


class AEFModel(nn.Module):
    """AEF（Angular Embedding Field）月度地理嵌入模型。

    输入支持多源时序遥感数据（如 S2/S1/Landsat），以及可选的稀疏高分辨率
    数据。模型通过 per-sensor encoder、时间/空间自注意力、vMF 瓶颈生成单位
    球面上的像素级与场景级嵌入，并解码回各目标模态。
    """

    def __init__(
        self,
        sensor_channels: dict[str, int],
        embed_dim: int,
        target_heads: dict[str, tuple[str, int]],
        num_time_heads: int = 8,
        num_space_heads: int = 8,
    ) -> None:
        """初始化 AEFModel。

        Args:
            sensor_channels: 数据源到输入通道数的映射，例如
                ``{"s2": 13, "s1": 2, "landsat": 11, "highres": 3}``。
                若使用高分辨率融合，必须包含 ``"highres"``。
            embed_dim: 统一嵌入维度，也是各 encoder/operator 的通道数。
            target_heads: 解码器配置，格式为 ``name -> (kind, channels)``，
                其中 ``kind`` 为 ``"continuous"`` 或 ``"categorical"``。
            num_time_heads: TimeOperator 的注意力头数。
            num_space_heads: SpaceOperator 的注意力头数。
        """
        super().__init__()
        self.sensor_channels = sensor_channels
        self.embed_dim = embed_dim
        self.target_heads = target_heads

        self.time_encoding = TimeEncoding(embed_dim)
        self.sensor_bank = SensorEncoderBank(sensor_channels, embed_dim)
        self.time_op = TimeOperator(embed_dim, num_heads=num_time_heads)
        self.space_op = SpaceOperator(embed_dim, num_heads=num_space_heads)
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

    def _encode_temporal_source(
        self,
        x: torch.Tensor,
        source: str,
        time_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """编码单个时序数据源。

        Args:
            x: 输入时序数据，形状 (B, T, C, H, W)。
            source: 数据源名称。
            time_emb: 时间编码，形状 (B, T, embed_dim)。
            mask: 时间有效掩码，形状 (B, T)；1 表示有效，0 表示缺失。

        Returns:
            该 source 融合后的空间特征，形状 (B, embed_dim, H, W)。
        """
        batch_size, time_steps, in_channels, height, width = x.shape

        # 1) 将时序帧展平为 (B*T, C, H, W)，过 per-sensor encoder。
        x = x.reshape(batch_size * time_steps, in_channels, height, width)
        x = self.sensor_bank(x, source)  # (B*T, embed_dim, H, W)
        x = x.view(batch_size, time_steps, self.embed_dim, height, width)

        # 2) 注入时间编码。
        x = x + time_emb[:, :, :, None, None]  # 广播到 (B, T, embed_dim, H, W)

        # 3) 对时间维度应用 TimeOperator：把 H*W 合并到 batch 维度。
        # 结果形状 (B*H*W, T, embed_dim)，每个空间位置独立做时间自注意力。
        x = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, time_steps, self.embed_dim
        )
        x = self.time_op(x)
        x = x.view(batch_size, height, width, time_steps, self.embed_dim).permute(
            0, 3, 4, 1, 2
        )  # (B, T, embed_dim, H, W)

        # 4) 按时间掩码做加权平均，得到 (B, embed_dim, H, W)。
        mask = mask[..., None, None, None]  # (B, T, 1, 1, 1)
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return x

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frame: torch.Tensor | None = None,
        highres_mask: torch.Tensor | None = None,
    ) -> AEFOutput:
        """AEFModel 前向传播。

        Args:
            source_frames: 各时序数据源的输入，格式 ``{source: (B, T, C, H, W)}``。
            source_masks: 各时序数据源的时间有效掩码，格式 ``{source: (B, T)}``。
            timestamps: 时间戳，形状 (B, T)。
            highres_frame: 可选的高分辨率单帧输入，形状 (B, C, H, W)。
            highres_mask: 高分辨率可用性掩码，形状 (B, 1, H, W)；
                提供 ``highres_frame`` 时也必须提供。

        Returns:
            AEFOutput，包含 embedding_map、embedding 与 reconstructions。
        """
        # 统一时间编码。
        time_emb = self.time_encoding(timestamps)  # (B, T, embed_dim)

        # 编码所有时序数据源并求平均。
        feats: list[torch.Tensor] = []
        for source, x in source_frames.items():
            if source not in self.sensor_channels:
                raise KeyError(
                    f"未知数据源: {source!r}，可用数据源: {list(self.sensor_channels.keys())}"
                )
            mask = source_masks.get(source)
            if mask is None:
                raise KeyError(f"缺少 source mask: {source!r}")
            feat = self._encode_temporal_source(x, source, time_emb, mask)
            feats.append(feat)

        base_feat = torch.stack(feats, dim=0).mean(dim=0)  # (B, embed_dim, H, W)

        # 空间自注意力：将 H*W 视为序列长度。
        batch_size, channels, height, width = base_feat.shape
        spatial = base_feat.view(batch_size, channels, height * width).permute(
            0, 2, 1
        )  # (B, H*W, C)
        spatial = self.space_op(spatial)
        base_feat = spatial.permute(0, 2, 1).view(batch_size, channels, height, width)

        # 可选的高分辨率 availability-aware 融合。
        if highres_frame is not None:
            if highres_mask is None:
                raise ValueError("提供 highres_frame 时必须同时提供 highres_mask")
            if "highres" not in self.sensor_channels:
                raise KeyError("sensor_channels 中未注册 'highres'，无法融合高分辨率数据")
            highres_feat = self.sensor_bank(highres_frame, "highres")
            base_feat = self.highres_fusion(base_feat, highres_feat, highres_mask)

        # vMF 瓶颈：输出位于单位球面。
        emb_map = self.bottleneck(base_feat)  # (B, embed_dim, H, W)

        # 全局平均池化得到场景级嵌入。
        emb = emb_map.mean(dim=[2, 3])  # (B, embed_dim)

        # 解码器重建目标模态。
        reconstructions = {name: decoder(emb_map) for name, decoder in self.decoders.items()}

        return AEFOutput(embedding_map=emb_map, embedding=emb, reconstructions=reconstructions)
