from __future__ import annotations

# 将 dataset/collate 输出的按 source 组织 batch 转换为模型可用的统一 batch。
from typing import Any

import torch
import torch.nn.functional as F


def _downsample_spatial(tensor: torch.Tensor, stride: int, mode: str = "bilinear") -> torch.Tensor:
    """按 stride 对空间张量进行下采样。

    参数:
        tensor: 输入张量，最后两维为空间维度 (H, W)。
        stride: 下采样倍数。
        mode: ``bilinear`` 用于连续特征，``nearest`` 用于类别标签。

    返回:
        下采样后的张量，形状 ``[..., H//stride, W//stride]``。
    """
    if stride <= 1:
        return tensor
    # 处理类别标签 [B, H, W]，需先/后增加通道维。
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1).float()
        down = F.interpolate(
            tensor,
            scale_factor=1.0 / stride,
            mode="nearest",
        )
        return down.squeeze(1)
    return F.avg_pool2d(tensor, kernel_size=stride)


def _head_source_name(
    head_name: str,
    head_cfg: dict[str, Any],
    available_sources: set[str],
) -> str | None:
    """根据 head 配置与可用 source 确定该 head 对应的数据源。

    匹配优先级:
        1. head 配置中的 ``source`` 字段。
        2. 去掉 ``_recon`` 后缀后的 head 名。
        3. head 名本身（若其本身是 source 名）。

    参数:
        head_name: head 名称。
        head_cfg: head 配置字典。
        available_sources: 当前 batch 中可用的 source 名称集合。

    返回:
        对应的 source 名，未找到时返回 ``None``。
    """
    explicit = head_cfg.get("source")
    if explicit is not None:
        return explicit
    if head_name.endswith("_recon"):
        stripped = head_name[: -len("_recon")]
        if stripped in available_sources:
            return stripped
    if head_name in available_sources:
        return head_name
    return None


def _weighted_temporal_mean(frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对时序帧按时间掩码加权取平均。

    参数:
        frames: 输入帧，形状 ``[B, T, C, H, W]``。
        mask: 时间有效掩码，形状 ``[B, T]``。

    返回:
        加权平均后的帧，形状 ``[B, C, H, W]``。
    """
    m = mask[..., None, None, None]
    weighted = (frames * m).sum(dim=1)
    count = mask.sum(dim=1, keepdim=True)[..., None, None]
    return weighted / (count + 1e-8)


def prepare_batch(
    batch: dict[str, Any],
    target_heads: dict[str, dict[str, Any]],
    spatial_stride: int = 1,
) -> dict[str, Any]:
    """将 collate 后的 batch 转换为 AEFModel / Trainer 需要的格式。

    本函数只负责 shape/head 转换与张量构造，不搬运 device；device 搬运由
    ``Trainer._move_batch_to_device`` 统一完成，避免职责重复。

    主要转换:
        - 从第一个有效的 source 提取全局 ``timestamps``（``[B, T]``）。
        - 若存在 ``highres`` source，将其聚合为单帧与可用性掩码并从时序源中移除。
        - 为每个 target head 构造 ``targets`` 与 ``target_masks``。
        - 当 ``spatial_stride > 1`` 时，将 target 下采样到与模型 embedding_map
          相同的空间分辨率。

    参数:
        batch: ``collate_fn`` 输出，包含 ``patch_ids``、``source_frames``、
            ``source_masks``、``timestamps``。
        target_heads: 配置中的 target_heads，每个 head 至少包含
            ``loss_type``（``continuous`` / ``categorical``）与 ``channels``，
            可选 ``source`` 与 ``weight``。
        spatial_stride: sensor encoder 的下采样倍数，用于对齐 target 分辨率。

    返回:
        转换后的 batch 字典，可直接喂给 ``Trainer``。
    """
    source_frames = batch["source_frames"]
    source_masks = batch["source_masks"]
    source_timestamps = batch["timestamps"]
    patch_ids = batch["patch_ids"]

    if not source_frames:
        raise ValueError("source_frames 不能为空字典")

    # 取第一个 T > 0 的 source 的时间戳作为全局 timestamps。
    global_timestamps: torch.Tensor | None = None
    for source, frames in source_frames.items():
        if frames.shape[1] > 0:
            global_timestamps = source_timestamps[source]
            break
    if global_timestamps is None:
        batch_size = next(iter(source_frames.values())).shape[0]
        global_timestamps = torch.zeros(batch_size, 0, dtype=torch.long)

    # 为每个 target head 构造 target 与 target_mask。
    available_sources = set(source_frames.keys())
    targets: dict[str, torch.Tensor] = {}
    target_masks: dict[str, torch.Tensor] = {}

    batch_size = next(iter(source_frames.values())).shape[0]
    spatial_h = next(iter(source_frames.values())).shape[-2]
    spatial_w = next(iter(source_frames.values())).shape[-1]
    out_h = spatial_h // spatial_stride
    out_w = spatial_w // spatial_stride

    for head_name, head_cfg in target_heads.items():
        source_name = _head_source_name(head_name, head_cfg, available_sources)
        loss_type = head_cfg["loss_type"]
        channels = head_cfg["channels"]

        if (
            source_name is not None
            and source_name in source_frames
            and source_frames[source_name].shape[1] > 0
        ):
            frames = source_frames[source_name]
            masks = source_masks[source_name]
            agg = _weighted_temporal_mean(frames, masks)
            if loss_type == "continuous":
                target = _downsample_spatial(agg, spatial_stride, mode="bilinear")
            else:  # categorical
                label = agg.argmax(dim=1).long()
                target = _downsample_spatial(label, spatial_stride, mode="nearest")
            height, width = target.shape[-2], target.shape[-1]
            target_mask = torch.ones(batch_size, height, width, dtype=torch.float32)
            targets[head_name] = target
            target_masks[head_name] = target_mask
        else:
            if loss_type == "continuous":
                targets[head_name] = torch.zeros(
                    batch_size,
                    channels,
                    out_h,
                    out_w,
                    dtype=torch.float32,
                )
            else:
                targets[head_name] = torch.zeros(
                    batch_size,
                    out_h,
                    out_w,
                    dtype=torch.long,
                )
            target_masks[head_name] = torch.zeros(
                batch_size,
                out_h,
                out_w,
                dtype=torch.float32,
            )

    # 分离高分辨率数据：按时间掩码加权聚合为单帧，并生成空间可用性掩码。
    highres_frame: torch.Tensor | None = None
    highres_mask: torch.Tensor | None = None
    if "highres" in source_frames:
        hr_frames = source_frames.pop("highres")
        hr_masks = source_masks.pop("highres")
        source_timestamps.pop("highres", None)
        if hr_frames.shape[1] > 0:
            highres_frame = _weighted_temporal_mean(hr_frames, hr_masks)
            _, _, height, width = highres_frame.shape
            avail = (hr_masks.sum(dim=1) > 0).float()
            highres_mask = avail[:, None, None, None].expand(-1, 1, height, width)

    return {
        "patch_ids": patch_ids,
        "source_frames": source_frames,
        "source_masks": source_masks,
        "timestamps": global_timestamps,
        "highres_frame": highres_frame,
        "highres_mask": highres_mask,
        "targets": targets,
        "target_masks": target_masks,
    }
