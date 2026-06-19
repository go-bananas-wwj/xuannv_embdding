from __future__ import annotations

# 将 dataset/collate 输出的按 source 组织 batch 转换为模型可用的统一 batch。
from typing import Any

import torch


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
    return weighted / count.clamp(min=1.0)


def prepare_batch(
    batch: dict[str, Any],
    target_heads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """将 collate 后的 batch 转换为 AEFModel / Trainer 需要的格式。

    本函数只负责 shape/head 转换与张量构造，不搬运 device；device 搬运由
    ``Trainer._move_batch_to_device`` 统一完成，避免职责重复。

    主要转换:
        - 从第一个有效的 source 提取全局 ``timestamps``（``[B, T]``）。
        - 若存在名称以 ``highres`` 开头的高分辨率 source，将其各自聚合为单帧与
          可用性掩码并从时序源中移除。
        - 为每个 target head 构造 ``targets`` 与 ``target_masks``，均与输入保持
          相同空间分辨率。

    参数:
        batch: ``collate_fn`` 输出，包含 ``patch_ids``、``source_frames``、
            ``source_masks``、``timestamps``。
        target_heads: 配置中的 target_heads，每个 head 至少包含
            ``loss_type``（``continuous`` / ``categorical``）与 ``channels``，
            可选 ``source`` 与 ``weight``。

    返回:
        转换后的 batch 字典，包含 ``highres_frames`` 与 ``highres_masks`` 两个字典
        （key 为原始高分辨率 source 名），可直接喂给 ``Trainer``。
    """
    source_frames = dict(batch["source_frames"])
    source_masks = dict(batch["source_masks"])
    source_timestamps = dict(batch["timestamps"])
    patch_ids = batch["patch_ids"]

    if not source_frames:
        raise ValueError("source_frames 不能为空字典")

    # 为每个 target head 构造 target 与 target_mask。
    available_sources = set(source_frames.keys())
    targets: dict[str, torch.Tensor] = {}
    target_masks: dict[str, torch.Tensor] = {}

    # 使用第一个非高分辨率 source 的空间尺寸作为参考分辨率。
    base_sources = [s for s in source_frames if not s.startswith("highres")]
    ref_source = base_sources[0] if base_sources else next(iter(source_frames))
    ref_frames = source_frames[ref_source]
    batch_size = ref_frames.shape[0]
    spatial_h = ref_frames.shape[-2]
    spatial_w = ref_frames.shape[-1]

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
                target = agg
                target_mask = torch.ones(batch_size, *target.shape[-2:], dtype=torch.float32)
            else:  # categorical
                # 单通道标签图直接取首通道；多通道 one-hot 取 argmax。
                if agg.shape[1] == 1:
                    label = agg[:, 0].long()
                else:
                    label = agg.argmax(dim=1).long()
                target = label
                # 以 0 作为 nodata/背景，不参与损失计算。
                target_mask = (label != 0).float()
            targets[head_name] = target
            target_masks[head_name] = target_mask
        else:
            if loss_type == "continuous":
                targets[head_name] = torch.zeros(
                    batch_size,
                    channels,
                    spatial_h,
                    spatial_w,
                    dtype=torch.float32,
                )
            else:
                targets[head_name] = torch.zeros(
                    batch_size,
                    spatial_h,
                    spatial_w,
                    dtype=torch.long,
                )
            target_masks[head_name] = torch.zeros(
                batch_size,
                spatial_h,
                spatial_w,
                dtype=torch.float32,
            )

    # 对齐所有非高分辨率时序源的时间长度，确保模型能使用统一的全局 timestamps。
    temporal_sources = [s for s in source_frames if not s.startswith("highres")]
    if temporal_sources:
        max_t = max(source_frames[s].shape[1] for s in temporal_sources)
        base_source = next(
            s for s in temporal_sources if source_frames[s].shape[1] == max_t
        )
        global_timestamps = source_timestamps[base_source]
        for source in temporal_sources:
            t = source_frames[source].shape[1]
            if t == max_t:
                continue
            b, _, c, h, w = source_frames[source].shape
            pad_t = max_t - t
            pad_frames = torch.zeros(b, pad_t, c, h, w, dtype=torch.float32)
            pad_masks = torch.zeros(b, pad_t, dtype=torch.float32)
            pad_timestamps = torch.zeros(b, pad_t, dtype=torch.long)
            if t == 0:
                source_frames[source] = pad_frames
                source_masks[source] = pad_masks
                source_timestamps[source] = pad_timestamps
            else:
                source_frames[source] = torch.cat(
                    [source_frames[source], pad_frames], dim=1
                )
                source_masks[source] = torch.cat(
                    [source_masks[source], pad_masks], dim=1
                )
                source_timestamps[source] = torch.cat(
                    [source_timestamps[source], pad_timestamps], dim=1
                )
    else:
        batch_size = next(iter(source_frames.values())).shape[0]
        global_timestamps = torch.zeros(batch_size, 0, dtype=torch.long)

    # 分离所有以 highres 开头的高分辨率数据源。
    highres_frames: dict[str, torch.Tensor] = {}
    highres_masks: dict[str, torch.Tensor] = {}
    highres_sources = [s for s in source_frames if s.startswith("highres")]
    for source in highres_sources:
        hr_frames = source_frames.pop(source)
        hr_masks = source_masks.pop(source)
        source_timestamps.pop(source, None)
        if hr_frames.shape[1] > 0:
            highres_frame = _weighted_temporal_mean(hr_frames, hr_masks)
            _, _, height, width = highres_frame.shape
            avail = (hr_masks.sum(dim=1) > 0).float()
            highres_mask = avail[:, None, None, None].expand(-1, 1, height, width)
            highres_frames[source] = highres_frame
            highres_masks[source] = highres_mask

    return {
        "patch_ids": patch_ids,
        "source_frames": source_frames,
        "source_masks": source_masks,
        "timestamps": global_timestamps,
        "highres_frames": highres_frames,
        "highres_masks": highres_masks,
        "targets": targets,
        "target_masks": target_masks,
    }
