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
        return head_name[: -len("_recon")]
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
    device: torch.device,
) -> dict[str, Any]:
    """将 collate 后的 batch 转换为 AEFModel / Trainer 需要的格式。

    主要转换:
        - 将 ``source_frames`` / ``source_masks`` 及构造出的统一 ``timestamps`` 移动到 device。
        - 从第一个有效的 source 提取全局 ``timestamps``（``[B, T]``）。
        - 若存在 ``highres`` source，将其聚合为单帧与可用性掩码并从时序源中移除。
        - 为每个 target head 构造 ``targets`` 与 ``target_masks``。

    参数:
        batch: ``collate_fn`` 输出，包含 ``patch_ids``、``source_frames``、
            ``source_masks``、``timestamps``。
        target_heads: 配置中的 target_heads，每个 head 至少包含
            ``loss_type``（``continuous`` / ``categorical``）与 ``channels``，
            可选 ``source`` 与 ``weight``。
        device: 目标设备。

    返回:
        转换后的 batch 字典，可直接喂给 ``Trainer``。
    """
    source_frames = batch["source_frames"]
    source_masks = batch["source_masks"]
    source_timestamps = batch["timestamps"]
    patch_ids = batch["patch_ids"]

    # 取第一个 T > 0 的 source 的时间戳作为全局 timestamps。
    global_timestamps: torch.Tensor | None = None
    for source, frames in source_frames.items():
        if frames.shape[1] > 0:
            global_timestamps = source_timestamps[source].to(device, non_blocking=True)
            break
    if global_timestamps is None:
        batch_size = next(iter(source_frames.values())).shape[0]
        global_timestamps = torch.zeros(
            batch_size,
            0,
            dtype=torch.long,
            device=device,
        )

    # 将时序源数据移动到目标设备。
    source_frames = {
        source: frames.to(device, non_blocking=True) for source, frames in source_frames.items()
    }
    source_masks = {
        source: masks.to(device, non_blocking=True) for source, masks in source_masks.items()
    }

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

    # 为每个 target head 构造 target 与 target_mask。
    available_sources = set(source_frames.keys())
    targets: dict[str, torch.Tensor] = {}
    target_masks: dict[str, torch.Tensor] = {}

    batch_size = next(iter(source_frames.values())).shape[0]
    spatial_h = next(iter(source_frames.values())).shape[-2]
    spatial_w = next(iter(source_frames.values())).shape[-1]

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
            height, width = agg.shape[2], agg.shape[3]
            target_mask = torch.ones(
                batch_size,
                height,
                width,
                dtype=torch.float32,
                device=device,
            )
            if loss_type == "continuous":
                targets[head_name] = agg
            else:  # categorical
                targets[head_name] = agg.argmax(dim=1).long()
            target_masks[head_name] = target_mask
        else:
            if loss_type == "continuous":
                targets[head_name] = torch.zeros(
                    batch_size,
                    channels,
                    spatial_h,
                    spatial_w,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                targets[head_name] = torch.zeros(
                    batch_size,
                    spatial_h,
                    spatial_w,
                    dtype=torch.long,
                    device=device,
                )
            target_masks[head_name] = torch.zeros(
                batch_size,
                spatial_h,
                spatial_w,
                dtype=torch.float32,
                device=device,
            )

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
