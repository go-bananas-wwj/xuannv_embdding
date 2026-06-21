from __future__ import annotations

# 将 dataset/collate 输出的按 source 组织 batch 转换为模型可用的统一 batch。
from typing import Any

import torch
import torch.nn.functional as F


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
        - ``timestamps`` 已由 ``collate_fn`` 合并为单一全局 ``(B, T_month)`` 张量。
        - 为每个 target head 构造逐月 ``targets`` 与 ``target_masks``。
        - 若存在名称以 ``highres`` 开头的高分辨率 source，将其各自聚合为单帧与
          可用性掩码并从时序源中移除。
        - 将 ``worldcover`` 等 target-only 源从喂给模型的 ``source_frames`` 中移除。

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
    global_timestamps = batch["timestamps"]
    patch_ids = batch["patch_ids"]

    if not source_frames:
        raise ValueError("source_frames 不能为空字典")

    if not isinstance(global_timestamps, torch.Tensor):
        raise TypeError("collate_fn 应返回单一全局 timestamps 张量")

    batch_size = global_timestamps.shape[0]
    num_months = global_timestamps.shape[1]

    # 使用第一个非高分辨率、非 target-only source 的空间尺寸作为参考分辨率。
    ref_source = next(
        (
            s
            for s in source_frames
            if not s.startswith("highres") and s != "worldcover"
        ),
        None,
    )
    if ref_source is None:
        # 仅有高分辨率或 worldcover 时，取第一个可用 source。
        ref_source = next(iter(source_frames))
    ref_frames = source_frames[ref_source]
    spatial_h = ref_frames.shape[-2]
    spatial_w = ref_frames.shape[-1]

    # 为每个 target head 构造逐月 target 与 target_mask。
    available_sources = set(source_frames.keys())
    targets: dict[str, torch.Tensor] = {}
    target_masks: dict[str, torch.Tensor] = {}

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
            t_in = frames.shape[1]

            if loss_type == "continuous":
                if source_name.startswith("highres"):
                    # 高分辨率 source 的观测不按月组织：先按时间掩码聚合为单帧，
                    # 下采样到与模型输出一致的空间尺寸，再复制到所有月度 bin；
                    # target_mask 反映该样本是否存在高分辨率观测。
                    highres_frame = _weighted_temporal_mean(frames, masks)  # (B, C, H_hr, W_hr)
                    target_frame = F.adaptive_avg_pool2d(
                        highres_frame, (spatial_h, spatial_w)
                    )  # (B, C, H, W)
                    target = target_frame.unsqueeze(1).expand(
                        -1, num_months, -1, -1, -1
                    ).clone()
                    sample_avail = (masks.sum(dim=1) > 0).float()  # (B,)
                    target_mask = sample_avail[:, None].expand(-1, num_months).clone()
                else:
                    # continuous head 直接使用逐月源帧作为目标。
                    target = frames
                    target_mask = masks
                    if target.shape[1] != num_months:
                        raise ValueError(
                            f"continuous head {head_name!r} 的目标时间维度 "
                            f"{target.shape[1]} 与 num_months {num_months} 不一致"
                        )
            else:  # categorical
                # 将 one-hot / 单通道标签转换为类别索引，形状 (B, T_in, H, W)。
                if frames.shape[2] == 1:
                    label = frames[:, :, 0].long()
                else:
                    label = frames.argmax(dim=2).long()

                # 静态标签（单时间步）复制到所有月份。
                if t_in == 1 and num_months > 1:
                    label = label.expand(-1, num_months, -1, -1)
                    target_mask = masks.expand(-1, num_months)
                else:
                    target_mask = masks

                if label.shape[1] != num_months:
                    raise ValueError(
                        f"categorical head {head_name!r} 的目标时间维度 "
                        f"{label.shape[1]} 与 num_months {num_months} 不一致"
                    )

                target = label
                # 以 0 作为 nodata/背景，不参与损失计算。
                target_mask = target_mask.unsqueeze(-1).unsqueeze(-1) * (target != 0).float()

            targets[head_name] = target
            target_masks[head_name] = target_mask
        else:
            if loss_type == "continuous":
                targets[head_name] = torch.zeros(
                    batch_size,
                    num_months,
                    channels,
                    spatial_h,
                    spatial_w,
                    dtype=torch.float32,
                )
            else:
                targets[head_name] = torch.zeros(
                    batch_size,
                    num_months,
                    spatial_h,
                    spatial_w,
                    dtype=torch.long,
                )
            target_masks[head_name] = torch.zeros(
                batch_size,
                num_months,
                dtype=torch.float32,
            )

    # 分离所有以 highres 开头的高分辨率数据源。
    highres_frames: dict[str, torch.Tensor] = {}
    highres_masks: dict[str, torch.Tensor] = {}
    highres_sources = [s for s in source_frames if s.startswith("highres")]
    for source in highres_sources:
        hr_frames = source_frames.pop(source)
        hr_masks = source_masks.pop(source)
        if hr_frames.shape[1] > 0:
            highres_frame = _weighted_temporal_mean(hr_frames, hr_masks)
            avail = (hr_masks.sum(dim=1) > 0).float()
            # 高分辨率 mask 最终会与编码到 (H, W) 的低分辨率特征融合，
            # 因此直接使用低分辨率参考尺寸即可；有高分数据时该空间掩码全 1。
            highres_mask = avail[:, None, None, None].expand(
                -1, 1, spatial_h, spatial_w
            )
            highres_frames[source] = highres_frame
            highres_masks[source] = highres_mask

    # 将 worldcover 等 target-only 源从模型输入中移除（模型不会注册它们）。
    for source in list(source_frames.keys()):
        if source == "worldcover":
            source_frames.pop(source)
            source_masks.pop(source)

    return {
        "patch_ids": patch_ids,
        "source_frames": source_frames,
        "source_masks": source_masks,
        "timestamps": global_timestamps,
        "highres_frames": highres_frames,
        "highres_masks": highres_masks,
        "targets": targets,
        "target_masks": target_masks,
        "teacher_embedding_map": batch.get("teacher_embedding_map"),
    }
