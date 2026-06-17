from __future__ import annotations

# 变长时序 batch 对齐
from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """对一个 batch 的样本进行时序长度补齐。

    不同样本同一 source 的时序长度 ``T`` 可能不同；本函数将其补齐到 batch 内最大 ``T``，
    缺失位置用 0 填充，并通过 ``source_masks`` 区分真实帧与填充帧。

    参数:
        batch: 由 ``MonthlyEmbeddingDataset.__getitem__`` 返回的样本列表。

    返回:
        拼接后的 batch 字典，包含 ``patch_ids``、``source_frames``、
        ``source_masks`` 与 ``timestamps``。
    """
    patch_ids = [item["patch_id"] for item in batch]
    source_names = list(batch[0]["source_frames"].keys())

    collated: dict[str, Any] = {
        "patch_ids": patch_ids,
        "source_frames": {},
        "source_masks": {},
        "timestamps": {},
    }

    for source in source_names:
        frames_list = [item["source_frames"][source] for item in batch]
        masks_list = [item["source_masks"][source] for item in batch]
        timestamps_list = [item["timestamps"][source] for item in batch]

        max_t = max(frames.shape[0] for frames in frames_list)

        if max_t == 0:
            # 该 source 在当前 batch 中完全缺失
            channels = frames_list[0].shape[1]
            height = frames_list[0].shape[2]
            width = frames_list[0].shape[3]
            batched_frames = torch.zeros(
                len(batch), 0, channels, height, width, dtype=torch.float32
            )
            batched_masks = torch.zeros(len(batch), 0, dtype=torch.float32)
            batched_timestamps = torch.zeros(len(batch), 0, dtype=torch.long)
        else:
            padded_frames: list[torch.Tensor] = []
            padded_masks: list[torch.Tensor] = []
            padded_timestamps: list[torch.Tensor] = []

            for frames, masks, timestamps in zip(frames_list, masks_list, timestamps_list):
                t = frames.shape[0]
                if t < max_t:
                    pad_t = max_t - t
                    channels, height, width = frames.shape[1:]

                    pad_frames = torch.zeros(pad_t, channels, height, width, dtype=torch.float32)
                    pad_masks = torch.zeros(pad_t, dtype=torch.float32)
                    pad_timestamps = torch.zeros(pad_t, dtype=torch.long)

                    padded_frames.append(torch.cat([frames, pad_frames], dim=0))
                    padded_masks.append(torch.cat([masks, pad_masks], dim=0))
                    padded_timestamps.append(torch.cat([timestamps, pad_timestamps], dim=0))
                else:
                    padded_frames.append(frames)
                    padded_masks.append(masks)
                    padded_timestamps.append(timestamps)

            batched_frames = torch.stack(padded_frames)
            batched_masks = torch.stack(padded_masks)
            batched_timestamps = torch.stack(padded_timestamps)

        collated["source_frames"][source] = batched_frames
        collated["source_masks"][source] = batched_masks
        collated["timestamps"][source] = batched_timestamps

    return collated
