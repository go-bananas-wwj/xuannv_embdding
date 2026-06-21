from __future__ import annotations

# 月度 batch 拼接：将 dataset 输出的固定月度 bin 张量沿 batch 维度堆叠。
from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """对月度 binned 的样本列表进行 batch 拼接。

    非高分辨率 source 已被 dataset 对齐到 ``num_months`` 个固定月度 bin，
    因此直接堆叠即可；高分辨率 source 仍可能具有不同的时序长度，需要补齐。
    所有非高分辨率 source 必须共享相同的月度 ``YYYYMM`` 时间戳，本函数会将其
    合并为单一全局 ``timestamps`` 张量。

    参数:
        batch: 由 ``MonthlyEmbeddingDataset.__getitem__`` 返回的样本列表。

    返回:
        拼接后的 batch 字典，包含 ``patch_ids``、``source_frames``、
        ``source_masks`` 与 ``timestamps``。
    """
    patch_ids = [item["patch_id"] for item in batch]
    source_names = list(batch[0]["source_frames"].keys())
    batch_size = len(batch)

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

        if source.startswith("highres"):
            # 高分辨率 source 保持原有变长时序补齐逻辑。
            max_t = max(frames.shape[0] for frames in frames_list)

            if max_t == 0:
                channels = frames_list[0].shape[1]
                height = frames_list[0].shape[2]
                width = frames_list[0].shape[3]
                batched_frames = torch.zeros(
                    batch_size, 0, channels, height, width, dtype=torch.float32
                )
                batched_masks = torch.zeros(batch_size, 0, dtype=torch.float32)
                batched_timestamps = torch.zeros(batch_size, 0, dtype=torch.long)
            else:
                padded_frames: list[torch.Tensor] = []
                padded_masks: list[torch.Tensor] = []
                padded_timestamps: list[torch.Tensor] = []

                for frames, masks, timestamps in zip(
                    frames_list, masks_list, timestamps_list
                ):
                    t = frames.shape[0]
                    if t < max_t:
                        pad_t = max_t - t
                        channels, height, width = frames.shape[1:]

                        pad_frames = torch.zeros(
                            pad_t, channels, height, width, dtype=torch.float32
                        )
                        pad_masks = torch.zeros(pad_t, dtype=torch.float32)
                        pad_timestamps = torch.zeros(pad_t, dtype=torch.long)

                        padded_frames.append(torch.cat([frames, pad_frames], dim=0))
                        padded_masks.append(torch.cat([masks, pad_masks], dim=0))
                        padded_timestamps.append(
                            torch.cat([timestamps, pad_timestamps], dim=0)
                        )
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
            continue

        # 非高分辨率 source：所有样本应具有相同的月度 bin 数与时间戳。
        frames_0 = frames_list[0]
        for frames in frames_list[1:]:
            if frames.shape[0] != frames_0.shape[0]:
                raise ValueError(
                    f"Source {source!r} 的月度 bin 数不一致: "
                    f"{frames.shape[0]} vs {frames_0.shape[0]}"
                )

        ts_0 = timestamps_list[0]
        for ts in timestamps_list[1:]:
            if not torch.equal(ts, ts_0):
                raise ValueError(
                    f"Source {source!r} 的月度时间戳不一致: {ts.tolist()} vs {ts_0.tolist()}"
                )

        collated["source_frames"][source] = torch.stack(frames_list)
        collated["source_masks"][source] = torch.stack(masks_list)
        collated["timestamps"][source] = torch.stack(timestamps_list)

    # 使用第一个非高分辨率 source 的时间戳作为全局月度时间戳。
    temporal_sources = [s for s in source_names if not s.startswith("highres")]
    if temporal_sources:
        global_timestamps = collated["timestamps"][temporal_sources[0]]
        for source in temporal_sources:
            if not torch.equal(
                collated["timestamps"][source], global_timestamps
            ):
                raise ValueError(
                    f"不同 source 的月度时间戳不一致: {source!r} vs {temporal_sources[0]!r}"
                )
        collated["timestamps"] = global_timestamps
    else:
        collated["timestamps"] = torch.zeros(batch_size, 0, dtype=torch.long)

    # 拼接教师 embedding map（如果存在）。
    if "teacher_embedding_map" in batch[0]:
        collated["teacher_embedding_map"] = torch.stack(
            [item["teacher_embedding_map"] for item in batch]
        )

    return collated
