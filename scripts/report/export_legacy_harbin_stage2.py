#!/usr/bin/env python3
"""Export one month from the legacy Harbin Stage 2 checkpoint.

Run this script with ``src`` and ``downstreams`` from git commit ``588c3e0``
at the front of ``PYTHONPATH``. The checkpoint predates the current STP and
upsampling implementations, so loading it with the current model code is not
valid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from downstreams.inference import build_inference_loader, load_model_for_inference
from torch.utils.data import DataLoader, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--region", default="harbin")
    parser.add_argument("--month", type=int, default=202604)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def shard_loader(
    loader: DataLoader,
    shard_id: int,
    num_shards: int,
    output_root: Path,
    month: int,
) -> DataLoader:
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    manifest = getattr(loader.dataset, "manifest", None)
    if manifest is None:
        raise AttributeError("Legacy dataset has no manifest for resume filtering")
    indices = []
    for index in range(shard_id, len(loader.dataset), num_shards):
        patch_id = str(manifest[index]["patch_id"])
        patch_dir = output_root / patch_id
        if (patch_dir / f"{month}_embedding_map.pt").exists() and (
            patch_dir / f"{month}_scene_embedding.pt"
        ).exists():
            continue
        indices.append(index)
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=False,
    )


def to_device_dict(
    values: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in values.items()}


def export_batch(
    model,
    batch: dict[str, Any],
    device: torch.device,
    output_root: Path,
    month: int,
) -> None:
    output = model(
        source_frames=to_device_dict(batch["source_frames"], device),
        source_masks=to_device_dict(batch["source_masks"], device),
        timestamps=batch["timestamps"].to(device),
        highres_frames=(
            to_device_dict(batch["highres_frames"], device) if batch.get("highres_frames") else None
        ),
        highres_masks=(
            to_device_dict(batch["highres_masks"], device) if batch.get("highres_masks") else None
        ),
    )
    embedding_map = output.embedding_map.detach().cpu()
    scene_embedding = output.embedding.detach().cpu()
    timestamps = batch["timestamps"].cpu()
    for batch_index, patch_id in enumerate(batch["patch_ids"]):
        month_indices = torch.where(timestamps[batch_index] == month)[0]
        if month_indices.numel() != 1:
            raise RuntimeError(
                f"{patch_id} expected exactly one {month} index, got {month_indices.tolist()}"
            )
        month_index = int(month_indices.item())
        patch_dir = output_root / patch_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            embedding_map[batch_index, month_index].contiguous().clone(),
            patch_dir / f"{month}_embedding_map.pt",
        )
        torch.save(
            scene_embedding[batch_index, month_index].contiguous().clone(),
            patch_dir / f"{month}_scene_embedding.pt",
        )


def batch_already_exported(batch: dict[str, Any], output_root: Path, month: int) -> bool:
    return all(
        (output_root / patch_id / f"{month}_embedding_map.pt").exists()
        and (output_root / patch_id / f"{month}_scene_embedding.pt").exists()
        for patch_id in batch["patch_ids"]
    )


def main() -> None:
    args = parse_args()
    model, config, device = load_model_for_inference(args.config, args.checkpoint)
    output_root = args.output_root / args.region
    output_root.mkdir(parents=True, exist_ok=True)
    loader = shard_loader(
        build_inference_loader(config, args.region),
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        output_root=output_root,
        month=args.month,
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch_already_exported(batch, output_root, args.month):
                continue
            export_batch(model, batch, device, output_root, args.month)


if __name__ == "__main__":
    main()
