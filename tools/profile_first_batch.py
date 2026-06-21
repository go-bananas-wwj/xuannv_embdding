#!/usr/bin/env python3
"""Profile a single training batch on one NPU to locate slowness/hang."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.training.losses import TotalLoss


def main() -> None:
    config_path = Path("configs/v1.1_distill_long_stable_50ep.yaml")
    cfg = Config.from_yaml(config_path)

    device = torch.device("npu:0")

    target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=target_heads,
        stem_dim=cfg.model.stem_dim,
        num_months=cfg.model.num_months,
        stp=cfg.model.stp,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    ).to(device)

    loss_type_map = {"continuous": "l1", "categorical": "ce"}
    target_cfg = {
        name: {
            "loss_type": loss_type_map[head_cfg["loss_type"]],
            "channels": head_cfg["channels"],
            "weight": head_cfg.get("weight", 1.0),
        }
        for name, head_cfg in cfg.model.target_heads.items()
    }
    criterion = TotalLoss(
        target_cfg,
        distill_weight=cfg.training.distill_weight,
        distill_months=cfg.training.distill_months,
    )

    dataset = MonthlyEmbeddingDataset(
        manifest_path=cfg.data.manifest_path,
        statistics_dir=cfg.data.statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        max_patches=cfg.data.max_patches,
        num_months=cfg.model.num_months,
        teacher_embedding_root=cfg.data.teacher_embedding_root,
        region=cfg.data.region,
        cache_dir=cfg.data.cache_dir,
    )

    def collate(batch):
        return prepare_batch(collate_fn(batch), cfg.model.target_heads)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )

    print(f"Dataset length: {len(dataset)}")
    print("Loading first batch (may trigger cache write)...")
    t0 = time.time()
    batch = next(iter(loader))
    t_load = time.time() - t0
    print(f"Data load time: {t_load:.2f}s")

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    batch = move(batch)

    torch.npu.synchronize(device)
    print("Forward pass...")
    t0 = time.time()
    output = model(
        source_frames=batch["source_frames"],
        source_masks=batch["source_masks"],
        timestamps=batch["timestamps"],
        highres_frames=batch.get("highres_frames"),
        highres_masks=batch.get("highres_masks"),
    )
    torch.npu.synchronize(device)
    t_forward = time.time() - t0
    print(f"Forward time: {t_forward:.2f}s")

    print("Loss computation...")
    t0 = time.time()
    losses = criterion(
        output,
        batch["targets"],
        batch["target_masks"],
        teacher_embedding_map=batch.get("teacher_embedding_map"),
    )
    torch.npu.synchronize(device)
    t_loss = time.time() - t0
    print(f"Loss time: {t_loss:.2f}s")

    print("Backward pass...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    optimizer.zero_grad()
    t0 = time.time()
    losses["total"].backward()
    torch.npu.synchronize(device)
    t_backward = time.time() - t0
    print(f"Backward time: {t_backward:.2f}s")

    mem_mb = torch.npu.memory_allocated(device) / 1e6
    print(f"NPU memory allocated: {mem_mb:.1f} MB")
    print(f"Loss total: {losses['total'].item():.4f}")


if __name__ == "__main__":
    main()
