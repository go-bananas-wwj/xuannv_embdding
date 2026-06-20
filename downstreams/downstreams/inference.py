from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def load_model_for_inference(
    config_path: str | Path,
    checkpoint_path: str | Path | None,
    random_init: bool = False,
) -> tuple[AEFModel, Config, torch.device]:
    cfg = Config.from_yaml(config_path)
    device = get_device()

    aef_target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=aef_target_heads,
        stem_dim=cfg.model.stem_dim,
        num_months=cfg.model.num_months,
        stp=cfg.model.stp,
        gradient_checkpointing=False,  # 推理关闭
    )
    model = model.to(device)

    if random_init:
        logger.info("使用随机初始化 AEFModel（无预训练权重）")
    elif checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        logger.info("加载模型: %s", checkpoint_path)
    else:
        raise ValueError("checkpoint_path 与 random_init 不能同时为空")
    model.eval()
    return model, cfg, device


def build_inference_loader(cfg: Config, region: str) -> DataLoader:
    manifest_path = Path(cfg.data.root).parent / region / "manifest.json"
    statistics_dir = Path(cfg.data.root).parent / "statistics" / region
    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        num_months=cfg.model.num_months,
    )

    target_heads = cfg.model.target_heads

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return prepare_batch(collate_fn(batch), target_heads)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=False,
    )


def precompute_embeddings(
    model: AEFModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            patch_ids = batch["patch_ids"]
            source_frames = {
                k: v.to(device, non_blocking=True) for k, v in batch["source_frames"].items()
            }
            source_masks = {
                k: v.to(device, non_blocking=True) for k, v in batch["source_masks"].items()
            }
            timestamps = batch["timestamps"].to(device)
            highres_frames = None
            highres_masks = None
            if batch.get("highres_frames"):
                highres_frames = {
                    k: v.to(device, non_blocking=True) for k, v in batch["highres_frames"].items()
                }
                highres_masks = {
                    k: v.to(device, non_blocking=True) for k, v in batch["highres_masks"].items()
                }

            output = model(
                source_frames=source_frames,
                source_masks=source_masks,
                timestamps=timestamps,
                highres_frames=highres_frames,
                highres_masks=highres_masks,
            )

            # output.embedding_map: (B, T_month, D, H, W)
            # output.embedding:     (B, T_month, D)
            emb_map = output.embedding_map.cpu()
            scene_emb = output.embedding.cpu()
            ts = batch["timestamps"].cpu()  # (B, T_month)

            for b, patch_id in enumerate(patch_ids):
                patch_dir = output_dir / patch_id
                patch_dir.mkdir(parents=True, exist_ok=True)
                for m in range(emb_map.shape[1]):
                    month_int = int(ts[b, m].item())
                    assert 190000 < month_int < 210000, f"timestamp {month_int} 不是 YYYYMM 格式"
                    torch.save(emb_map[b, m], patch_dir / f"{month_int}_embedding_map.pt")
                    torch.save(scene_emb[b, m], patch_dir / f"{month_int}_scene_embedding.pt")


def write_meta_json(
    output_dir: Path,
    checkpoint_path: Path | None,
    config_path: Path,
    command_line: str,
) -> None:
    if checkpoint_path is not None:
        sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]
    else:
        sha = None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if commit == "":
        logger.warning("无法获取 git commit")
    dirty = (
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
        != ""
    )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command_line": command_line,
        "git_commit": commit,
        "git_dirty": dirty,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha,
        "config_path": str(config_path),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "torch_version": torch.__version__,
        "month_format": "YYYYMM",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
