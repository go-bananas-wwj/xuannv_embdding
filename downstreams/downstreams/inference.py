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
from torch.utils.data import DataLoader, Subset

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def manifest_provenance(manifest_path: Path) -> dict[str, str | int]:
    """Return content and patch-set hashes for an auditable export manifest."""
    manifest_path = Path(manifest_path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    patch_ids = [str(entry["patch_id"]) for entry in raw]
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError(f"Manifest contains duplicate patch IDs: {manifest_path}")
    patch_payload = "\n".join(sorted(patch_ids)).encode("utf-8")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "patch_count": len(patch_ids),
        "patch_ids_sha256": hashlib.sha256(patch_payload).hexdigest(),
    }


def validate_manifest_region(manifest_path: Path, expected_region: str) -> None:
    """Reject an explicit manifest whose records belong to another region."""
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    regions = {str(entry.get("region", "")) for entry in raw}
    if regions != {expected_region}:
        raise ValueError(
            f"Explicit manifest regions {sorted(regions)} do not match {expected_region!r}"
        )


def load_model_for_inference(
    config_path: str | Path,
    checkpoint_path: str | Path | None,
    random_init: bool = False,
    device_preference: str | None = None,
) -> tuple[AEFModel, Config, torch.device]:
    cfg = Config.from_yaml(config_path)
    device = get_device(device_preference)

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
        ref_year=getattr(cfg.model, "ref_year", 2025),
        ref_month=getattr(cfg.model, "ref_month", 1),
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


def build_inference_loader(
    cfg: Config,
    region: str,
    split: str = "all",
    context_margin: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
    manifest_path: Path | None = None,
) -> DataLoader:
    # TODO: 当 MonthlyEmbeddingDataset 支持 split 过滤时，根据 split 值筛选 patch。
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"推理 manifest 不存在: {manifest_path}")
        statistics_dir = cfg.data.statistics_dir or (
            Path(cfg.data.root).parent / "statistics" / region
        )
        statistics_dirs_by_region = cfg.data.statistics_dirs_by_region
        region_filter = None
    elif cfg.data.statistics_dirs_by_region:
        manifest_path = cfg.data.manifest_path
        statistics_dir = cfg.data.statistics_dir
        statistics_dirs_by_region = cfg.data.statistics_dirs_by_region
        region_filter = region
    elif region == cfg.data.region and cfg.data.manifest_path.exists():
        manifest_path = cfg.data.manifest_path
        statistics_dir = cfg.data.statistics_dir or (
            Path(cfg.data.root).parent / "statistics" / region
        )
        statistics_dirs_by_region = {}
        region_filter = None
    else:
        manifest_path = Path(cfg.data.root).parent / region / "manifest.json"
        statistics_dir = Path(cfg.data.root).parent / "statistics" / region
        statistics_dirs_by_region = {}
        region_filter = None
    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=statistics_dir,
        statistics_dirs_by_region=statistics_dirs_by_region,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        context_margin=getattr(cfg.data, "context_margin", 0)
        if context_margin is None
        else context_margin,
        patch_grid_path=getattr(cfg.data, "patch_grid_path", None),
        num_months=cfg.model.num_months,
        ref_year=getattr(cfg.model, "ref_year", 2025),
        ref_month=getattr(cfg.model, "ref_month", 1),
        region_filter=region_filter,
    )
    if num_shards is not None:
        if shard_id is None:
            raise ValueError("设置 num_shards 时必须同时设置 shard_id")
        if num_shards <= 0:
            raise ValueError(f"num_shards 必须为正数，当前为 {num_shards}")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"shard_id 必须位于 [0, {num_shards})，当前为 {shard_id}")
        indices = list(range(shard_id, len(dataset), num_shards))
        dataset = Subset(dataset, indices)

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
    months: list[str] | None = None,
    center_crop_size: int | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_months = {str(month) for month in months} if months is not None else None
    produced_patch_ids: list[str] = []
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
            emb_map_tensor = output.embedding_map
            if center_crop_size is not None:
                crop = int(center_crop_size)
                height, width = emb_map_tensor.shape[-2:]
                if height < crop or width < crop:
                    raise ValueError(
                        f"无法将 embedding_map 从 {(height, width)} 中心裁剪到 {crop}"
                    )
                top = (height - crop) // 2
                left = (width - crop) // 2
                emb_map_tensor = emb_map_tensor[..., top : top + crop, left : left + crop]
            emb_map = emb_map_tensor.cpu()
            scene_emb = torch.nn.functional.normalize(
                emb_map_tensor.mean(dim=[3, 4]),
                p=2,
                dim=-1,
            ).cpu()
            ts = batch["timestamps"].cpu()  # (B, T_month)

            for b, patch_id in enumerate(patch_ids):
                produced_patch_ids.append(str(patch_id))
                patch_dir = output_dir / patch_id
                patch_dir.mkdir(parents=True, exist_ok=True)
                for m in range(emb_map.shape[1]):
                    month_int = int(ts[b, m].item())
                    assert 190000 < month_int < 210000, f"timestamp {month_int} 不是 YYYYMM 格式"
                    if selected_months is not None and str(month_int) not in selected_months:
                        continue
                    month_map = emb_map[b, m].contiguous().clone()
                    month_scene = scene_emb[b, m].contiguous().clone()
                    torch.save(month_map, patch_dir / f"{month_int}_embedding_map.pt")
                    torch.save(month_scene, patch_dir / f"{month_int}_scene_embedding.pt")
    return produced_patch_ids


def write_meta_json(
    output_dir: Path,
    checkpoint_path: Path | None,
    config_path: Path,
    command_line: str,
    manifest_path: Path | None = None,
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
    if manifest_path is not None:
        meta["manifest"] = manifest_provenance(manifest_path)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
