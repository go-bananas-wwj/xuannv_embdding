#!/usr/bin/env python3
from __future__ import annotations

# DDP / 单卡训练入口。
import argparse
import hashlib
import json
import logging
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

from xuannv_embedding.config import Config, ConfigError
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.data.sampler import DistributedWeightedSampler
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.trainer import Trainer
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def reject_base_config(config_path: str | Path) -> None:
    """Reject `_base_` in actual training configs so experiments are self-contained."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, dict) and "_base_" in raw:
        raise ConfigError(
            f"实际训练配置必须自包含，禁止使用 `_base_`: {path}. " "请复制完整配置后直接修改字段。"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 可选参数列表，用于测试；为 ``None`` 时解析 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="AEF 月度地理嵌入训练入口")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="待恢复的 checkpoint 路径（恢复 optimizer/scheduler/epoch）",
    )
    parser.add_argument(
        "--init-from",
        dest="init_from",
        type=str,
        default=None,
        help="仅加载模型权重的 checkpoint 路径（用于阶段二从阶段一初始化）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖自动设备选择，例如 cpu / cuda:0 / npu:0",
    )
    return parser.parse_args(argv)


def setup_distributed() -> tuple[bool, int]:
    """若由 ``torchrun`` 启动，则初始化分布式进程组。

    返回:
        (是否进入分布式模式, local_rank)。
    """
    if "RANK" not in os.environ:
        return False, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "gloo"
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            backend = "hccl"
    except ImportError:
        pass

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if backend == "hccl":
        torch.npu.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True, local_rank


def _set_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu

        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed_all(seed)
    except ImportError:
        pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_registered_paper_v5_sidecars(config_path: str | Path) -> None:
    """Verify v5 manifest/statistics provenance before loading any training data."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
        raise ConfigError("训练配置缺少 data 段")
    data = raw["data"]
    if data.get("paper_protocol_name") != "rse_v5_registered_20260726":
        return
    required = (
        "paper_spatial_split",
        "paper_spatial_split_sha256",
        "paper_subset_registry",
        "paper_subset_registry_sha256",
        "paper_manifest_audit",
        "paper_manifest_audit_sha256",
        "paper_normalization_statistics",
        "paper_normalization_statistics_sha256",
        "paper_normalization_statistics_audit",
        "paper_normalization_statistics_audit_sha256",
        "paper_fold",
        "train_manifest_path",
        "val_manifest_path",
        "statistics_dir",
    )
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise ConfigError(f"v5 注册论文配置缺少 provenance 字段: {', '.join(missing)}")

    split_path = Path(data["paper_spatial_split"])
    subset_path = Path(data["paper_subset_registry"])
    audit_path = Path(data["paper_manifest_audit"])
    statistics_path = Path(data["paper_normalization_statistics"])
    expected_hashes = (
        ("paper_spatial_split", split_path, data["paper_spatial_split_sha256"]),
        ("paper_subset_registry", subset_path, data["paper_subset_registry_sha256"]),
        ("paper_manifest_audit", audit_path, data["paper_manifest_audit_sha256"]),
        (
            "paper_normalization_statistics",
            statistics_path,
            data["paper_normalization_statistics_sha256"],
        ),
    )
    for name, path, expected_hash in expected_hashes:
        if not path.exists() or _sha256(path) != expected_hash:
            raise ConfigError(f"v5 {name} 哈希不匹配")

    split_hash = _sha256(split_path)
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    fold_id = int(data["paper_fold"])
    if subset.get("source_split_sha256") != split_hash:
        raise ConfigError("v5 subset registry 与 spatial split 不匹配")
    if audit.get("spatial_split_sha256") != split_hash:
        raise ConfigError("v5 manifest audit 与 spatial split 不匹配")
    if audit.get("subset_registry_sha256") != _sha256(subset_path):
        raise ConfigError("v5 manifest audit 与 subset registry 不匹配")
    if audit.get("fold") != fold_id:
        raise ConfigError("v5 manifest audit fold 不匹配")
    if statistics.get("source_split_sha256") != split_hash:
        raise ConfigError("v5 normalization statistics 与 spatial split 不匹配")
    fold_statistics = statistics.get("folds", {}).get(str(fold_id))
    if not isinstance(fold_statistics, dict):
        raise ConfigError("v5 normalization statistics 缺少 fold")
    if fold_statistics.get("manifest_audit_sha256") != _sha256(audit_path):
        raise ConfigError("v5 normalization statistics 与 manifest audit 不匹配")
    train_pool = audit.get("manifests", {}).get("train_pool", {})
    if fold_statistics.get("source_manifest") != train_pool.get("path") or fold_statistics.get(
        "source_manifest_sha256"
    ) != train_pool.get("sha256"):
        raise ConfigError("v5 normalization statistics source manifest 不匹配")
    if Path(data["statistics_dir"]) != Path(str(fold_statistics.get("statistics_dir"))):
        raise ConfigError("v5 statistics_dir 不等于注册 fold-only statistics 目录")
    statistics_audit_path = Path(data["paper_normalization_statistics_audit"])
    if not statistics_audit_path.exists():
        raise ConfigError("v5 fold-only statistics audit 不存在")
    statistics_audit = json.loads(statistics_audit_path.read_text(encoding="utf-8"))
    if statistics_audit.get("source_manifest") != fold_statistics.get(
        "source_manifest"
    ) or statistics_audit.get("source_manifest_sha256") != fold_statistics.get(
        "source_manifest_sha256"
    ):
        raise ConfigError("v5 fold-only statistics audit source manifest 不匹配")
    statistics_files = statistics_audit.get("statistics_files")
    if not isinstance(statistics_files, dict) or not statistics_files:
        raise ConfigError("v5 fold-only statistics audit 缺少 statistics files")
    configured_sources = data.get("sources")
    if not isinstance(configured_sources, list) or not all(
        isinstance(source, str) and source for source in configured_sources
    ):
        raise ConfigError("v5 config data.sources 无效")
    expected_sources = set(configured_sources)
    if (
        len(expected_sources) != len(configured_sources)
        or set(statistics_files) != expected_sources
    ):
        raise ConfigError("v5 statistics audit sources do not match configured sources")
    statistics_dir = Path(data["statistics_dir"]).resolve()
    for source, record in statistics_files.items():
        if not isinstance(record, dict):
            raise ConfigError(f"v5 statistics file record 无效: {source}")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_relative_to(statistics_dir):
            raise ConfigError(f"v5 statistics file is outside statistics_dir: {source}")
        if not path.exists() or record.get("sha256") != _sha256(path):
            raise ConfigError(f"v5 statistics file 哈希不匹配: {source}")
    if fold_statistics.get("status") != "materialized":
        raise ConfigError("v5 normalization statistics fold is not materialized")
    source_key = "__".join(sorted(expected_sources))
    registry_audit = fold_statistics.get("statistics_audits", {}).get(source_key)
    if not isinstance(registry_audit, dict):
        raise ConfigError("v5 normalization statistics has no audit for configured sources")
    actual_audit_hash = _sha256(statistics_audit_path)
    if (
        registry_audit.get("path") != str(statistics_audit_path)
        or registry_audit.get("sha256") != actual_audit_hash
        or registry_audit.get("sources") != sorted(expected_sources)
        or data["paper_normalization_statistics_audit_sha256"] != actual_audit_hash
    ):
        raise ConfigError("v5 normalization statistics audit hash mismatch")

    manifests = audit.get("manifests", {})
    budget = len(json.loads(Path(data["train_manifest_path"]).read_text(encoding="utf-8")))
    expected_train = manifests.get(f"train_{budget}", {})
    expected_val = manifests.get("val", {})
    if (
        expected_train.get("path") != data["train_manifest_path"]
        or expected_train.get("sha256") != _sha256(Path(data["train_manifest_path"]))
        or expected_val.get("path") != data["val_manifest_path"]
        or expected_val.get("sha256") != _sha256(Path(data["val_manifest_path"]))
    ):
        raise ConfigError("v5 config manifests do not match the manifest audit sidecar")
    overlap = audit.get("overlap", {})
    if any(overlap.values()) or not audit.get("strictly_nested"):
        raise ConfigError("v5 manifest audit contains held-out geography overlap")


def validate_registered_paper_manifests(cfg: Config) -> None:
    """Fail closed when a registered paper config does not match its frozen subsets."""
    registry_path = cfg.data.paper_subset_registry
    if registry_path is None:
        return
    split_path = cfg.data.paper_spatial_split
    fold_id = cfg.data.paper_fold
    train_path = cfg.data.train_manifest_path
    if split_path is None or fold_id is None or train_path is None:
        raise ConfigError(
            "注册论文配置必须同时提供 paper_spatial_split、paper_fold 和 train_manifest_path"
        )

    expected_hash = cfg.data.paper_subset_registry_sha256
    actual_hash = _sha256(registry_path)
    if expected_hash is None or actual_hash != expected_hash:
        raise ConfigError(f"论文 subset registry 哈希不匹配: {actual_hash} != {expected_hash}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    split_hash = _sha256(split_path)
    if registry.get("source_split_sha256") != split_hash:
        raise ConfigError("论文 subset registry 与 spatial split 不匹配")

    records = json.loads(train_path.read_text(encoding="utf-8"))
    actual_ids = [str(record["patch_id"]) for record in records]
    budget = str(len(actual_ids))
    expected_ids = registry.get("folds", {}).get(str(fold_id), {}).get(budget)
    if expected_ids is None or actual_ids != expected_ids:
        raise ConfigError(f"训练 manifest 不等于注册子集: fold={fold_id}, budget={budget}")

    if cfg.data.num_samples is not None and len(actual_ids) != cfg.data.num_samples:
        raise ConfigError("训练 manifest 样本数与 data.num_samples 不一致")

    if cfg.data.val_manifest_path is not None:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        expected_val = split["folds"][fold_id]["val"]
        val_records = json.loads(cfg.data.val_manifest_path.read_text(encoding="utf-8"))
        actual_val = [str(record["patch_id"]) for record in val_records]
        if actual_val != expected_val:
            raise ConfigError(f"验证 manifest 不等于注册空间验证集: fold={fold_id}")


def _resolve_device(args_device: str | None, is_distributed: bool) -> torch.device:
    """根据用户参数与分布式环境解析训练设备。

    分布式模式下固定按 ``LOCAL_RANK`` 选择设备，忽略用户传入的 ``--device``，
    避免所有 rank 被绑定到同一设备导致 DDP 失败。
    """
    if not is_distributed:
        if args_device is not None:
            return get_device(args_device)
        return get_device()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = dist.get_backend() if dist.is_initialized() else "gloo"
    if backend == "hccl":
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


class _PreparedLoader:
    """Apply stochastic batch preparation in the rank process.

    DataLoader workers are responsible only for file IO and CPU collation.
    Running masking and other torch tensor transforms in those workers can
    inherit an initialized NPU runtime and make DDP ranks progress at wildly
    different speeds.
    """

    def __init__(
        self,
        loader: DataLoader,
        target_heads: dict[str, dict[str, Any]],
        source_dropout_probs: dict[str, float],
        input_masking: dict[str, Any],
        keep_highres_inputs: bool,
    ) -> None:
        self.loader = loader
        self.sampler = loader.sampler
        self.target_heads = target_heads
        self.source_dropout_probs = source_dropout_probs
        self.input_masking = input_masking
        self.keep_highres_inputs = keep_highres_inputs

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self.loader:
            yield prepare_batch(
                batch,
                self.target_heads,
                source_dropout_probs=self.source_dropout_probs,
                input_masking=self.input_masking,
                keep_highres_inputs=self.keep_highres_inputs,
            )

    def __len__(self) -> int:
        return len(self.loader)


def _build_loader(
    cfg: Config,
    target_heads: dict[str, dict[str, Any]],
    split: str,
    is_distributed: bool,
) -> _PreparedLoader:
    """构造训练或验证 DataLoader，内置 ``prepare_batch`` 转换。"""
    manifest_path = cfg.data.manifest_path
    if split == "train" and cfg.data.train_manifest_path is not None:
        manifest_path = cfg.data.train_manifest_path
    if split == "val" and cfg.data.val_manifest_path is not None:
        manifest_path = cfg.data.val_manifest_path

    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=cfg.data.statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        context_margin=cfg.data.context_margin,
        patch_grid_path=cfg.data.patch_grid_path,
        max_patches=cfg.data.max_patches,
        num_months=cfg.model.num_months,
        ref_year=cfg.model.ref_year,
        ref_month=cfg.model.ref_month,
        statistics_dirs_by_region=cfg.data.statistics_dirs_by_region,
        supervised_label_roots=cfg.data.supervised_label_roots,
        teacher_feature_root=cfg.data.teacher_feature_root,
    )

    sampler = None
    sampling_cfg = cfg.data.supervised_sampling or {}
    use_supervised_sampling = split == "train" and bool(sampling_cfg.get("enabled", False))
    if use_supervised_sampling:
        weights = dataset.compute_supervised_sampling_weights(
            positive_boost=float(sampling_cfg.get("positive_boost", 4.0)),
            max_weight=float(sampling_cfg.get("max_weight", 8.0)),
            task_weights=sampling_cfg.get("task_weights", {}),
            region_weights=sampling_cfg.get("region_weights", {}),
        )
        if is_distributed:
            sampler = DistributedWeightedSampler(weights, seed=cfg.experiment.seed)
        else:
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(dataset),
                replacement=True,
            )
        if not is_distributed or dist.get_rank() == 0:
            logger.info(
                "启用监督正样本采样: min=%.3f max=%.3f mean=%.3f positive_weighted=%d/%d",
                float(weights.min().item()),
                float(weights.max().item()),
                float(weights.mean().item()),
                int((weights > 1.0).sum().item()),
                len(weights),
            )
    elif is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=(split == "train"),
        )

    shuffle = (split == "train") and (sampler is None)
    drop_last = split == "train"

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
    return _PreparedLoader(
        loader,
        target_heads,
        cfg.data.source_dropout_probs if split == "train" else {},
        cfg.training.input_masking if split == "train" else {},
        bool(cfg.model.stp.get("highres_fusion_to_embedding", True)),
    )


def _init_model_from_checkpoint(
    model: nn.Module,
    init_from: str,
    is_main_process: bool,
) -> None:
    """仅加载模型权重，用于阶段二从阶段一 checkpoint 初始化。

    不恢复 optimizer、scheduler 与 epoch；阶段二使用全新的训练状态。
    """
    state = torch.load(init_from, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)

    if missing or unexpected:
        missing_preview = ", ".join(missing[:20])
        unexpected_preview = ", ".join(unexpected[:20])
        raise RuntimeError(
            "初始化 checkpoint 与当前模型结构不完全匹配，已中止以避免半加载训练。"
            f" missing={len(missing)} [{missing_preview}],"
            f" unexpected={len(unexpected)} [{unexpected_preview}]"
        )

    if is_main_process:
        logger.info("从 %s 初始化模型权重", init_from)


def main() -> None:
    """训练入口主函数。"""
    args = parse_args()
    is_distributed, local_rank = setup_distributed()

    reject_base_config(args.config)
    validate_registered_paper_v5_sidecars(args.config)
    cfg = Config.from_yaml(args.config)
    validate_registered_paper_manifests(cfg)
    _set_seed(cfg.experiment.seed)

    device = _resolve_device(args.device, is_distributed)
    if is_distributed and device.type == "cuda":
        torch.cuda.set_device(local_rank)

    if not is_distributed or dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        logger.info("实验名称: %s", cfg.experiment.name)
        logger.info("配置文件: %s", Path(args.config).resolve())
        logger.info("训练设备: %s", device)
        logger.info("分布式模式: %s", is_distributed)

    # 构造 AEFModel 所需的 target_heads: {name -> (kind, channels)}。
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
        ref_year=cfg.model.ref_year,
        ref_month=cfg.model.ref_month,
        stp=cfg.model.stp,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )

    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume 与 --init-from 不能同时使用")

    is_main = not is_distributed or dist.get_rank() == 0
    if args.init_from is not None:
        _init_model_from_checkpoint(model, args.init_from, is_main)

    # 构造 TotalLoss 所需的 target_cfg。
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
        uniformity_weight=cfg.training.uniformity_weight,
        uniformity_warmup_epochs=cfg.training.uniformity_warmup_epochs,
        uniformity_temperature=cfg.training.uniformity_temperature,
        covariance_weight=cfg.training.covariance_weight,
        covariance_warmup_epochs=cfg.training.covariance_warmup_epochs,
        covariance_std_target=cfg.training.covariance_std_target,
        covariance_pool_size=cfg.training.covariance_pool_size,
        patch_discrimination_weight=cfg.training.patch_discrimination_weight,
        patch_discrimination_warmup_epochs=(cfg.training.patch_discrimination_warmup_epochs),
        patch_discrimination_temperature=(cfg.training.patch_discrimination_temperature),
        patch_discrimination_pool_size=cfg.training.patch_discrimination_pool_size,
        patch_discrimination_max_tokens=cfg.training.patch_discrimination_max_tokens,
        temporal_endpoint_weight=cfg.training.temporal_endpoint_weight,
        temporal_endpoint_warmup_epochs=cfg.training.temporal_endpoint_warmup_epochs,
        temporal_endpoint_margin=cfg.training.temporal_endpoint_margin,
        temporal_contrast_weight=cfg.training.temporal_contrast_weight,
        temporal_contrast_warmup_epochs=cfg.training.temporal_contrast_warmup_epochs,
        temporal_contrast_margin=cfg.training.temporal_contrast_margin,
        temporal_contrast_change_z=cfg.training.temporal_contrast_change_z,
        temporal_contrast_stable_z=cfg.training.temporal_contrast_stable_z,
        temporal_contrast_sources=cfg.training.temporal_contrast_sources,
        supervised_change_weight=cfg.training.supervised_change_weight,
        supervised_change_warmup_epochs=cfg.training.supervised_change_warmup_epochs,
        supervised_change_pos_margin=cfg.training.supervised_change_pos_margin,
        supervised_change_neg_margin=cfg.training.supervised_change_neg_margin,
        supervised_change_tasks=cfg.training.supervised_change_tasks,
        supervised_change_pos_weight=cfg.training.supervised_change_pos_weight,
        supervised_change_neg_weight=cfg.training.supervised_change_neg_weight,
        supervised_change_hard_negative_ratio=(cfg.training.supervised_change_hard_negative_ratio),
        supervised_change_task_weights=cfg.training.supervised_change_task_weights,
        semantic_probe_embed_dim=cfg.model.embed_dim,
        semantic_probe_weight=cfg.training.semantic_probe_weight,
        semantic_probe_warmup_epochs=cfg.training.semantic_probe_warmup_epochs,
        semantic_probe_tasks=cfg.training.semantic_probe_tasks,
        semantic_probe_task_weights=cfg.training.semantic_probe_task_weights,
        semantic_probe_pos_weight=cfg.training.semantic_probe_pos_weight,
        semantic_probe_pos_weights=cfg.training.semantic_probe_pos_weights,
        semantic_probe_hidden_dim=cfg.training.semantic_probe_hidden_dim,
        semantic_probe_hard_negative_ratio=(cfg.training.semantic_probe_hard_negative_ratio),
        semantic_probe_hard_negative_weight=(cfg.training.semantic_probe_hard_negative_weight),
        semantic_probe_hard_negative_warmup_epochs=(
            cfg.training.semantic_probe_hard_negative_warmup_epochs
        ),
        distill_embed_dim=cfg.model.embed_dim,
        distill_weight=cfg.training.distill_weight,
        distill_gram_weight=cfg.training.distill_gram_weight,
        distill_warmup_epochs=cfg.training.distill_warmup_epochs,
        distill_final_weight=cfg.training.distill_final_weight,
        distill_decay_start_epoch=cfg.training.distill_decay_start_epoch,
        distill_decay_end_epoch=cfg.training.distill_decay_end_epoch,
        distill_teacher_dim=cfg.training.distill_teacher_dim,
        distill_max_tokens=cfg.training.distill_max_tokens,
        prototype_contrast_embed_dim=cfg.model.embed_dim,
        prototype_contrast_weight=cfg.training.prototype_contrast_weight,
        prototype_contrast_warmup_epochs=(cfg.training.prototype_contrast_warmup_epochs),
        prototype_contrast_tasks=cfg.training.prototype_contrast_tasks,
        prototype_contrast_task_weights=cfg.training.prototype_contrast_task_weights,
        prototype_contrast_negative_margin=(cfg.training.prototype_contrast_negative_margin),
        prototype_contrast_separation_margin=(cfg.training.prototype_contrast_separation_margin),
        prototype_contrast_max_pixels=cfg.training.prototype_contrast_max_pixels,
        boundary_contrast_weight=cfg.training.boundary_contrast_weight,
        boundary_contrast_warmup_epochs=cfg.training.boundary_contrast_warmup_epochs,
        boundary_contrast_tasks=cfg.training.boundary_contrast_tasks,
        boundary_contrast_band_radius=cfg.training.boundary_contrast_band_radius,
        boundary_contrast_margin=cfg.training.boundary_contrast_margin,
        boundary_contrast_max_pixels=cfg.training.boundary_contrast_max_pixels,
        latent_reconstruction_embed_dim=cfg.model.embed_dim,
        latent_reconstruction_weight=cfg.training.latent_reconstruction_weight,
        latent_reconstruction_warmup_epochs=(cfg.training.latent_reconstruction_warmup_epochs),
        latent_reconstruction_targets=cfg.training.latent_reconstruction_targets,
        latent_reconstruction_target_weights=(cfg.training.latent_reconstruction_target_weights),
        latent_reconstruction_hidden_dim=(cfg.training.latent_reconstruction_hidden_dim),
        latent_reconstruction_loss_type=cfg.training.latent_reconstruction_loss_type,
        loss_crop_size=cfg.training.loss_crop_size,
    )

    train_loader = _build_loader(
        cfg,
        cfg.model.target_heads,
        split="train",
        is_distributed=is_distributed,
    )
    val_loader = _build_loader(
        cfg,
        cfg.model.target_heads,
        split="val",
        is_distributed=is_distributed,
    )

    train_sampler = train_loader.sampler if hasattr(train_loader.sampler, "set_epoch") else None

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        train_sampler=train_sampler,
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.fit()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
