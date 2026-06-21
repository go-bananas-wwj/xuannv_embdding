from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import torch

from downstreams.scripts.train_task import (
    JointMultiTaskEmbeddingDataset,
    _filter_patch_ids,
)


REGIONS = ("haidian", "harbin")
TASK = "construction"
MONTHS = ["202512", "202605"]
EMB_SHAPE = (64, 16, 16)


def _write_mask(mask_dir: Path, patch_id: str) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((EMB_SHAPE[-2], EMB_SHAPE[-1]), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=EMB_SHAPE[-2],
        width=EMB_SHAPE[-1],
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)


def _write_embedding(emb_root: Path, region: str, patch_id: str) -> None:
    emb_dir = emb_root / region / patch_id
    emb_dir.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        torch.save(
            torch.randn(*EMB_SHAPE),
            emb_dir / f"{month}_embedding_map.pt",
        )


def _make_joint_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path], dict[str, Path], dict[str, Any]]:
    """构造联合训练数据布局。

    label_root 对应 ``--label-root`` 传入的父目录：
        {label_root}/{task}/split_joint_5fold.json
        {label_root}/{region}/labels/{task}/masks/{patch_id}.tif

    返回 (label_root, emb_root, current_mask_dirs, old_mask_dirs, split)。
    """
    label_root = tmp_path / "processed"
    emb_root = tmp_path / "embeddings"

    region_patch_ids: dict[str, list[str]] = {}
    for region in REGIONS:
        region_patch_ids[region] = [f"patch_{i:06d}" for i in range(3)]
        mask_dir = label_root / region / "labels" / TASK / "masks"
        for patch_id in region_patch_ids[region]:
            _write_mask(mask_dir, patch_id)
            _write_embedding(emb_root, region, patch_id)

    all_prefixed_ids = [
        f"{region}_{pid}" for region in REGIONS for pid in region_patch_ids[region]
    ]
    region_of = {pid: pid.split("_", 1)[0] for pid in all_prefixed_ids}

    split = {
        "seed": 42,
        "n_folds": 5,
        "val_ratio": 0.2,
        "stratify_by": "positive_pixel_ratio",
        "region_of": region_of,
        "folds": [
            {
                "fold": 0,
                "train": all_prefixed_ids,
                "val": [],
                "test": [],
            }
        ],
    }
    split_path = label_root / TASK / "split_joint_5fold.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    current_mask_dirs = {
        r: label_root / r / "labels" / TASK / "masks" for r in REGIONS
    }
    # 复现修复前的错误路径：向上跳两级后再按 region 组织。
    old_mask_dirs = {
        r: label_root.parent.parent / r / "labels" / TASK / "masks" for r in REGIONS
    }

    return label_root, emb_root, current_mask_dirs, old_mask_dirs, split


def test_current_mask_dirs_filter_non_empty(tmp_path: Path) -> None:
    """当前 mask_dirs 构造能正确过滤出可训练的 patch_id。"""
    _label_root, emb_root, current_mask_dirs, _old_mask_dirs, split = _make_joint_fixture(
        tmp_path
    )
    train_ids = split["folds"][0]["train"]
    filtered = _filter_patch_ids(
        train_ids,
        emb_root,
        MONTHS,
        region_of=split["region_of"],
        mask_dirs=current_mask_dirs,
    )
    assert len(filtered) == len(train_ids)


def test_old_mask_dirs_filter_empty(tmp_path: Path) -> None:
    """修复前的错误 mask_dirs 构造找不到 mask，应过滤为空。"""
    _label_root, emb_root, _current_mask_dirs, old_mask_dirs, split = _make_joint_fixture(
        tmp_path
    )
    train_ids = split["folds"][0]["train"]
    filtered = _filter_patch_ids(
        train_ids,
        emb_root,
        MONTHS,
        region_of=split["region_of"],
        mask_dirs=old_mask_dirs,
    )
    assert len(filtered) == 0


def test_joint_dataset_length_matches_filter(tmp_path: Path) -> None:
    """使用当前 mask_dirs 构造的 JointMultiTaskEmbeddingDataset 长度正确。"""
    _label_root, emb_root, current_mask_dirs, _old_mask_dirs, split = _make_joint_fixture(
        tmp_path
    )
    train_ids = split["folds"][0]["train"]
    ds = JointMultiTaskEmbeddingDataset(
        embedding_root=emb_root,
        mask_dirs=current_mask_dirs,
        patch_ids=train_ids,
        region_of=split["region_of"],
        task_name=TASK,
        months=MONTHS,
        bitemporal=True,
        include_diff=True,
    )
    assert len(ds) == len(train_ids)
    sample = ds[0]
    assert sample["embedding_map"].shape == (3 * EMB_SHAPE[0], *EMB_SHAPE[-2:])
    assert sample["mask"].shape == EMB_SHAPE[-2:]
    assert sample["patch_id"] in train_ids
