#!/usr/bin/env python3
"""Reproduce / verify the joint construction train_ids filtering bug.

Run without NPU:
    ASCEND_RT_VISIBLE_DEVICES="" python downstreams/scripts/reproduce_joint_filter.py

Expected output after the fix:
    raw fold-0 train ids: 162
    filtered with old mask_dirs: 0
    filtered with current mask_dirs: 70
    train dataset length: 70
    OK
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `downstreams` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstreams.scripts.train_task import _filter_patch_ids
from downstreams.data.multi_task_dataset import MultiTaskEmbeddingDataset

LABEL_ROOT = Path("/data/xuannv_embedding/processed")
TASK = "construction"
REGIONS = ["haidian", "harbin"]
MONTHS = ["202512", "202605"]
EMB_ROOT = Path("/data/xuannv_embedding/embeddings")
SPLIT_PATH = LABEL_ROOT / TASK / "split_joint_5fold.json"


def build_mask_dirs(label_root: Path, regions: list[str], task: str) -> dict[str, Path]:
    return {r: label_root / r / "labels" / task / "masks" for r in regions}


def main() -> None:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"split not found: {SPLIT_PATH}")

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    fold_info = split["folds"][0]
    region_of = split["region_of"]
    train_ids = fold_info["train"]
    print(f"raw fold-0 train ids: {len(train_ids)}")

    # Old / buggy construction used in benchmark_heads.py / train_task.py
    old_mask_dirs = {r: LABEL_ROOT.parent.parent / r / "labels" / TASK / "masks" for r in REGIONS}
    filtered_old = _filter_patch_ids(
        train_ids, EMB_ROOT, MONTHS, region_of=region_of, mask_dirs=old_mask_dirs
    )
    print(f"filtered with old mask_dirs: {len(filtered_old)}")

    # Current / fixed construction
    mask_dirs = build_mask_dirs(LABEL_ROOT, REGIONS, TASK)
    filtered = _filter_patch_ids(
        train_ids, EMB_ROOT, MONTHS, region_of=region_of, mask_dirs=mask_dirs
    )
    print(f"filtered with current mask_dirs: {len(filtered)}")

    if len(filtered) == 0:
        raise RuntimeError("filtered train_ids is empty -- fix not applied?")

    # Construct the dataset to ensure JointMultiTaskEmbeddingDataset can resolve items.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from downstreams.scripts.train_task import JointMultiTaskEmbeddingDataset

    ds = JointMultiTaskEmbeddingDataset(
        embedding_root=EMB_ROOT,
        mask_dirs=mask_dirs,
        patch_ids=filtered,
        region_of=region_of,
        task_name=TASK,
        months=MONTHS,
        bitemporal=True,
        include_diff=True,
    )
    print(f"train dataset length: {len(ds)}")

    if len(ds) == 0:
        raise RuntimeError("dataset length is 0")

    print("OK")


if __name__ == "__main__":
    main()
