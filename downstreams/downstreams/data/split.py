from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from sklearn.model_selection import KFold, StratifiedKFold


def _positive_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum() / total)


def _quantize(values: np.ndarray, n_bins: int) -> np.ndarray:
    """将连续值按分位数分桶为 0..n_bins-1 的整数 strata。"""
    if len(values) == 0 or n_bins <= 1:
        return np.zeros(len(values), dtype=int)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(values, quantiles))
    if len(bins) < 2:
        return np.zeros(len(values), dtype=int)
    # 使用内部断点（不包含最小值），避免最小值被分到 -1
    return np.digitize(values, bins[1:-1])


def create_stratified_folds(
    mask_dir: Path,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """按 mask 正像素比例分层，生成 5-fold。

    每 fold：4 folds 训练，1 fold 测试；训练集内部再切 val_ratio 做验证。
    """
    mask_paths = sorted(mask_dir.glob("*.tif"))
    patch_ids = [p.stem for p in mask_paths]
    ratios = np.array([_positive_ratio(p) for p in mask_paths])
    pid_to_idx = {pid: i for i, pid in enumerate(patch_ids)}

    # 分层：按正像素比例分桶
    strata = _quantize(ratios, n_bins=4)

    # 若任一 stratum 样本数不足 n_folds，则无法按 strata 分层，退化为非分层 KFold
    _, counts = np.unique(strata, return_counts=True)
    use_stratified = counts.min() >= n_folds

    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        splitter.split(patch_ids, strata if use_stratified else None)
    ):
        train_val_ids = [patch_ids[i] for i in train_val_idx]
        test_ids = [patch_ids[i] for i in test_idx]

        rng = np.random.default_rng(seed + fold_idx)
        n_val = max(1, int(len(train_val_ids) * val_ratio))
        val_indices = rng.choice(len(train_val_ids), size=n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(train_val_ids)), val_indices)

        train_ids = [train_val_ids[i] for i in train_indices]
        val_ids = [train_val_ids[i] for i in val_indices]

        folds.append({"fold": fold_idx, "train": train_ids, "val": val_ids, "test": test_ids})

    # 生成 10/25/50/100% 标签比例子集
    frac_offsets = {"0.1": 11, "0.25": 22, "0.5": 33, "1.0": 44}
    fractions = {"0.1": {}, "0.25": {}, "0.5": {}, "1.0": {}}
    for fold in folds:
        train_ids = fold["train"]
        train_ratios = np.array([ratios[pid_to_idx[pid]] for pid in train_ids])
        train_strata = _quantize(train_ratios, n_bins=2)

        for frac_str, frac in [("0.1", 0.1), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0)]:
            n = max(1, int(len(train_ids) * frac))
            frac_seed = seed + fold["fold"] * 7 + frac_offsets[frac_str]
            selected = _stratified_sample(train_ids, train_strata, n, frac_seed)
            fractions[frac_str][f"fold_{fold['fold']}"] = selected

    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_ratio": val_ratio,
        "stratify_by": "positive_pixel_ratio",
        "folds": folds,
        "fractions": fractions,
    }


def _stratified_sample(ids: list[str], strata: np.ndarray, n: int, seed: int) -> list[str]:
    assert len(ids) == len(strata), "ids 与 strata 长度必须一致"
    assert len(ids) == len(set(ids)), "ids 必须唯一"
    rng = np.random.default_rng(seed)
    n = int(n)
    n = min(n, len(ids))
    if n >= len(ids):
        return list(ids)

    unique_strata, counts = np.unique(strata, return_counts=True)
    total = len(ids)
    selected_set: set[str] = set()
    selected: list[str] = []

    for s, stratum_count in zip(unique_strata, counts):
        stratum_indices = np.where(strata == s)[0]
        target = int(n * stratum_count / total)
        target = min(target, stratum_count)
        if target <= 0:
            continue
        candidates = [ids[i] for i in stratum_indices]
        sampled = rng.choice(candidates, size=target, replace=False).tolist()
        selected.extend(sampled)
        selected_set.update(sampled)

    # 补足到 n
    if len(selected) < n:
        remaining = [i for i in ids if i not in selected_set]
        k = min(n - len(selected), len(remaining))
        extra = rng.choice(remaining, size=k, replace=False).tolist()
        selected.extend(extra)

    return selected
