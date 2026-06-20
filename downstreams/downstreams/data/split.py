from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


def _positive_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum() / total)


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

    # 分层：按正像素比例分桶
    bins = np.percentile(ratios, [0, 25, 50, 75, 100])
    strata = np.digitize(ratios, bins[:-1]) - 1

    # 若样本太少或比例过于集中导致唯一 strata 不足 n_folds，或任一 strata 样本数不足 n_folds，
    # 则退化为非分层 KFold
    unique_strata, counts = np.unique(strata, return_counts=True)
    use_stratified = len(unique_strata) >= n_folds and counts.min() >= n_folds

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

        np.random.seed(seed + fold_idx)
        n_val = max(1, int(len(train_val_ids) * val_ratio))
        val_indices = np.random.choice(len(train_val_ids), size=n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(train_val_ids)), val_indices)

        train_ids = [train_val_ids[i] for i in train_indices]
        val_ids = [train_val_ids[i] for i in val_indices]

        folds.append({"fold": fold_idx, "train": train_ids, "val": val_ids, "test": test_ids})

    # 生成 10/25/50/100% 标签比例子集
    fractions = {"0.1": {}, "0.25": {}, "0.5": {}, "1.0": {}}
    for fold in folds:
        train_ids = fold["train"]
        train_ratios = np.array([ratios[patch_ids.index(pid)] for pid in train_ids])
        train_strata = np.digitize(train_ratios, np.percentile(train_ratios, [0, 50, 100])[:-1]) - 1
        for frac_str, frac in [("0.1", 0.1), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0)]:
            n = max(1, int(len(train_ids) * frac))
            selected = _stratified_sample(train_ids, train_strata, n, seed)
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
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    unique_strata = np.unique(strata)
    per_stratum = max(1, n // len(unique_strata))
    for s in unique_strata:
        candidates = [ids[i] for i in np.where(strata == s)[0]]
        k = min(per_stratum, len(candidates))
        selected.extend(rng.choice(candidates, size=k, replace=False).tolist())
    # 补足到 n
    if len(selected) < n:
        remaining = [i for i in ids if i not in selected]
        k = min(n - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=k, replace=False).tolist())
    return selected
