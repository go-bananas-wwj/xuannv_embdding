from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from sklearn.model_selection import KFold, StratifiedKFold


import re


_MASK_MONTH_SUFFIX = re.compile(r"_20\d{4}$")


def _base_patch_id(stem: str) -> str:
    """去掉 mask 文件名末尾的 ``_YYYYMM`` 月份后缀，得到与 embedding 目录一致的 patch_id。"""
    return _MASK_MONTH_SUFFIX.sub("", stem)


def _positive_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum() / total)


def _collect_mask_ids(mask_dir: Path) -> tuple[list[str], list[float]]:
    """收集 mask 目录下的基础 patch_id 与对应的最大正像素比例。

    同一 patch_id 的多个月份 mask 取最大正像素比例，避免重复样本。
    """
    mask_paths = sorted(mask_dir.glob("*.tif"))
    if not mask_paths:
        raise ValueError(f"mask_dir 中不存在 *.tif 文件: {mask_dir}")

    ratio_by_id: dict[str, float] = {}
    for p in mask_paths:
        base_id = _base_patch_id(p.stem)
        ratio = _positive_ratio(p)
        ratio_by_id[base_id] = max(ratio_by_id.get(base_id, 0.0), ratio)

    patch_ids = sorted(ratio_by_id.keys())
    ratios = np.array([ratio_by_id[pid] for pid in patch_ids])
    return patch_ids, ratios


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


def _build_folds_from_ids(
    patch_ids: list[str],
    ratios: np.ndarray,
    n_folds: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    """根据已排序的 patch_ids 与正像素比例生成 stratified folds 与 fractions。"""
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
    fractions: dict[str, dict[str, list[str]]] = {"0.1": {}, "0.25": {}, "0.5": {}, "1.0": {}}
    for fold in folds:
        train_ids = fold["train"]
        train_ratios = np.array([ratios[pid_to_idx[pid]] for pid in train_ids])
        train_strata = _quantize(train_ratios, n_bins=2)

        for frac_str, frac in [("0.1", 0.1), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0)]:
            n = max(1, int(len(train_ids) * frac))
            frac_seed = seed + fold["fold"] * 7 + frac_offsets[frac_str]
            selected = _stratified_sample(train_ids, train_strata, n, frac_seed)
            fractions[frac_str][f"fold_{fold['fold']}"] = selected

    return folds, fractions


def create_stratified_folds(
    mask_dir: Path,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """按 mask 正像素比例分层，生成 n_folds-fold。

    每 fold：4 folds 训练，1 fold 测试；训练集内部再切 val_ratio 做验证。
    patch_id 会去掉 ``_YYYYMM`` 月份后缀，以便与 embedding 目录对齐。
    """
    patch_ids, ratios = _collect_mask_ids(mask_dir)

    folds, fractions = _build_folds_from_ids(patch_ids, ratios, n_folds, val_ratio, seed)

    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_ratio": val_ratio,
        "stratify_by": "positive_pixel_ratio",
        "folds": folds,
        "fractions": fractions,
    }


def create_combined_stratified_folds(
    mask_dirs: dict[str, Path],
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """跨多个 region 的 mask 目录生成统一 stratified folds。

    Args:
        mask_dirs: {region_name: Path_to_masks_dir}，每个目录下包含 *.tif 掩码。

    Returns:
        与 create_stratified_folds 相同结构，但 patch_id 前缀为 ``{region}_{base_patch_id}``；
        额外包含 ``region_of`` 字段，记录每个前缀 patch_id 对应的 region。
    """
    all_patch_ids: list[str] = []
    all_ratios: list[float] = []
    region_of: dict[str, str] = {}

    for region, mask_dir in mask_dirs.items():
        patch_ids, ratios = _collect_mask_ids(mask_dir)
        for pid, ratio in zip(patch_ids, ratios):
            prefixed_id = f"{region}_{pid}"
            all_patch_ids.append(prefixed_id)
            all_ratios.append(ratio)
            region_of[prefixed_id] = region

    patch_ids = all_patch_ids
    ratios = np.array(all_ratios)
    folds, fractions = _build_folds_from_ids(patch_ids, ratios, n_folds, val_ratio, seed)

    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_ratio": val_ratio,
        "stratify_by": "positive_pixel_ratio",
        "region_of": region_of,
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
