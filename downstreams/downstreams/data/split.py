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
    """按 mask 正像素比例分层，生成 n_folds-fold。

    每 fold：4 folds 训练，1 fold 测试；训练集内部再切 val_ratio 做验证。
    """
    mask_paths = sorted(mask_dir.glob("*.tif"))
    if not mask_paths:
        raise ValueError(f"mask_dir 中不存在 *.tif 文件: {mask_dir}")
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


def create_spatial_block_folds(
    mask_dir: Path,
    patch_metadata_path: Path,
    n_folds: int = 5,
    seed: int = 42,
    buffer_patch_widths: float = 1.0,
) -> dict[str, Any]:
    """Create contiguous spatial folds with a neighbouring validation band.

    Patches are ordered along the dominant geographic axis and divided into
    contiguous bands.  For fold ``i``, band ``i`` is test data and band
    ``i + 1`` is validation data.  Training patches touching either held-out
    band are removed as a spatial buffer.
    """
    import json

    if n_folds < 3:
        raise ValueError("Spatial block evaluation requires at least three folds")
    if buffer_patch_widths < 0:
        raise ValueError("buffer_patch_widths must be non-negative")

    mask_paths = sorted(mask_dir.glob("*.tif"))
    if not mask_paths:
        raise ValueError(f"mask_dir 中不存在 *.tif 文件: {mask_dir}")
    patch_ids = [path.stem for path in mask_paths]
    ratios = np.array([_positive_ratio(path) for path in mask_paths])
    pid_to_idx = {patch_id: idx for idx, patch_id in enumerate(patch_ids)}

    metadata = json.loads(patch_metadata_path.read_text(encoding="utf-8"))
    bounds_by_id = {
        str(item["patch_id"]): np.asarray(item["bounds"], dtype=np.float64)
        for item in metadata
        if item.get("patch_id") in pid_to_idx and len(item.get("bounds", [])) == 4
    }
    missing = sorted(set(patch_ids) - set(bounds_by_id))
    if missing:
        raise ValueError(
            f"Patch metadata missing {len(missing)} mask patches; first entries: {missing[:5]}"
        )

    bounds = np.stack([bounds_by_id[patch_id] for patch_id in patch_ids])
    centers = np.column_stack(
        ((bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2)
    )
    centered = centers - centers.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    # Keep the ordering deterministic when the SVD axis sign is arbitrary.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    order = np.argsort(centered @ axis, kind="stable")
    bands = [[patch_ids[idx] for idx in band] for band in np.array_split(order, n_folds)]

    patch_width = float(np.median(bounds[:, 2] - bounds[:, 0]))
    patch_height = float(np.median(bounds[:, 3] - bounds[:, 1]))
    buffer_x = patch_width * buffer_patch_widths + 1e-6
    buffer_y = patch_height * buffer_patch_widths + 1e-6

    def touches_heldout(patch_id: str, heldout_ids: set[str]) -> bool:
        idx = pid_to_idx[patch_id]
        heldout_idx = np.fromiter((pid_to_idx[value] for value in heldout_ids), dtype=int)
        delta = np.abs(centers[heldout_idx] - centers[idx])
        return bool(np.any((delta[:, 0] <= buffer_x) & (delta[:, 1] <= buffer_y)))

    folds: list[dict[str, Any]] = []
    all_ids = set(patch_ids)
    for fold_idx in range(n_folds):
        test_ids = bands[fold_idx]
        val_ids = bands[(fold_idx + 1) % n_folds]
        heldout = set(test_ids) | set(val_ids)
        candidates = sorted(all_ids - heldout)
        buffer_ids = [patch_id for patch_id in candidates if touches_heldout(patch_id, heldout)]
        train_ids = sorted(set(candidates) - set(buffer_ids))
        if not train_ids:
            raise ValueError(f"Spatial fold {fold_idx} has no training patches after buffering")
        folds.append(
            {
                "fold": fold_idx,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "buffer": buffer_ids,
            }
        )

    frac_offsets = {"0.1": 11, "0.25": 22, "0.5": 33, "1.0": 44}
    fractions = {key: {} for key in frac_offsets}
    for fold in folds:
        train_ids = fold["train"]
        train_ratios = np.array([ratios[pid_to_idx[patch_id]] for patch_id in train_ids])
        train_strata = _quantize(train_ratios, n_bins=2)
        for frac_str, frac in (("0.1", 0.1), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0)):
            n = max(1, int(len(train_ids) * frac))
            frac_seed = seed + fold["fold"] * 7 + frac_offsets[frac_str]
            fractions[frac_str][f"fold_{fold['fold']}"] = _stratified_sample(
                train_ids, train_strata, n, frac_seed
            )

    return {
        "seed": seed,
        "n_folds": n_folds,
        "strategy": "contiguous_dominant_axis_bands",
        "validation_strategy": "next_spatial_band",
        "buffer_patch_widths": buffer_patch_widths,
        "patch_metadata_path": str(patch_metadata_path),
        "dominant_axis": axis.tolist(),
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
