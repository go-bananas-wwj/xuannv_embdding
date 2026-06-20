# downstreams/tests/test_split.py
from pathlib import Path

import numpy as np
import pytest
import rasterio
from downstreams.data.split import _stratified_sample, create_stratified_folds


def _make_mask_dir(tmp_path: Path, n: int, ratio_fn) -> Path:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for i in range(n):
        mask = np.zeros((16, 16), dtype=np.uint8)
        r = ratio_fn(i)
        if r > 0:
            n_pixels = int(r * mask.size)
            mask.flat[:n_pixels] = 1
        with rasterio.open(
            mask_dir / f"patch_{i:06d}.tif",
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=mask.dtype,
            crs=None,
            transform=rasterio.Affine.identity(),
        ) as dst:
            dst.write(mask, 1)
    return mask_dir


def test_create_folds(tmp_path: Path) -> None:
    mask_dir = _make_mask_dir(tmp_path, 10, lambda i: 0.0625 if i % 2 == 0 else 0.0)
    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    assert len(split["folds"]) == 5
    for fold in split["folds"]:
        assert len(fold["train"]) > 0
        assert len(fold["test"]) > 0
        assert set(fold["train"]) & set(fold["test"]) == set()


def test_train_val_test_disjoint(tmp_path: Path) -> None:
    mask_dir = _make_mask_dir(tmp_path, 20, lambda i: 0.0625 if i % 2 == 0 else 0.0)
    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    all_ids = {f"patch_{i:06d}" for i in range(20)}
    for fold in split["folds"]:
        train = set(fold["train"])
        val = set(fold["val"])
        test = set(fold["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert train | val | test == all_ids


def test_fractions_are_subsets(tmp_path: Path) -> None:
    mask_dir = _make_mask_dir(tmp_path, 30, lambda i: 0.0625 if i % 2 == 0 else 0.0)
    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    for fold in split["folds"]:
        train_set = set(fold["train"])
        for frac_str, frac_dict in split["fractions"].items():
            selected = frac_dict[f"fold_{fold['fold']}"]
            assert set(selected) <= train_set
            frac = float(frac_str)
            expected = min(max(1, int(len(fold["train"]) * frac)), len(fold["train"]))
            assert len(selected) == expected


def test_reproducible_fractions(tmp_path: Path) -> None:
    mask_dir = _make_mask_dir(tmp_path, 30, lambda i: 0.0625 if i % 2 == 0 else 0.0)
    split1 = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    split2 = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    for frac_str in split1["fractions"]:
        for fold_key in split1["fractions"][frac_str]:
            sel1 = split1["fractions"][frac_str][fold_key]
            sel2 = split2["fractions"][frac_str][fold_key]
            assert sel1 == sel2


def test_stratified_sample_duplicate_ids() -> None:
    ids = ["a", "a", "b", "b"]
    strata = np.array([0, 0, 1, 1])
    with pytest.raises(AssertionError):
        _stratified_sample(ids, strata, n=2, seed=0)


def test_stratified_sample_n_smaller_than_strata() -> None:
    ids = [f"id_{i}" for i in range(10)]
    strata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    selected = _stratified_sample(ids, strata, n=3, seed=42)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert set(selected) <= set(ids)


def test_kfold_fallback(tmp_path: Path) -> None:
    # 6 个样本、2 个 strata（count 2/4）、5-fold：StratifiedKFold 条件不满足，退化为 KFold
    mask_dir = _make_mask_dir(
        tmp_path,
        6,
        lambda i: 0.5 if i < 2 else 0.0,
    )
    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    assert len(split["folds"]) == 5
    all_ids = {f"patch_{i:06d}" for i in range(6)}
    for fold in split["folds"]:
        assert set(fold["train"]) | set(fold["val"]) | set(fold["test"]) == all_ids
        assert set(fold["train"]) & set(fold["test"]) == set()
        assert set(fold["train"]) & set(fold["val"]) == set()
        assert set(fold["val"]) & set(fold["test"]) == set()
