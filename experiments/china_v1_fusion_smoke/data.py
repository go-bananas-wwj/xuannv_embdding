from __future__ import annotations

from dataclasses import dataclass
from zipfile import ZipFile

import numpy as np
import torch
from rasterio.io import MemoryFile

from experiments.china_v1_fusion_smoke.archive import MonthRef, PatchSelection

EXPECTED_HEIGHT = 128
EXPECTED_WIDTH = 128
S2_BANDS = 10
S1_BANDS = 2


@dataclass(frozen=True)
class PatchYearBatch:
    """quarter-major 的批量输入，样本顺序为 patch 外层、year 内层。"""

    s2: torch.Tensor
    s1: torch.Tensor
    valid_s2: torch.Tensor
    valid_s1: torch.Tensor
    patch_ids: tuple[str, ...]
    years: tuple[int, ...]


def _decode_raster(
    month_ref: MonthRef, expected_bands: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, tuple[float, ...]]]:
    """从 ZIP member 直接解码单幅 GeoTIFF，并在进入训练前验证其空间契约。"""
    with ZipFile(month_ref.archive) as archive:
        info = archive.getinfo(month_ref.member)
        if info.CRC != month_ref.crc:
            raise ValueError(f"CRC changed for {month_ref.archive}!{month_ref.member}")
        with archive.open(info) as member_file, MemoryFile(member_file.read()) as memory_file:
            with memory_file.open() as dataset:
                if (dataset.height, dataset.width) != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
                    raise ValueError(
                        f"unexpected raster shape for {month_ref.member}: "
                        f"{dataset.height}x{dataset.width}"
                    )
                if dataset.count != expected_bands:
                    raise ValueError(
                        f"unexpected band count for {month_ref.member}: "
                        f"{dataset.count}, expected {expected_bands}"
                    )
                if dataset.crs is None:
                    raise ValueError(f"raster has no CRS: {month_ref.member}")
                transform = dataset.transform
                if not (
                    np.isclose(transform.a, 10.0)
                    and np.isclose(transform.b, 0.0)
                    and np.isclose(transform.d, 0.0)
                    and np.isclose(transform.e, -10.0)
                ):
                    raise ValueError(f"invalid patch grid transform: {month_ref.member}")
                raw = dataset.read()
                finite = np.isfinite(raw)
                valid = np.all(dataset.read_masks() > 0, axis=0, keepdims=True) & np.all(
                    finite, axis=0, keepdims=True
                )
                raw = np.where(finite, raw, 0.0)
                grid = (dataset.crs.to_string(), tuple(transform))
    return raw, valid, grid


def _refs_by_sensor(selection: PatchSelection, year: int) -> dict[tuple[str, int], MonthRef]:
    refs = {
        (month_ref.sensor, month_ref.month): month_ref
        for month_ref in selection.months
        if month_ref.year == year
    }
    expected = {(sensor, month) for sensor in ("pc-s2", "pc-s1") for month in range(1, 13)}
    if set(refs) != expected:
        raise ValueError(f"selection {selection.patch_id} is incomplete for year {year}")
    return refs


def _validate_selection_months(selection: PatchSelection) -> None:
    expected = {
        (sensor, year, month)
        for sensor in ("pc-s2", "pc-s1")
        for year in (2020, 2021)
        for month in range(1, 13)
    }
    keys = tuple((ref.sensor, ref.year, ref.month) for ref in selection.months)
    if len(keys) != 48 or len(set(keys)) != 48 or set(keys) != expected:
        raise ValueError(
            f"selection {selection.patch_id} must contain exactly 48 unique month references"
        )


def _validate_grid(
    expected_grid: tuple[str, tuple[float, ...]] | None,
    grid: tuple[str, tuple[float, ...]],
    month_ref: MonthRef,
) -> tuple[str, tuple[float, ...]]:
    if expected_grid is None:
        return grid
    if grid[0] != expected_grid[0] or not np.allclose(grid[1], expected_grid[1]):
        raise ValueError(f"patch grid differs for {month_ref.member}")
    return expected_grid


def load_patch_year_batch(
    selections: tuple[PatchSelection, ...], years: tuple[int, int] = (2020, 2021)
) -> PatchYearBatch:
    """读取完整 patch selections，输出 ``[B, quarter, month, channel, H, W]`` 张量。"""
    if not selections:
        raise ValueError("at least one patch selection is required")
    if years != (2020, 2021):
        raise ValueError("smoke loader requires years 2020 and 2021")

    s2_samples: list[np.ndarray] = []
    s1_samples: list[np.ndarray] = []
    valid_s2_samples: list[np.ndarray] = []
    valid_s1_samples: list[np.ndarray] = []
    patch_ids: list[str] = []
    sample_years: list[int] = []
    for selection in selections:
        _validate_selection_months(selection)
        patch_grid: tuple[str, tuple[float, ...]] | None = None
        for year in years:
            refs = _refs_by_sensor(selection, year)
            s2 = np.empty((4, 3, S2_BANDS, EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=np.float32)
            s1 = np.empty((4, 3, S1_BANDS, EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=np.float32)
            valid_s2 = np.empty((4, 3, 1, EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=bool)
            valid_s1 = np.empty((4, 3, 1, EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=bool)
            for month in range(1, 13):
                quarter_index = (month - 1) // 3
                month_index = (month - 1) % 3
                s2_ref = refs["pc-s2", month]
                s1_ref = refs["pc-s1", month]
                raw_s2, mask_s2, s2_grid = _decode_raster(s2_ref, S2_BANDS)
                patch_grid = _validate_grid(patch_grid, s2_grid, s2_ref)
                raw_s1, mask_s1, s1_grid = _decode_raster(s1_ref, S1_BANDS)
                patch_grid = _validate_grid(patch_grid, s1_grid, s1_ref)
                s2[quarter_index, month_index] = np.where(
                    mask_s2, raw_s2.astype("float32") * 0.0001, 0.0
                )
                s1_vv_vh = raw_s1.astype("float32")[[1, 0]]
                s1[quarter_index, month_index] = np.where(
                    mask_s1, 10.0 * np.log10(np.maximum(s1_vv_vh, 1.0e-6)), 0.0
                )
                valid_s2[quarter_index, month_index] = mask_s2
                valid_s1[quarter_index, month_index] = mask_s1
            s2_samples.append(s2)
            s1_samples.append(s1)
            valid_s2_samples.append(valid_s2)
            valid_s1_samples.append(valid_s1)
            patch_ids.append(selection.patch_id)
            sample_years.append(year)

    return PatchYearBatch(
        s2=torch.from_numpy(np.stack(s2_samples)),
        s1=torch.from_numpy(np.stack(s1_samples)),
        valid_s2=torch.from_numpy(np.stack(valid_s2_samples)),
        valid_s1=torch.from_numpy(np.stack(valid_s1_samples)),
        patch_ids=tuple(patch_ids),
        years=tuple(sample_years),
    )
