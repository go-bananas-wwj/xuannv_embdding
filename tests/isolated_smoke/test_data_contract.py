from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from experiments.china_v1_fusion_smoke.archive import (
    PatchSelection,
    select_complete_patches,
)
from experiments.china_v1_fusion_smoke.data import load_patch_year_batch

PATCH_IDS = [
    "parent_32643:310:3383",
    "parent_32643:311:3390",
    "parent_32643:312:3346",
    "parent_32643:313:3352",
]


def _tiff_bytes(
    sensor: str,
    month: int,
    *,
    crs: str = "EPSG:32643",
    transform: Affine | None = None,
    invalid_value: float | None = None,
) -> bytes:
    band_count = 10 if sensor == "pc-s2" else 2
    raw = np.empty((band_count, 128, 128), dtype=np.float32)
    for band in range(band_count):
        raw[band].fill(month * 10000 + band * 1000)
    if invalid_value is not None:
        raw[:, 0, 0] = invalid_value
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=128,
            height=128,
            count=band_count,
            dtype=raw.dtype,
            crs=crs,
            transform=transform or from_origin(310000.0, 3384000.0, 10.0, 10.0),
        ) as dataset:
            dataset.write(raw)
        return memory_file.read()


def _write_complete_archive(path: Path, sensor: str, year: int, month: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        contents = _tiff_bytes(sensor, month)
        for patch_id in reversed(PATCH_IDS):
            archive.writestr(f"{sensor}/{year}/{month:02d}/{patch_id}.tif", contents)


@pytest.fixture(scope="module")
def fake_complete_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("complete-archives")
    for sensor in ("pc-s2", "pc-s1"):
        for year in (2020, 2021):
            for month in range(1, 13):
                _write_complete_archive(
                    root / sensor / str(year) / f"{month:02d}" / f"{sensor}_{year}_{month:02d}.zip",
                    sensor,
                    year,
                    month,
                )
    return root


@pytest.fixture(scope="module")
def fake_complete_selection(fake_complete_root: Path) -> tuple[PatchSelection, ...]:
    return select_complete_patches(fake_complete_root, count=4)


def _replacement_selection(
    selection: PatchSelection,
    tmp_path: Path,
    *,
    reference_index: int,
    crs: str = "EPSG:32643",
    transform: Affine | None = None,
    invalid_value: float | None = None,
) -> PatchSelection:
    original = selection.months[reference_index]
    archive_path = tmp_path / f"replacement-{reference_index}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    contents = _tiff_bytes(
        original.sensor,
        original.month,
        crs=crs,
        transform=transform,
        invalid_value=invalid_value,
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(original.member, contents)
        info = archive.getinfo(original.member)
    replacement = replace(original, archive=archive_path, crc=info.CRC)
    months = list(selection.months)
    months[reference_index] = replacement
    return replace(selection, months=tuple(months))


def test_patch_year_batch_has_quarter_major_shapes(
    fake_complete_selection: tuple[PatchSelection, ...],
) -> None:
    """四 patch、两年、四季度、每季三个月的模型张量布局不可被月序布局替代。"""
    batch = load_patch_year_batch(fake_complete_selection, years=(2020, 2021))

    assert batch.s2.shape == (8, 4, 3, 10, 128, 128)
    assert batch.s1.shape == (8, 4, 3, 2, 128, 128)
    assert batch.valid_s2.shape == (8, 4, 3, 1, 128, 128)
    assert batch.valid_s1.shape == (8, 4, 3, 1, 128, 128)


def test_loader_scales_s2_reorders_s1_and_places_months_by_quarter(
    fake_complete_selection: tuple[PatchSelection, ...],
) -> None:
    """原始 S1 的 VH/VV 必须变为模型 VV/VH，且三月必须写入第一季度的第三个槽。"""
    batch = load_patch_year_batch(fake_complete_selection, years=(2020, 2021))

    assert batch.patch_ids == tuple(
        patch.patch_id for patch in fake_complete_selection for _ in (2020, 2021)
    )
    assert batch.years == (2020, 2021, 2020, 2021, 2020, 2021, 2020, 2021)
    np.testing.assert_allclose(batch.s2[0, 0, 2, 0, 0, 0].item(), 3.0)
    np.testing.assert_allclose(batch.s2[0, 1, 0, 0, 0, 0].item(), 4.0)
    np.testing.assert_allclose(batch.s1[0, 0, 2, 0, 0, 0].item(), 10.0 * np.log10(31000.0))
    np.testing.assert_allclose(batch.s1[0, 0, 2, 1, 0, 0].item(), 10.0 * np.log10(30000.0))
    assert bool(batch.valid_s2.all())
    assert bool(batch.valid_s1.all())


def test_loader_rejects_duplicate_missing_and_extra_month_references(
    fake_complete_selection: tuple[PatchSelection, ...],
) -> None:
    """完整 selection 必须恰好含有 48 个唯一的 sensor/year/month 键。"""
    selection = fake_complete_selection[0]
    duplicate = replace(selection, months=selection.months + (selection.months[0],))
    missing = replace(selection, months=selection.months[1:])
    extra = replace(selection, months=selection.months + (replace(selection.months[0], month=13),))

    for malformed in (duplicate, missing, extra):
        with pytest.raises(ValueError, match=r"exactly 48 unique month references"):
            load_patch_year_batch((malformed,))


@pytest.mark.parametrize(
    ("crs", "transform"),
    [
        ("EPSG:32643", from_origin(310010.0, 3384000.0, 10.0, 10.0)),
        ("EPSG:32644", from_origin(310000.0, 3384000.0, 10.0, 10.0)),
        ("EPSG:32643", Affine(10.0, 1.0, 310000.0, 0.0, -10.0, 3384000.0)),
        ("EPSG:32643", from_origin(310000.0, 3384000.0, 20.0, 20.0)),
        ("EPSG:32643", Affine(10.0, 0.0, 310000.0, 0.0, -10.0, 3384010.0)),
    ],
    ids=(
        "shifted-transform",
        "different-crs",
        "rotated-transform",
        "non-10m-transform",
        "northing-shift-10m",
    ),
)
def test_loader_rejects_nonuniform_or_invalid_patch_grid(
    fake_complete_selection: tuple[PatchSelection, ...],
    tmp_path: Path,
    crs: str,
    transform: Affine,
) -> None:
    """S1/S2 所有月份必须共享同一 CRS 和精确的 north-up 10 m 仿射网格。"""
    malformed = _replacement_selection(
        fake_complete_selection[0],
        tmp_path,
        reference_index=0,
        crs=crs,
        transform=transform,
    )

    with pytest.raises(ValueError, match=r"patch grid"):
        load_patch_year_batch((malformed,))


def test_loader_masks_and_sanitizes_nonfinite_pixels(
    fake_complete_selection: tuple[PatchSelection, ...], tmp_path: Path
) -> None:
    """GDAL 有效像素中的 NaN/Inf 也必须置零并标为无效。"""
    with_nan_s1 = _replacement_selection(
        fake_complete_selection[0], tmp_path / "s1", reference_index=0, invalid_value=float("nan")
    )
    with_nan_and_inf = _replacement_selection(
        with_nan_s1, tmp_path / "s2", reference_index=24, invalid_value=float("inf")
    )

    batch = load_patch_year_batch((with_nan_and_inf,))

    assert not bool(batch.valid_s1[0, 0, 0, 0, 0, 0])
    assert not bool(batch.valid_s2[0, 0, 0, 0, 0, 0])
    assert batch.s1[0, 0, 0, :, 0, 0].eq(0).all()
    assert batch.s2[0, 0, 0, :, 0, 0].eq(0).all()
    assert bool(torch.isfinite(batch.s1).all())
    assert bool(torch.isfinite(batch.s2).all())
