from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from experiments.china_v1_fusion_smoke.archive import PatchSelection, select_complete_patches
from experiments.china_v1_fusion_smoke.data import load_patch_year_batch

PATCH_IDS = [
    "parent_32643:310:3383",
    "parent_32643:311:3390",
    "parent_32643:312:3346",
    "parent_32643:313:3352",
]


def _tiff_bytes(sensor: str, month: int) -> bytes:
    band_count = 10 if sensor == "pc-s2" else 2
    raw = np.empty((band_count, 128, 128), dtype=np.float32)
    for band in range(band_count):
        raw[band].fill(month * 10000 + band * 1000)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=128,
            height=128,
            count=band_count,
            dtype=raw.dtype,
            crs="EPSG:32643",
            transform=from_origin(310000.0, 3384000.0, 10.0, 10.0),
        ) as dataset:
            dataset.write(raw)
        return memory_file.read()


def _write_complete_archive(path: Path, sensor: str, year: int, month: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        contents = _tiff_bytes(sensor, month)
        for patch_id in reversed(PATCH_IDS):
            archive.writestr(f"{sensor}/{year}/{month:02d}/{patch_id}.tif", contents)


@pytest.fixture
def fake_complete_selection(tmp_path: Path) -> tuple[PatchSelection, ...]:
    for sensor in ("pc-s2", "pc-s1"):
        for year in (2020, 2021):
            for month in range(1, 13):
                _write_complete_archive(
                    tmp_path
                    / sensor
                    / str(year)
                    / f"{month:02d}"
                    / f"{sensor}_{year}_{month:02d}.zip",
                    sensor,
                    year,
                    month,
                )
    return select_complete_patches(tmp_path, count=4)


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
