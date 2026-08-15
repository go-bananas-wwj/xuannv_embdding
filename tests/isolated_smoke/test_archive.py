from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from experiments.china_v1_fusion_smoke.archive import (
    inspect_selection_headers,
    select_complete_patches,
)

COMPLETE_PATCH_IDS = [
    "parent_32643:310:3383",
    "parent_32643:311:3390",
    "parent_32643:312:3346",
    "parent_32643:313:3352",
    "parent_32643:314:3358",
]
INCOMPLETE_PATCH_ID = "parent_32643:309:3377"


def _write_archive(path: Path, sensor: str, year: int, month: int, patch_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for patch_id in patch_ids:
            member = f"{sensor}/{year}/{month:02d}/{patch_id}.tif"
            archive.writestr(member, f"{sensor}-{year}-{month}-{patch_id}".encode())


@pytest.fixture
def fake_archive_root(tmp_path: Path) -> Path:
    """构造与真实目录及 member 命名一致、但不含真实栅格负载的 ZIP 集。"""
    for sensor in ("pc-s2", "pc-s1"):
        for year in (2020, 2021):
            for month in range(1, 13):
                patch_ids = list(reversed(COMPLETE_PATCH_IDS))
                if sensor == "pc-s1" and year == 2021 and month == 12:
                    patch_ids.remove(COMPLETE_PATCH_IDS[-1])
                if sensor == "pc-s2" and year == 2020 and month == 1:
                    patch_ids.append(INCOMPLETE_PATCH_ID)
                _write_archive(
                    tmp_path
                    / sensor
                    / str(year)
                    / f"{month:02d}"
                    / f"{sensor}_{year}_{month:02d}.zip",
                    sensor,
                    year,
                    month,
                    patch_ids,
                )
    return tmp_path


def test_selector_returns_first_four_complete_patch_ids(fake_archive_root: Path) -> None:
    """遗漏任一月份或传感器的 patch 不得进入按字典序截取的完整交集。"""
    selected = select_complete_patches(fake_archive_root, count=4)

    assert [item.patch_id for item in selected] == COMPLETE_PATCH_IDS[:4]
    assert all(len(item.months) == 48 for item in selected)


def test_selector_captures_member_crc_and_preserves_archive_paths(fake_archive_root: Path) -> None:
    """选择结果必须保留 ZIP 内 member 及其目录元数据 CRC，供后续可复核读取。"""
    selection = select_complete_patches(fake_archive_root, count=1)[0]
    first = selection.months[0]

    with ZipFile(first.archive) as archive:
        info = archive.getinfo(first.member)

    assert first.sensor == "pc-s1"
    assert (first.year, first.month) == (2020, 1)
    assert first.member == f"pc-s1/2020/01/{selection.patch_id}.tif"
    assert first.crc == info.CRC


def test_selector_ignores_nested_tiff_members(fake_archive_root: Path) -> None:
    """仅直接位于约定目录下的 TIFF 才能成为可重建的 patch member。"""
    nested_patch_id = "parent_32643:001:0001"
    for archive_path in fake_archive_root.rglob("*.zip"):
        sensor, year, month = archive_path.parts[-4:-1]
        with ZipFile(archive_path, "a", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                f"{sensor}/{year}/{month}/nested/{nested_patch_id}.tif",
                b"not-a-direct-member",
            )

    selected = select_complete_patches(fake_archive_root, count=4)

    assert [item.patch_id for item in selected] == COMPLETE_PATCH_IDS[:4]


def test_selector_rejects_requested_count_larger_than_complete_intersection(
    fake_archive_root: Path,
) -> None:
    """调用方请求四个以上完整 patch 时不能静默降级为较小 batch。"""
    with pytest.raises(ValueError, match=r"requested 5 complete patches, found 4"):
        select_complete_patches(fake_archive_root, count=5)


def test_selector_does_not_modify_archive_tree(fake_archive_root: Path) -> None:
    """检索只能读取 ZIP 目录，调用前后输入文件集合与时间戳保持不变。"""
    before = {
        path.relative_to(fake_archive_root): path.stat().st_mtime_ns
        for path in fake_archive_root.rglob("*")
    }

    select_complete_patches(fake_archive_root, count=4)

    after = {
        path.relative_to(fake_archive_root): path.stat().st_mtime_ns
        for path in fake_archive_root.rglob("*")
    }
    assert after == before


def test_header_inspection_rejects_non_raster_members(fake_archive_root: Path) -> None:
    """header 检查必须尝试真实 GeoTIFF 解码，而非把 ZIP member 当作已验证栅格。"""
    selection = select_complete_patches(fake_archive_root, count=1)[0]

    with pytest.raises(Exception):
        inspect_selection_headers(selection)
