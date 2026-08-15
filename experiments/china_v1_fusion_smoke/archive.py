from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from rasterio.io import MemoryFile

SENSORS = ("pc-s1", "pc-s2")
YEARS = (2020, 2021)
MONTHS = tuple(range(1, 13))


@dataclass(frozen=True)
class MonthRef:
    """一个月度传感器栅格在源 ZIP 中的只读定位信息。"""

    sensor: str
    year: int
    month: int
    archive: Path
    member: str
    crc: int


@dataclass(frozen=True)
class PatchSelection:
    """一个同时覆盖两个传感器、两年完整月份的 patch。"""

    patch_id: str
    months: tuple[MonthRef, ...]


@dataclass(frozen=True)
class RasterHeader:
    """从 ZIP member 直接读取的栅格元数据，不持久化 TIFF。"""

    sensor: str
    year: int
    month: int
    width: int
    height: int
    count: int
    dtype: str
    crs: str
    transform: tuple[float, float, float, float, float, float, float, float, float]


def _archive_path(source_root: Path, sensor: str, year: int, month: int) -> Path:
    return source_root / sensor / str(year) / f"{month:02d}" / f"{sensor}_{year}_{month:02d}.zip"


def _patch_id(info_name: str, sensor: str, year: int, month: int) -> str | None:
    """若 ZIP member 遵循约定，返回它的 patch ID。"""
    expected_prefix = f"{sensor}/{year}/{month:02d}/"
    if not info_name.startswith(expected_prefix) or not info_name.endswith(".tif"):
        return None
    return Path(info_name).stem


def _member_ids(spec: tuple[Path, str, int, int]) -> set[str]:
    """读取一个 ZIP 的合规 TIFF ID；只检查中央目录。"""
    archive_path, sensor, year, month = spec
    members: set[str] = set()
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            patch_id = _patch_id(info.filename, sensor, year, month)
            if patch_id is None:
                continue
            if patch_id in members:
                raise ValueError(f"duplicate patch member in {archive_path}: {patch_id}")
            members.add(patch_id)
    return members


def _selected_crcs(
    spec: tuple[Path, str, int, int], selected_ids: frozenset[str]
) -> dict[str, int]:
    """从一个 ZIP 的中央目录捕获已选 patch 的 CRC，不读取 TIFF 内容。"""
    archive_path, sensor, year, month = spec
    crcs: dict[str, int] = {}
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            patch_id = _patch_id(info.filename, sensor, year, month)
            if patch_id not in selected_ids:
                continue
            if patch_id in crcs:
                raise ValueError(f"duplicate selected patch member in {archive_path}: {patch_id}")
            crcs[patch_id] = info.CRC
    return crcs


def _selected_crcs_for_spec(
    call: tuple[tuple[Path, str, int, int], frozenset[str]],
) -> dict[str, int]:
    """为 ProcessPoolExecutor 提供可序列化的固定选中集合调用。"""
    spec, selected_ids = call
    return _selected_crcs(spec, selected_ids)


def select_complete_patches(source_root: Path, count: int = 4) -> tuple[PatchSelection, ...]:
    """选择按 patch ID 稳定排序的完整 S1/S2 月度 ZIP 交集，绝不解压源档案。"""
    if count < 1:
        raise ValueError("count must be positive")

    root = Path(source_root)
    specs = tuple(
        (_archive_path(root, sensor, year, month), sensor, year, month)
        for sensor in SENSORS
        for year in YEARS
        for month in MONTHS
    )
    for archive, _, _, _ in specs:
        if not archive.is_file():
            raise FileNotFoundError(f"required source archive is missing: {archive}")

    complete_ids: set[str] | None = None
    for start in range(0, len(specs), 8):
        batch = specs[start : start + 8]
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            for members in executor.map(_member_ids, batch):
                complete_ids = members if complete_ids is None else complete_ids & members
    assert complete_ids is not None
    selected_ids = tuple(sorted(complete_ids)[:count])
    selected_set = frozenset(selected_ids)
    crc_by_spec: dict[tuple[Path, str, int, int], dict[str, int]] = {}
    for start in range(0, len(specs), 8):
        batch = specs[start : start + 8]
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            calls = ((spec, selected_set) for spec in batch)
            for spec, crcs in zip(batch, executor.map(_selected_crcs_for_spec, calls)):
                if set(crcs) != selected_set:
                    raise ValueError(f"selected patch disappeared from source archive: {spec[0]}")
                crc_by_spec[spec] = crcs

    selections: list[PatchSelection] = []
    for patch_id in selected_ids:
        refs: list[MonthRef] = []
        for sensor in SENSORS:
            for year in YEARS:
                for month in MONTHS:
                    archive = _archive_path(root, sensor, year, month)
                    member = f"{sensor}/{year}/{month:02d}/{patch_id}.tif"
                    crc = crc_by_spec[archive, sensor, year, month][patch_id]
                    refs.append(MonthRef(sensor, year, month, archive, member, crc))
        selections.append(PatchSelection(patch_id, tuple(refs)))
    return tuple(selections)


def _inspect_month_header(month_ref: MonthRef) -> RasterHeader:
    """经由 ``ZipFile.open`` 与内存 TIFF 读取一幅栅格头。"""
    with ZipFile(month_ref.archive) as archive:
        info = archive.getinfo(month_ref.member)
        if info.CRC != month_ref.crc:
            raise ValueError(f"CRC changed for {month_ref.archive}!{month_ref.member}")
        with archive.open(info) as member_file, MemoryFile(member_file.read()) as memory_file:
            with memory_file.open() as dataset:
                if dataset.crs is None:
                    raise ValueError(f"raster has no CRS: {month_ref.member}")
                return RasterHeader(
                    sensor=month_ref.sensor,
                    year=month_ref.year,
                    month=month_ref.month,
                    width=dataset.width,
                    height=dataset.height,
                    count=dataset.count,
                    dtype=dataset.dtypes[0],
                    crs=dataset.crs.to_string(),
                    transform=tuple(dataset.transform),
                )


def inspect_selection_headers(selection: PatchSelection) -> tuple[RasterHeader, ...]:
    """并发检查一个 selection 的全部 ZIP TIFF 头，返回顺序保持月度引用顺序。"""
    with ProcessPoolExecutor(max_workers=min(8, len(selection.months))) as executor:
        return tuple(executor.map(_inspect_month_header, selection.months))
