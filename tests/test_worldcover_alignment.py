"""QA: WorldCover 标签 patch 必须与对应 S2 patch 的网格严格对齐。

该测试覆盖 Haidian 与 Harbin 的样本 patch，防止 LULC 预处理中的
"same CRS shortcut" 导致标签相对光学影像产生亚像素偏移。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import rasterio

DATA_ROOT = Path("/data/xuannv_embedding/processed")
REGIONS = ["haidian", "harbin"]
TOL_M = 0.05  # 半分米级容差，远小于 10 m 分辨率


def _grid_key(name: str) -> str | None:
    # 支持参考项目格式 patch_000000 与旧格式 pXXX_rXXX
    m = re.search(r"(patch_\d{6})\.", name)
    if m:
        return m.group(1)
    m = re.search(r"(p\d{3}_r\d{3})\.", name)
    return m.group(1) if m else None


@pytest.mark.parametrize("region", REGIONS)
def test_worldcover_transform_matches_s2(region: str) -> None:
    s2_dir = DATA_ROOT / region / "patches" / "s2"
    wc_dir = DATA_ROOT / region / "labels" / "worldcover"
    if not s2_dir.exists() or not wc_dir.exists():
        pytest.skip(f"processed data missing for {region}")

    # 以 WorldCover 的 grid key 为锚，找同名 S2 patch
    wc_paths = sorted(wc_dir.glob("*.tif"))
    if not wc_paths:
        pytest.skip(f"no WorldCover patches for {region}")

    checked = 0
    for wc_path in wc_paths:
        grid = _grid_key(wc_path.name)
        if grid is None:
            continue
        s2_candidates = list(s2_dir.glob(f"s2_*_{grid}.tif"))
        if not s2_candidates:
            continue
        s2_path = s2_candidates[0]
        with rasterio.open(s2_path) as s2, rasterio.open(wc_path) as wc:
            s2_t = s2.transform
            wc_t = wc.transform
            # 仅比较原点；分辨率与旋转假定相同
            assert abs(s2_t.c - wc_t.c) < TOL_M, (
                f"{region} {grid}: WorldCover x origin {wc_t.c} != S2 x origin {s2_t.c}"
            )
            assert abs(s2_t.f - wc_t.f) < TOL_M, (
                f"{region} {grid}: WorldCover y origin {wc_t.f} != S2 y origin {s2_t.f}"
            )
        checked += 1
        if checked >= 5:
            break

    assert checked > 0, f"no matching S2/WorldCover pairs found for {region}"
