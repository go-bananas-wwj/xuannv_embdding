#!/usr/bin/env python3
"""从上游展示/API仓库同步哈尔滨、海淀的AOI GeoJSON与patch网格元数据。

数据来源:
- 哈尔滨: https://raw.githubusercontent.com/go-bananas-wwj/xuannv_show/dev/data/harbin/patches_meta.json
- 海淀: http://60.31.21.42:22065/regions/haidian/patches?page={}&page_size=100

输出:
- configs/regions/{harbin,haidian}.geojson: 精确覆盖全部patch WGS84外包框的多边形。
- configs/regions/{harbin,haidian}_patches.json: 包含 crs / patch_size_m / patches 的网格元数据。

脚本可重复执行（幂等），每次都会重新拉取上游最新数据并覆盖输出文件。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGIONS_DIR = ROOT / "configs" / "regions"

PATCH_SIZE_M = 1280

HARBIN_URL = (
    "https://raw.githubusercontent.com/go-bananas-wwj/xuannv_show/dev/data/harbin/patches_meta.json"
)
HAIDIAN_API = "http://60.31.21.42:22065/regions/haidian/patches?page={page}&page_size=100"


def _fetch_json(url: str, timeout: int = 60) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _fetch_harbin_patches() -> list[dict[str, Any]]:
    data = _fetch_json(HARBIN_URL)
    if not isinstance(data, list):
        raise ValueError(f"Harbin metadata is not a JSON list, got {type(data)}")
    return data


def _fetch_haidian_patches() -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _fetch_json(HAIDIAN_API.format(page=page))
        batch = data.get("patches", [])
        patches.extend(batch)
        if not data.get("has_next") or len(batch) == 0:
            break
        page += 1
        # 安全上限，避免异常情况下无限循环
        if page > 10:
            raise RuntimeError("Haidian pagination exceeded safety limit")
    return patches


def _derive_grid_index(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为没有 ix/iy 的patch根据UTM坐标推导网格索引。"""
    x_min = min(p["bounds"][0] for p in patches)
    y_min = min(p["bounds"][1] for p in patches)
    for p in patches:
        ix = int(round((p["bounds"][0] - x_min) / PATCH_SIZE_M))
        iy = int(round((p["bounds"][1] - y_min) / PATCH_SIZE_M))
        p["ix"] = ix
        p["iy"] = iy
    return patches


def _normalize_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_id": patch["patch_id"],
        "ix": int(patch["ix"]),
        "iy": int(patch["iy"]),
        "bounds": [float(v) for v in patch["bounds"]],
        "bounds_wgs84": [float(v) for v in patch["bounds_wgs84"]],
    }


def _compute_wgs84_bbox(patches: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    left = min(p["bounds_wgs84"][0] for p in patches)
    bottom = min(p["bounds_wgs84"][1] for p in patches)
    right = max(p["bounds_wgs84"][2] for p in patches)
    top = max(p["bounds_wgs84"][3] for p in patches)
    return left, bottom, right, top


def _update_geojson(region: str, bbox: tuple[float, float, float, float]) -> None:
    path = REGIONS_DIR / f"{region}.geojson"
    geo = json.loads(path.read_text(encoding="utf-8"))

    left, bottom, right, top = bbox
    polygon_coords = [
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ]
    ]

    # 保持原有 FeatureCollection 结构，仅替换geometry坐标
    geo["features"][0]["geometry"]["coordinates"] = polygon_coords

    path.write_text(json.dumps(geo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_patches_json(region: str, crs: str, patches: list[dict[str, Any]]) -> None:
    path = REGIONS_DIR / f"{region}_patches.json"
    payload = {
        "crs": crs,
        "patch_size_m": PATCH_SIZE_M,
        "patches": [_normalize_patch(p) for p in patches],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_region(region: str, crs: str, patches: list[dict[str, Any]]) -> None:
    if region == "haidian":
        patches = _derive_grid_index(patches)

    bbox = _compute_wgs84_bbox(patches)
    _update_geojson(region, bbox)
    _write_patches_json(region, crs, patches)

    print(f"{region}: {len(patches)} patches, WGS84 bbox = {bbox}")


def main() -> None:
    REGIONS_DIR.mkdir(parents=True, exist_ok=True)

    harbin_patches = _fetch_harbin_patches()
    process_region("harbin", "EPSG:32652", harbin_patches)

    haidian_patches = _fetch_haidian_patches()
    process_region("haidian", "EPSG:32650", haidian_patches)


if __name__ == "__main__":
    main()
