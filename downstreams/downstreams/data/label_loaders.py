from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def parse_patch_id_from_labelme_name(path: Path) -> str:
    """从 labelme 文件名解析 patch id。

    例：patch_000002_20260430_rgb_uint8.json -> patch_000002
    """
    name = path.stem
    parts = name.split("_")
    if len(parts) < 2 or not parts[0].startswith("patch"):
        raise ValueError(f"无法解析 patch id: {path}")
    return f"{parts[0]}_{parts[1]}"


def load_labelme_shapes(label_path: Path) -> list[dict[str, Any]]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shapes", [])


def rasterize_labelme(
    label_path: Path,
    out_shape: tuple[int, int],
    class_map: dict[str, int] | None = None,
) -> np.ndarray:
    """把 labelme 多边形栅格化为 (H, W) uint8 mask。

    像素值：0=背景，class_map[label]=前景。
    多边形洞/重叠通过 shapely 处理为有效几何后栅格化。
    """
    if class_map is None:
        class_map = {"jiazhudongdi": 1}

    shapes: list[tuple[Any, int]] = []
    for s in load_labelme_shapes(label_path):
        label = s.get("label")
        if label not in class_map:
            continue
        cls = class_map[label]
        points = s.get("points", [])
        if len(points) < 3:
            continue
        geom = Polygon(points)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        shapes.append((geom, cls))

    mask = rasterize(
        shapes,
        out_shape=out_shape,
        fill=0,
        default_value=0,
        dtype=np.uint8,
        transform=Affine.identity(),
    )
    return mask


def get_reference_patch_path(patch_dir: Path, patch_id: str) -> Path | None:
    """在 patch_dir 中查找匹配 patch_id 的参考影像。"""
    candidates = sorted(patch_dir.glob(f"*_{patch_id}.tif"))
    if not candidates:
        return None
    return candidates[0]
