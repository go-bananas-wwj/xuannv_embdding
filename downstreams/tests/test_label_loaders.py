# downstreams/tests/test_label_loaders.py
import json
from pathlib import Path

import numpy as np
from downstreams.data.label_loaders import (
    get_reference_patch_path,
    parse_patch_id_from_labelme_name,
    rasterize_labelme,
)


def test_parse_patch_id() -> None:
    path = Path("patch_000002_20260430_rgb_uint8.json")
    assert parse_patch_id_from_labelme_name(path) == "patch_000002"


def test_parse_patch_id_invalid() -> None:
    invalid_names = [
        Path("notpatch_000002.json"),
        Path("patch.json"),
        Path("foo.txt"),
        Path("patch_.json"),
    ]
    for name in invalid_names:
        try:
            parse_patch_id_from_labelme_name(name)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {name}")


def test_get_reference_patch_path_found_and_not_found(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (patch_dir / "region_haidian_patch_000001.tif").write_text("fake")
    (patch_dir / "region_haidian_patch_000002.tif").write_text("fake")

    found1 = get_reference_patch_path(patch_dir, "patch_000001")
    assert found1 == patch_dir / "region_haidian_patch_000001.tif"
    found2 = get_reference_patch_path(patch_dir, "patch_000002")
    assert found2 == patch_dir / "region_haidian_patch_000002.tif"
    assert get_reference_patch_path(patch_dir, "patch_000003") is None


def _write_labelme(path: Path, shapes: list[dict]) -> None:
    data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": "dummy.jpg",
        "imageData": None,
        "imageHeight": 128,
        "imageWidth": 128,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_rasterize_labelme_basic(tmp_path: Path) -> None:
    label_path = tmp_path / "patch_000001.json"
    _write_labelme(
        label_path,
        [
            {
                "label": "jiazhudongdi",
                "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
    )

    mask = rasterize_labelme(label_path, (128, 128), {"jiazhudongdi": 1})
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    assert np.unique(mask).tolist() == [0, 1]


def test_rasterize_labelme_empty_shapes(tmp_path: Path) -> None:
    label_path = tmp_path / "patch_000001.json"
    _write_labelme(label_path, [])

    mask = rasterize_labelme(label_path, (64, 64), {"jiazhudongdi": 1})
    assert mask.shape == (64, 64)
    assert mask.sum() == 0
    assert np.unique(mask).tolist() == [0]


def test_rasterize_labelme_invalid_geometry(tmp_path: Path) -> None:
    """自相交多边形应通过 buffer(0) 修复，且不抛出异常。"""
    label_path = tmp_path / "patch_000001.json"
    # 蝴蝶结形状（自相交）
    _write_labelme(
        label_path,
        [
            {
                "label": "jiazhudongdi",
                "points": [[10, 10], [90, 90], [90, 10], [10, 90]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
    )

    mask = rasterize_labelme(label_path, (128, 128), {"jiazhudongdi": 1})
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    # buffer(0) 修复后仍应产生合理的前景像素
    assert mask.sum() > 0
