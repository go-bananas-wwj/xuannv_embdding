"""Smoke/unit tests for scripts/process_harbin_patches.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from process_harbin_patches import (
    _default_class_map,
    _parse_class_map_arg,
    process_harbin,
)


def _quadrant_json(label: str, image_size: tuple[int, int] = (100, 100)) -> dict:
    w, h = image_size
    return {
        "version": "5.0.1",
        "flags": {},
        "imagePath": "dummy.jpg",
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "shapes": [
            {
                "label": label,
                "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
    }


def test_parse_class_map_arg():
    cm = _parse_class_map_arg('{"Construction Site":1,"JiaZhuDongDi":1}')
    assert cm == {"construction site": 1, "jiazhudongdi": 1}


def test_default_class_map_construction():
    cm = _default_class_map("construction")
    assert cm["construction site"] == 1
    assert cm["gongdi"] == 1


def test_default_class_map_unknown():
    with pytest.raises(ValueError):
        _default_class_map("not_a_task")


def test_process_harbin_creates_masks_and_raw(tmp_path: Path):
    raw_root = tmp_path / "raw"
    patch_dir = raw_root / "patch_000000"
    patch_dir.mkdir(parents=True)
    for month in ("20251201", "20260501"):
        (patch_dir / f"01_{month}.json").write_text(
            json.dumps(_quadrant_json("construction site")), encoding="utf-8"
        )

    output_root = tmp_path / "out" / "construction"
    class_map = {"construction site": 1}
    process_harbin(raw_root, output_root, target_mask_size=(128, 128), class_map=class_map)

    masks_dir = output_root / "masks"
    raw_out_base = output_root / "labelme_raw" / "patch_000000"

    assert masks_dir.exists()
    assert raw_out_base.exists()

    expected_masks = {"patch_000000_202512.tif", "patch_000000_202605.tif"}
    assert {p.name for p in masks_dir.glob("*.tif")} == expected_masks

    for mpath in masks_dir.glob("*.tif"):
        with __import__("rasterio").open(mpath) as src:
            arr = src.read(1)
        assert arr.shape == (128, 128)
        assert arr.dtype == np.uint8
        assert arr.max() == 1

    for month in ("202512", "202605"):
        assert len(list((raw_out_base / month).glob("*.json"))) == 1

    summary_path = output_root / "harbin_patches_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert all(s["positive_ratio"] > 0 for s in summary)
