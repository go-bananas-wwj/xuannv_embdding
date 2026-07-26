from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/report/build_pujiang_slide01.py"
SPEC = importlib.util.spec_from_file_location("build_pujiang_slide01", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_api_patch_layout_uses_projected_bounds() -> None:
    patches = [
        {"patch_id": "patch_000001", "bounds": [1280.0, 0.0, 2560.0, 1280.0]},
        {"patch_id": "patch_000000", "bounds": [0.0, 0.0, 1280.0, 1280.0]},
        {"patch_id": "patch_000002", "bounds": [0.0, 1280.0, 1280.0, 2560.0]},
    ]

    layouts, rows, cols = MODULE.api_patch_layout(patches)

    positions = {item.patch_id: (item.row, item.col) for item in layouts}
    assert positions == {
        "patch_000002": (0, 0),
        "patch_000000": (1, 0),
        "patch_000001": (1, 1),
    }
    assert (rows, cols) == (2, 2)


def test_binary_prediction_mosaic_applies_registered_threshold() -> None:
    layouts = [
        MODULE.PatchLayout("patch_000000", 0, 0),
        MODULE.PatchLayout("patch_000001", 0, 1),
    ]
    predictions = {
        "patch_000000": np.array([[0.1, 0.99], [0.98, 0.2]], dtype=np.float32),
        "patch_000001": np.array([[0.7, 0.8], [0.1, 0.2]], dtype=np.float32),
    }

    mosaic, stats = MODULE.binary_prediction_mosaic(
        layouts,
        rows=1,
        cols=2,
        predictions=predictions,
        threshold=0.978237,
    )

    red = np.all(mosaic == np.array([230, 0, 0], dtype=np.uint8), axis=-1)
    assert red.tolist() == [[False, True, False, False], [True, False, False, False]]
    assert stats["positive_pixel_ratio"] == 0.25
    assert stats["patches"] == 2


def test_read_url_retries_transient_failures() -> None:
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"ok"

    def opener(_url: str, timeout: int):
        nonlocal attempts
        assert timeout == 120
        attempts += 1
        if attempts < 3:
            raise OSError("transient")
        return Response()

    assert MODULE.read_url_with_retry("http://example.test", opener=opener, sleep=lambda _: None) == b"ok"
    assert attempts == 3


def test_read_url_stops_after_retry_budget() -> None:
    def opener(_url: str, timeout: int):
        raise OSError("still unavailable")

    with pytest.raises(OSError, match="still unavailable"):
        MODULE.read_url_with_retry(
            "http://example.test",
            attempts=2,
            opener=opener,
            sleep=lambda _: None,
        )


def test_rgb_result_mosaic_preserves_api_tiles() -> None:
    layouts = [
        MODULE.PatchLayout("patch_000000", 0, 0),
        MODULE.PatchLayout("patch_000001", 0, 1),
    ]
    images = {
        "patch_000000": np.full((2, 2, 3), (230, 0, 0), dtype=np.uint8),
        "patch_000001": np.full((2, 2, 3), (0, 100, 200), dtype=np.uint8),
    }

    mosaic = MODULE.rgb_result_mosaic(layouts, 1, 2, images)

    assert mosaic.shape == (2, 4, 3)
    assert np.all(mosaic[:, :2] == (230, 0, 0))
    assert np.all(mosaic[:, 2:] == (0, 100, 200))


def test_binary_api_result_can_normalize_red_or_nonblack_foreground() -> None:
    red_white = np.array(
        [[[255, 255, 255], [230, 0, 0]]],
        dtype=np.uint8,
    )
    black_blue = np.array(
        [[[0, 0, 0], [0, 100, 200]]],
        dtype=np.uint8,
    )
    gray_blue = np.array(
        [[[180, 180, 180], [0, 100, 200]]],
        dtype=np.uint8,
    )

    normalized_red = MODULE.normalize_binary_api_result(red_white, "red")
    normalized_nonblack = MODULE.normalize_binary_api_result(black_blue, "nonblack")
    normalized_chromatic = MODULE.normalize_binary_api_result(gray_blue, "chromatic")

    expected = np.array([[[255, 255, 255], [230, 0, 0]]], dtype=np.uint8)
    assert np.array_equal(normalized_red, expected)
    assert np.array_equal(normalized_nonblack, expected)
    assert np.array_equal(normalized_chromatic, expected)


def test_tree_sha256_changes_when_cached_api_asset_changes(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"first")
    initial = MODULE.tree_sha256(tmp_path)

    (tmp_path / "a.png").write_bytes(b"second")
    changed = MODULE.tree_sha256(tmp_path)

    assert len(initial) == 64
    assert initial != changed
