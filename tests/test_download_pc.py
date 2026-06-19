from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from scripts.data.download_pc import _validate_coverage, main


def _make_coverage_data_array(valid_ratio: float) -> xr.DataArray:
    """构造一个 1x1x2x4 的 DataArray（含 band 维度），按 valid_ratio 填充有效像素。"""
    total = 8
    n_valid = int(round(valid_ratio * total))
    values = np.concatenate(
        [np.ones(n_valid, dtype=np.float32), np.zeros(total - n_valid, dtype=np.float32)]
    )
    np.random.shuffle(values)
    data = values.reshape(1, 1, 2, 4)
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": [np.datetime64("2025-01-01")],
            "band": ["B01"],
        },
    )


def test_validate_coverage_passes_at_threshold() -> None:
    """有效像素比例为 0.5 时，阈值 0.5 应通过。"""
    arr = _make_coverage_data_array(valid_ratio=0.5)
    _validate_coverage(arr, min_valid_ratio=0.5)


def test_validate_coverage_fails_at_higher_threshold() -> None:
    """有效像素比例为 0.5 时，阈值 0.9 应失败。"""
    arr = _make_coverage_data_array(valid_ratio=0.5)
    with pytest.raises(ValueError, match="coverage"):
        _validate_coverage(arr, min_valid_ratio=0.9)


def test_main_accepts_min_valid_ratio_in_range() -> None:
    """--min-valid-ratio 在 [0, 1] 范围内时应返回 0。"""
    with patch(
        "scripts.data.download_pc.download_source",
        return_value=Path("/fake/output.nc"),
    ) as mock_download:
        rc = main(
            [
                "--region",
                "haidian",
                "--source",
                "s1",
                "--start",
                "2025-01-01",
                "--end",
                "2025-01-31",
                "--region-file",
                "/fake/region.geojson",
                "--min-valid-ratio",
                "0.01",
            ]
        )
    assert rc == 0
    assert mock_download.call_args.kwargs["min_valid_ratio"] == pytest.approx(0.01)


def test_main_rejects_min_valid_ratio_out_of_range() -> None:
    """--min-valid-ratio 超出 [0, 1] 范围时应返回非 0。"""
    rc = main(
        [
            "--region",
            "haidian",
            "--source",
            "s1",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-31",
            "--region-file",
            "/fake/region.geojson",
            "--min-valid-ratio",
            "1.5",
        ]
    )
    assert rc != 0
