from __future__ import annotations

from pathlib import Path

import pytest

from experiments.china_v1_fusion_smoke.config import load_smoke_config
from experiments.china_v1_fusion_smoke.safety import (
    SafetyError,
    ensure_sandbox,
    validate_write_path,
)

CONFIG_PATH = Path("configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml")


def test_checked_config_is_fixed_to_four_patches_two_years_and_npu2():
    cfg = load_smoke_config(CONFIG_PATH)

    assert cfg.max_patches == 4
    assert cfg.years == (2020, 2021)
    assert cfg.physical_npu == 2
    assert cfg.max_optimizer_steps == 2


@pytest.mark.parametrize(
    "unsafe",
    [
        "/root/workspace/xuannv",
        "/data/xuannv_embedding/processed",
        "/data/xuannv_embedding/outputs",
        "/data/xuannv_embedding/embeddings",
        "/data2/china_xuannv_embedding/data",
    ],
)
def test_write_guard_rejects_production_roots(unsafe: str, tmp_path: Path):
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.mkdir()
    (root / ".xuannv_isolated_smoke").touch()

    with pytest.raises(SafetyError):
        validate_write_path(Path(unsafe), root)


def test_ensure_sandbox_requires_the_isolated_smoke_sentinel(tmp_path: Path):
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.mkdir()

    with pytest.raises(SafetyError, match="sentinel"):
        ensure_sandbox(root)


def test_write_guard_allows_a_new_file_inside_the_sentinel_sandbox(tmp_path: Path):
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.mkdir()
    (root / ".xuannv_isolated_smoke").touch()

    assert validate_write_path(root / "outputs" / "result.json", root) == (
        root / "outputs" / "result.json"
    )
