from __future__ import annotations

import tomllib
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


def test_ensure_sandbox_rejects_root_symlink_to_outside(tmp_path: Path) -> None:
    """固定 sandbox 路径自身为 symlink 时不得跟随到带 sentinel 的外部目录。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".xuannv_isolated_smoke").touch()
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SafetyError, match="symlink"):
        ensure_sandbox(root)


def test_ensure_sandbox_rejects_sentinel_symlink(tmp_path: Path) -> None:
    """sentinel 必须是 sandbox 内真实常规文件，不能是指向任意文件的 symlink。"""
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.mkdir()
    outside = tmp_path / "outside-sentinel"
    outside.touch()
    (root / ".xuannv_isolated_smoke").symlink_to(outside)

    with pytest.raises(SafetyError, match="sentinel.*symlink"):
        ensure_sandbox(root)


def test_smoke_zarr_dependencies_are_declared_as_an_optional_extra() -> None:
    """隔离 smoke 的直接 Zarr imports 必须有可安装、受版本约束的项目依赖合同。"""
    raw = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert raw["project"]["optional-dependencies"]["smoke"] == [
        "zarr>=2.18,<3",
        "numcodecs>=0.12,<0.16",
    ]
