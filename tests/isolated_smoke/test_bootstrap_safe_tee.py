from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest


def _bootstrap_module():
    return importlib.import_module("experiments.china_v1_fusion_smoke.bootstrap")


def _safe_tee_module():
    return importlib.import_module("experiments.china_v1_fusion_smoke.safe_tee")


def test_bootstrap_rejects_a_symlink_root_before_touching_its_target(tmp_path: Path) -> None:
    """根目录 symlink 不能让 bootstrap 在外部创建 sentinel、目录或 venv。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "sandbox"
    root.symlink_to(outside, target_is_directory=True)

    module = _bootstrap_module()
    with pytest.raises(module.BootstrapError, match="symlink"):
        module.bootstrap_sandbox(root, expected_root=root, create_environment=False)

    assert list(outside.iterdir()) == []


def test_bootstrap_rejects_a_symlink_ancestor_before_creating_root(tmp_path: Path) -> None:
    """任一既有祖先为 symlink 时，固定 root 自身也不得被创建。"""
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    root = alias / "sandbox"

    module = _bootstrap_module()
    with pytest.raises(module.BootstrapError, match="ancestor.*symlink"):
        module.bootstrap_sandbox(root, expected_root=root, create_environment=False)

    assert not (real_parent / "sandbox").exists()


def test_bootstrap_rejects_success_before_any_mutation(tmp_path: Path) -> None:
    """已封口沙箱必须在 mkdir、sentinel 或 venv 之前完全只读地失败。"""
    root = tmp_path / "sandbox"
    root.mkdir()
    success = root / "SUCCESS"
    success.write_text("sealed\n", encoding="utf-8")
    before = {path.name: path.lstat() for path in root.iterdir()}

    module = _bootstrap_module()
    with pytest.raises(module.BootstrapError, match="SUCCESS"):
        module.bootstrap_sandbox(root, expected_root=root, create_environment=False)

    after = {path.name: path.lstat() for path in root.iterdir()}
    assert set(after) == set(before) == {"SUCCESS"}
    assert success.read_text(encoding="utf-8") == "sealed\n"


def test_bootstrap_safely_creates_the_fixed_layout_and_regular_sentinel(tmp_path: Path) -> None:
    """合法空 root 应一次性得到真实目录和常规 sentinel，且无需 shell touch。"""
    root = tmp_path / "sandbox"

    module = _bootstrap_module()
    created = module.bootstrap_sandbox(root, expected_root=root, create_environment=False)

    assert created == root
    assert not root.is_symlink()
    sentinel = root / ".xuannv_isolated_smoke"
    assert sentinel.is_file() and not sentinel.is_symlink()
    for relative in (
        "cache",
        "synthetic/aef",
        "synthetic/highres_2m",
        "outputs",
        "checkpoints",
        "logs",
        "manifests",
        "tmp",
    ):
        path = root / relative
        assert path.is_dir() and not path.is_symlink()


def _sentinel_sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / ".xuannv_isolated_smoke").touch()
    return root


def test_safe_tee_rejects_logs_symlink_without_writing_outside(tmp_path: Path) -> None:
    """logs 祖先 symlink 不能把 launcher 输出引出固定 sandbox。"""
    root = _sentinel_sandbox(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "logs").symlink_to(outside, target_is_directory=True)

    module = _safe_tee_module()
    with pytest.raises(module.SafeTeeError, match="symlink"):
        module.reserve_log(root, root / "logs/npu_smoke.log")

    assert list(outside.iterdir()) == []


def test_safe_tee_rejects_log_symlink_without_touching_target(tmp_path: Path) -> None:
    """最终 log 自身为 symlink 时不能被打开或截断。"""
    root = _sentinel_sandbox(tmp_path)
    (root / "logs").mkdir()
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"preserve")
    (root / "logs/npu_smoke.log").symlink_to(outside)

    module = _safe_tee_module()
    with pytest.raises(module.SafeTeeError, match="symlink|exists"):
        module.reserve_log(root, root / "logs/npu_smoke.log")

    assert outside.read_bytes() == b"preserve"


def test_safe_tee_refuses_an_existing_log_without_truncating_it(tmp_path: Path) -> None:
    """失败后无 marker 的既有日志必须 fail closed，不能被下一次 launcher 覆盖。"""
    root = _sentinel_sandbox(tmp_path)
    (root / "logs").mkdir()
    log = root / "logs/npu_smoke.log"
    log.write_bytes(b"failed attempt evidence\n")

    module = _safe_tee_module()
    with pytest.raises(module.SafeTeeError, match="exists"):
        module.reserve_log(root, log)

    assert log.read_bytes() == b"failed attempt evidence\n"


def test_safe_tee_streams_only_into_the_exclusively_reserved_log(tmp_path: Path) -> None:
    """成功路径必须同时复制 stdout 和 log，且从不使用 truncate 打开方式。"""
    root = _sentinel_sandbox(tmp_path)
    (root / "logs").mkdir()
    log = root / "logs/npu_smoke.log"
    source = io.BytesIO(b"foreground output\n")
    forwarded = io.BytesIO()

    module = _safe_tee_module()
    module.reserve_log(root, log)
    module.stream_reserved_log(root, log, source=source, destination=forwarded)

    assert log.read_bytes() == b"foreground output\n"
    assert forwarded.getvalue() == b"foreground output\n"
