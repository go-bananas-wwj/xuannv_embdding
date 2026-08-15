from __future__ import annotations

import os
from pathlib import Path

SENTINEL = ".xuannv_isolated_smoke"


class SafetyError(RuntimeError):
    """隔离 smoke test 试图越过沙箱边界时抛出。"""


def ensure_sandbox(root: Path) -> Path:
    """验证一个已创建且带有隔离哨兵文件的 smoke 沙箱。"""
    absolute = Path(os.path.abspath(root))
    if absolute.is_symlink():
        raise SafetyError(f"sandbox root must not be a symlink: {absolute}")
    for ancestor in absolute.parents:
        if ancestor.is_symlink():
            raise SafetyError(f"sandbox ancestor must not be a symlink: {ancestor}")
    if not absolute.is_dir():
        raise SafetyError(f"sandbox is not a directory: {absolute}")
    sentinel = absolute / SENTINEL
    if sentinel.is_symlink():
        raise SafetyError("isolated smoke sentinel must not be a symlink")
    if not sentinel.is_file():
        raise SafetyError("isolated smoke sentinel is missing")
    return absolute


def validate_write_path(path: Path, sandbox_root: Path) -> Path:
    """只允许写入无 symlink 的 sentinel 沙箱及其无 symlink 子路径。"""
    root = ensure_sandbox(sandbox_root)
    candidate = Path(os.path.abspath(path))
    if candidate != root and not candidate.is_relative_to(root):
        raise SafetyError(f"write path escapes sandbox: {candidate}")
    current = root
    for component in candidate.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise SafetyError(f"write path contains a symlink: {current}")
    resolved = candidate.resolve(strict=False)
    if resolved != root and not resolved.is_relative_to(root):
        raise SafetyError(f"write path escapes sandbox: {resolved}")
    return candidate
