from __future__ import annotations

from pathlib import Path

SENTINEL = ".xuannv_isolated_smoke"


class SafetyError(RuntimeError):
    """隔离 smoke test 试图越过沙箱边界时抛出。"""


def ensure_sandbox(root: Path) -> Path:
    """验证一个已创建且带有隔离哨兵文件的 smoke 沙箱。"""
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise SafetyError(f"sandbox is not a directory: {resolved}")
    if not (resolved / SENTINEL).is_file():
        raise SafetyError("isolated smoke sentinel is missing")
    return resolved


def validate_write_path(path: Path, sandbox_root: Path) -> Path:
    """只允许写入带 sentinel 的精确 smoke 沙箱及其子目录。"""
    root = ensure_sandbox(sandbox_root)
    resolved = path.resolve(strict=False)
    if resolved == root or resolved.is_relative_to(root):
        return resolved
    raise SafetyError(f"write path escapes sandbox: {resolved}")
