"""为固定 China V1 smoke 根目录执行不跟随 symlink 的最小 bootstrap。"""

from __future__ import annotations

import os
import stat
import venv
from pathlib import Path

from experiments.china_v1_fusion_smoke.safety import SENTINEL


class BootstrapError(RuntimeError):
    """bootstrap 目标不是预期固定、安全根目录时抛出。"""


_LAYOUT = (
    "cache",
    "synthetic/aef",
    "synthetic/highres_2m",
    "outputs",
    "checkpoints",
    "logs",
    "manifests",
    "tmp",
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _check_existing_ancestors(path: Path) -> None:
    for candidate in (*reversed(path.parents), path):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            label = "root" if candidate == path else "ancestor"
            raise BootstrapError(f"sandbox {label} must not be a symlink: {candidate}")
        if candidate != path and not stat.S_ISDIR(metadata.st_mode):
            raise BootstrapError(f"sandbox ancestor must be a directory: {candidate}")


def _mkdir_chain(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        os.mkdir(directory, mode=0o700)


def bootstrap_sandbox(
    root: Path,
    *,
    expected_root: Path,
    create_environment: bool = True,
) -> Path:
    """仅在精确根目录中创建 sentinel、固定布局和只读复用依赖的 venv。"""
    absolute = _absolute(root)
    expected = _absolute(expected_root)
    if absolute != expected:
        raise BootstrapError(f"sandbox root differs from fixed expected root: {absolute}")
    _check_existing_ancestors(absolute)
    success = absolute / "SUCCESS"
    if success.exists() or success.is_symlink():
        raise BootstrapError("SUCCESS exists; sealed sandbox must not be bootstrapped")
    if absolute.exists() and not absolute.is_dir():
        raise BootstrapError(f"sandbox root must be a directory: {absolute}")

    _mkdir_chain(absolute)
    sentinel = absolute / SENTINEL
    if sentinel.exists() or sentinel.is_symlink():
        if sentinel.is_symlink() or not sentinel.is_file():
            raise BootstrapError("isolated smoke sentinel must be a regular file")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(sentinel, flags, 0o600)
        os.close(descriptor)

    for relative in _LAYOUT:
        directory = absolute / relative
        _check_existing_ancestors(directory)
        _mkdir_chain(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise BootstrapError(f"bootstrap layout path is unsafe: {directory}")

    if create_environment:
        environment = absolute / "env"
        _check_existing_ancestors(environment)
        if environment.exists():
            if environment.is_symlink() or not environment.is_dir():
                raise BootstrapError(f"environment path is unsafe: {environment}")
        else:
            venv.EnvBuilder(system_site_packages=True, with_pip=False).create(environment)
    return absolute


def main() -> int:
    fixed = Path("/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815")
    bootstrap_sandbox(fixed, expected_root=fixed)
    print(fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
