"""launcher 使用的 no-follow/no-clobber 前台日志复制器。"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO

from experiments.china_v1_fusion_smoke.safety import (
    SafetyError,
    ensure_unsealed_sandbox,
    validate_write_path,
)


class SafeTeeError(RuntimeError):
    """日志路径不满足独占、常规文件约束时抛出。"""


def _guard(root: Path, log: Path) -> tuple[Path, Path]:
    try:
        sandbox = ensure_unsealed_sandbox(root)
        target = validate_write_path(log, sandbox)
    except SafetyError as exc:
        raise SafeTeeError(str(exc)) from exc
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SafeTeeError(f"log parent is missing or a symlink: {parent}")
    return sandbox, target


def reserve_log(root: Path, log: Path) -> Path:
    """用 O_EXCL/O_NOFOLLOW 独占预留一个空日志，绝不覆盖旧证据。"""
    _sandbox, target = _guard(root, log)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise SafeTeeError(f"launcher log already exists: {target}") from exc
    except OSError as exc:
        raise SafeTeeError(f"cannot safely reserve launcher log: {target}") from exc
    os.close(descriptor)
    return target


def stream_reserved_log(
    root: Path,
    log: Path,
    *,
    source: BinaryIO | None = None,
    destination: BinaryIO | None = None,
) -> None:
    """只向已独占预留的空常规文件追加，并同步复制到 stdout。"""
    _sandbox, target = _guard(root, log)
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(target, flags)
        metadata = os.fstat(descriptor)
        path_metadata = target.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
            raise SafeTeeError("reserved launcher log must be an empty regular file")
        if (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino):
            raise SafeTeeError("reserved launcher log changed before streaming")
        source = source if source is not None else sys.stdin.buffer
        destination = destination if destination is not None else sys.stdout.buffer
        while chunk := source.read(1024 * 1024):
            os.write(descriptor, chunk)
            destination.write(chunk)
            destination.flush()
        os.fsync(descriptor)
    except OSError as exc:
        raise SafeTeeError(f"cannot safely stream launcher log: {target}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reserve", action="store_true")
    mode.add_argument("--stream-reserved", action="store_true")
    args = parser.parse_args(argv)
    if args.reserve:
        reserve_log(args.sandbox_root, args.log)
    else:
        stream_reserved_log(args.sandbox_root, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
