"""Small, dependency-free helpers for safely reserving embedding export roots."""

from __future__ import annotations

from pathlib import Path


def resolve_export_root(output_root: Path, export_name: str | None, default_name: str) -> Path:
    """Return one output-root child while rejecting traversal and symlink redirects."""
    name = default_name if export_name is None else export_name
    candidate = Path(name)
    if not name or candidate.name != name or name in {".", ".."}:
        raise ValueError("--export-name must be a literal child directory name")
    output_path = output_root / name
    if output_path.is_symlink():
        raise ValueError("--export-name must not resolve through a symbolic link")
    return output_path
