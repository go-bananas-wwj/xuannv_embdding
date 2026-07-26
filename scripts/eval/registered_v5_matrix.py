"""Lightweight, fail-closed loader for the registered V5 experiment matrix.

The shell entrypoints use this module before any NPU/runtime setup.  Keeping it
free of PyTorch and raster dependencies makes dry-run admission deterministic
and quick while retaining the same Git-HEAD, checksum, and protocol checks as
the full downstream runner.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

V5_MATRIX = Path("configs/eval/rse_v5_osm_assisted_matrix.json")
V5_MATRIX_SHA256 = "b9f243c9583a35f36a0792ab0c21fb08334553ba02db17b78fad766ad4ad20ca"
V5_SPLIT = Path("configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json")
V5_SPLIT_SHA256 = "9a6d98d6d6456ce0a791ef74ac725e4d360e7d4e0a17c9cb5edfceb850093c0b"
V5_EVAL_MANIFEST_SHA256 = "9bd55c663616322c13804b7c8c0d2e32860ca1fda1d2e5d59dfca8404b901ab3"
V5_EVAL_MANIFEST = Path(
    "/data/xuannv_embedding/processed/haidian/"
    "manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"
)
V5_STATISTICS_REGISTRY_SHA256 = "ba1fb10bc105a29286750367dff2ab45e02f89c32f69725ab89da994d0c56929"


def sha256_file(path: Path) -> str:
    """Return the content hash used by registered protocol records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_head_file(path: Path) -> None:
    """Require a Git-tracked file whose bytes match its current HEAD revision."""
    resolved = path.resolve()
    root = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if root.returncode != 0 or not root.stdout.strip():
        raise ValueError("External registry must be inside a Git repository")
    repo_root = Path(root.stdout.strip()).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("External registry must be inside the repository") from exc
    relative_text = str(relative)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_text],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("External registry must be Git tracked")
    head = subprocess.run(
        ["git", "show", f"HEAD:{relative_text}"], cwd=repo_root, capture_output=True, check=False
    )
    if head.returncode != 0 or head.stdout != resolved.read_bytes():
        raise ValueError("External registry must exactly match the current Git HEAD")


def _validate_v5_matrix_bindings(matrix: dict[str, Any]) -> None:
    """Check the immutable V5 input bindings without importing training code."""
    required = {
        "spatial_split": str(V5_SPLIT),
        "spatial_split_sha256": V5_SPLIT_SHA256,
        "manifest_sha256": V5_EVAL_MANIFEST_SHA256,
        "statistics_registry_sha256": V5_STATISTICS_REGISTRY_SHA256,
    }
    for key, expected in required.items():
        if matrix.get(key) != expected:
            raise ValueError(f"Registered v5 matrix {key} does not match the pinned protocol")


def load_registered_v5_matrix(
    matrix_path: Path | None = None, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Load the sealed V5 matrix without importing downstream training dependencies."""
    path = matrix_path or V5_MATRIX
    expected_sha = expected_sha256 or V5_MATRIX_SHA256
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered v5 matrix: {path}")
    verify_git_head_file(path)
    if sha256_file(path) != expected_sha:
        raise ValueError("Registered v5 matrix hash does not match the pinned SHA-256")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1 or matrix.get("protocol_id") != "v5_osm_assisted":
        raise ValueError("Registered v5 matrix has an invalid schema or protocol")
    _validate_v5_matrix_bindings(matrix)
    if not isinstance(matrix.get("families"), dict) or not isinstance(matrix.get("probe"), dict):
        raise ValueError("Registered v5 matrix lacks family or probe declarations")
    return matrix
