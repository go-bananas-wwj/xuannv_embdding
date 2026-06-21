#!/usr/bin/env python3
"""Generate the v1.0 release manifest.

Reads ``/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json``,
computes SHA-256 checksums for the best-fold head weights and the encoder
checkpoint, captures the current git commit of the worktree, and writes
``/data/xuannv_embedding/experiments/v1.0/v1.0_manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

V1_ROOT = Path("/data/xuannv_embedding/experiments/v1.0")
SUMMARY_JSON = V1_ROOT / "all_tasks_summary_final.json"
HEADS_DIR = V1_ROOT / "heads"
CONFIGS_DIR = V1_ROOT / "configs"
ENCODER_DIR = V1_ROOT / "encoder"

# Repository root is two parents above this script: scripts/ -> downstreams/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """Return the current git commit hash of the worktree."""
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.STDOUT,
        )
        .decode()
        .strip()
    )


def git_branch() -> str:
    """Return the current git branch of the worktree."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def infer_best_fold(task: str) -> int | None:
    """Infer the best fold for a task from the collected head weight filename."""
    candidates = list(HEADS_DIR.glob(f"{task}_fold*_best.pt"))
    if not candidates:
        return None
    # If multiple matches, prefer the one with the highest fold number as a tie
    # breaker (collect_v1.0_artifacts.py copies only one head per task).
    best = sorted(candidates, key=lambda p: int(p.stem.split("_fold")[1].split("_")[0]))[-1]
    fold_part = best.stem.split("_fold")[1]
    return int(fold_part.split("_")[0])


def build_task_entry(task: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Build a manifest entry for a single downstream task."""
    best_fold = infer_best_fold(task)
    if best_fold is None:
        print(f"WARNING: could not infer best fold for {task}", file=sys.stderr)
        best_fold = 0

    head_path = HEADS_DIR / f"{task}_fold{best_fold}_best.pt"
    cfg_path = CONFIGS_DIR / f"{task}.yaml"

    entry: dict[str, Any] = {
        "best_fold": best_fold,
        "metrics": {k: v for k, v in meta.items() if k != "status"},
        "status": meta.get("status", "unknown"),
        "head_file": str(head_path.relative_to(V1_ROOT)),
        "head_sha256": sha256_file(head_path) if head_path.exists() else None,
        "config_file": str(cfg_path.relative_to(V1_ROOT)) if cfg_path.exists() else None,
        "experiment_dir": str((V1_ROOT / task).relative_to(V1_ROOT)),
    }
    return entry


def build_encoder_entry() -> dict[str, Any]:
    """Build the manifest entry describing the encoder checkpoint files."""
    if not ENCODER_DIR.exists():
        return {"files": [], "sha256": {}}

    files = sorted(str(p.relative_to(V1_ROOT)) for p in ENCODER_DIR.rglob("*") if p.is_file())
    checksums = {
        str(p.relative_to(V1_ROOT)): sha256_file(p) for p in ENCODER_DIR.rglob("*") if p.is_file()
    }
    return {"files": files, "sha256": checksums}


def main() -> int:
    if not SUMMARY_JSON.exists():
        print(f"ERROR: summary not found: {SUMMARY_JSON}", file=sys.stderr)
        return 1

    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    manifest: dict[str, Any] = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_branch(),
        "git_commit": git_commit(),
        "github_release_tag": "v1.0-multitask-downstream",
        "modelscope_repo": "WeijieWu/xuannv_train_data",
        "modelscope_tag": "v1.0",
        "tasks": {},
        "encoder": build_encoder_entry(),
    }

    for task, meta in summary.items():
        manifest["tasks"][task] = build_task_entry(task, meta)

    out_path = V1_ROOT / "v1.0_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Manifest written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
