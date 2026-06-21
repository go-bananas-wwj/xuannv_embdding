#!/usr/bin/env python3
"""Collect best-fold head weights and configs for the v1.0 release.

Reads ``/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json``,
for each task determines the best fold (highest ``f1_best`` from the per-fold
summary, defaulting to fold 0), and copies:

* ``fold_<best>/checkpoints/best.pt`` -> ``heads/<task>_fold<best>_best.pt``
* the task config -> ``configs/<task>.yaml``
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

V1_ROOT = Path("/data/xuannv_embedding/experiments/v1.0")
SUMMARY_JSON = V1_ROOT / "all_tasks_summary_final.json"
HEADS_DIR = V1_ROOT / "heads"
CONFIGS_DIR = V1_ROOT / "configs"

# Two parents: scripts/ -> downstreams/ -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SEARCH_DIR = REPO_ROOT / "downstreams" / "configs"

STOPWORDS = {
    "yaml",
    "yml",
    "v1",
    "v2",
    "v3",
    "v4",
    "5fold",
    "experiments",
    "data",
}


def tokenize(text: str) -> set[str]:
    """Split a string into lowercase alphanumeric tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in STOPWORDS}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_fold(task_dir: Path) -> int:
    """Return the fold with the highest f1_best from the per-fold summary."""
    summary_path = task_dir / "summary_5fold.json"
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
            folds = summary.get("folds", [])
            if folds:
                best = max(folds, key=lambda fold: fold.get("f1_best", -1.0))
                return int(best.get("fold", 0))
        except Exception as exc:  # pragma: no cover
            print(f"  Warning: could not read {summary_path}: {exc}", file=sys.stderr)

    # Fallback: look for individual fold metrics files.
    best_fold = 0
    best_f1 = -1.0
    for fold_dir in sorted(task_dir.glob("fold*")):
        match = re.match(r"fold_?(\d+)", fold_dir.name)
        if not match:
            continue
        fold_idx = int(match.group(1))
        metrics_file = fold_dir / f"fold_{fold_idx}" / "metrics.json"
        if not metrics_file.exists():
            metrics_file = fold_dir / "metrics.json"
        if metrics_file.exists():
            try:
                metrics = load_json(metrics_file)
                f1 = metrics.get("f1_best", -1.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_fold = fold_idx
            except Exception:  # pragma: no cover
                continue
    return best_fold


def find_checkpoint(task_dir: Path, fold: int) -> Path | None:
    """Return the path to the best checkpoint for a fold, if it exists."""
    candidates = [
        task_dir / f"fold_{fold}" / "checkpoints" / "best.pt",
        task_dir / f"fold{fold}" / "checkpoints" / "best.pt",
        task_dir / f"fold{fold}" / f"fold_{fold}" / "checkpoints" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_config(task_dir: Path, task: str) -> Path | None:
    """Discover the config file used for a task."""
    # 1. Prefer a YAML file inside the experiment directory (excluding outputs).
    yaml_in_exp = [
        p
        for p in task_dir.rglob("*.yaml")
        if p.name != "_base_.yaml" and "visualizations" not in p.parts
    ]
    if len(yaml_in_exp) == 1:
        return yaml_in_exp[0]

    # 2. Fall back to downstreams/configs/.
    if not CONFIG_SEARCH_DIR.exists():
        return None

    config_files = [p for p in CONFIG_SEARCH_DIR.glob("*.yaml") if p.name != "_base_.yaml"]
    if not config_files:
        return None

    target_name = task_dir.resolve().name
    target_tokens = tokenize(task) | tokenize(target_name)

    def score(path: Path) -> float:
        candidate_tokens: set[str] = tokenize(path.stem)
        try:
            text = path.read_text(encoding="utf-8")
            name_match = re.search(
                r"^experiment:\s*\n(?:\s+.*\n)*?\s+name:\s*(.+?)$",
                text,
                re.MULTILINE,
            )
            if name_match:
                candidate_tokens |= tokenize(name_match.group(1))
        except Exception:  # pragma: no cover
            pass
        if not candidate_tokens or not target_tokens:
            return 0.0
        intersection = candidate_tokens & target_tokens
        union = candidate_tokens | target_tokens
        return len(intersection) / len(union)

    ranked = sorted(config_files, key=score, reverse=True)
    best = ranked[0]
    best_score = score(best)
    if best_score == 0.0:
        # Last resort: pick a config whose filename contains the task name.
        for path in ranked:
            if task.replace("_", "").lower() in path.stem.replace("_", "").lower():
                return path
        return None
    return best


def collect_task(task: str, _meta: dict[str, Any]) -> bool:
    print(f"Task: {task}")
    task_dir = V1_ROOT / task
    if not task_dir.exists():
        print(f"  ERROR: task directory not found: {task_dir}", file=sys.stderr)
        return False

    best_fold = find_best_fold(task_dir)
    print(f"  best_fold = {best_fold}")

    ok = True

    # Head weight
    ckpt = find_checkpoint(task_dir, best_fold)
    if ckpt is None:
        print(f"  ERROR: best checkpoint not found for fold {best_fold}", file=sys.stderr)
        ok = False
    else:
        dest = HEADS_DIR / f"{task}_fold{best_fold}_best.pt"
        shutil.copy2(ckpt, dest)
        print(f"  copied head -> {dest}")

    # Config
    config = discover_config(task_dir, task)
    if config is None:
        print(f"  ERROR: config not found for {task}", file=sys.stderr)
        ok = False
    else:
        dest = CONFIGS_DIR / f"{task}.yaml"
        shutil.copy2(config, dest)
        print(f"  copied config -> {dest} (from {config.name})")

    return ok


def main() -> int:
    if not SUMMARY_JSON.exists():
        print(f"ERROR: summary not found: {SUMMARY_JSON}", file=sys.stderr)
        return 1

    summary = load_json(SUMMARY_JSON)
    if not summary:
        print("ERROR: summary is empty", file=sys.stderr)
        return 1

    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for task, meta in summary.items():
        if not collect_task(task, meta):
            all_ok = False
        print()

    # Verification: ensure every task has exactly one config and one head.
    expected = len(summary)
    verification_ok = True
    for task in summary:
        config_path = CONFIGS_DIR / f"{task}.yaml"
        if not config_path.exists():
            print(f"ERROR: missing config for {task}", file=sys.stderr)
            verification_ok = False
        heads = list(HEADS_DIR.glob(f"{task}_fold*_best.pt"))
        if len(heads) != 1:
            print(
                f"ERROR: expected exactly one head for {task}, found {len(heads)}",
                file=sys.stderr,
            )
            verification_ok = False

    n_heads = len(list(HEADS_DIR.glob("*.pt")))
    n_configs = len(list(CONFIGS_DIR.glob("*.yaml")))
    print(f"Collected {n_heads} head weights and {n_configs} configs.")
    if n_heads != expected or n_configs != expected:
        print(
            f"ERROR: expected {expected} heads and {expected} configs",
            file=sys.stderr,
        )
        return 1

    return 0 if (all_ok and verification_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
