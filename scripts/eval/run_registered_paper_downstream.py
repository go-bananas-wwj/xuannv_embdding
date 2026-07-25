#!/usr/bin/env python3
"""Fail-closed helpers for registered paper downstream probes."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    """Return the content hash used by result provenance records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_encoder_provenance(
    config_path: Path,
    checkpoint_path: Path,
    expected_fold: int,
) -> dict[str, str | int]:
    """Reject an encoder checkpoint if its registered fold differs."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing encoder config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint_path}")
    if checkpoint_path.name != "best.pt":
        raise ValueError(
            "Registered evaluation requires the validation-selected encoder checkpoint "
            "named best.pt"
        )
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_fold = raw.get("data", {}).get("paper_fold")
    if configured_fold is None:
        raise ValueError(f"Encoder config does not declare data.paper_fold: {config_path}")
    if int(configured_fold) != expected_fold:
        raise ValueError(
            "Encoder paper_fold "
            f"{configured_fold} does not match requested evaluation fold {expected_fold}"
        )
    return {
        "encoder_fold": int(configured_fold),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def sigmoid_probabilities(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable sigmoid without per-image scaling."""
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


def binary_f1(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """Compute pooled binary F1 at one frozen threshold."""
    predicted = np.asarray(probabilities) >= threshold
    truth = np.asarray(targets).astype(bool)
    true_positive = int(np.logical_and(predicted, truth).sum())
    false_positive = int(np.logical_and(predicted, ~truth).sum())
    false_negative = int(np.logical_and(~predicted, truth).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def select_validation_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select F1 threshold on the registered validation grid only."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have identical flattened shapes")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape != targets.shape:
            raise ValueError("valid_mask must have the same flattened shape as targets")
    else:
        valid_mask = np.ones_like(targets, dtype=bool)
    valid_mask &= np.isin(targets, [0, 1])
    probabilities = probabilities[valid_mask]
    targets = targets[valid_mask].astype(np.uint8)
    if not np.isfinite(probabilities).all():
        raise ValueError("Validation probabilities must be finite")
    if targets.size == 0 or targets.sum() in {0, targets.size}:
        raise ValueError("Validation pixels must contain both positive and negative labels")
    best_threshold = 0.001
    best_f1 = -1.0
    for threshold in np.arange(0.001, 1.0, 0.001):
        score = binary_f1(probabilities, targets, float(threshold))
        if score >= best_f1:
            best_threshold = float(threshold)
            best_f1 = score
    return best_threshold, best_f1


def build_shot_manifest(
    task_name: str,
    train_ids: list[str],
    fold: int,
    seed: int,
    label_sha256: str,
    split_sha256: str,
    pixel_count: Callable[[str], int],
    budgets: tuple[int, ...] = (5, 10, 50),
    min_positive_pixels: int = 64,
) -> dict[str, Any]:
    """Create one immutable, nested positive/negative patch schedule.

    Patch ordering is sampled once at the largest budget.  Smaller shot levels
    are prefixes of that ordering, so all representations consume identical
    examples and the 5/10/50-shot sets are genuinely nested.
    """
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("Shot budgets must be non-empty positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Shot budgets must be strictly increasing without duplicates")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Training patch IDs must be unique for sampling without replacement")
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    for patch_id in train_ids:
        count = int(pixel_count(patch_id))
        if count >= min_positive_pixels:
            positive_ids.append(patch_id)
        elif count == 0:
            negative_ids.append(patch_id)
    rng = random.Random(seed + fold * 1009)
    rng.shuffle(positive_ids)
    rng.shuffle(negative_ids)
    largest_budget = budgets[-1]
    if len(positive_ids) < largest_budget or len(negative_ids) < largest_budget:
        raise RuntimeError(
            f"Exact {largest_budget}+{largest_budget} shot budget infeasible for {task_name}: "
            f"positive={len(positive_ids)} negative={len(negative_ids)}"
        )
    sets: dict[str, dict[str, list[str]]] = {}
    for budget in budgets:
        selected_pos = positive_ids[:budget]
        selected_neg = negative_ids[:budget]
        combined = selected_pos + selected_neg
        level_rng = random.Random(seed + fold * 1009 + budget)
        level_rng.shuffle(combined)
        sets[str(budget)] = {
            "positive_patch_ids": selected_pos,
            "negative_patch_ids": selected_neg,
            "train_patch_ids": combined,
        }
    return {
        "schema_version": 1,
        "task": task_name,
        "fold": fold,
        "seed": seed,
        "rule": {
            "min_positive_pixels": min_positive_pixels,
            "negative_rule": "exactly_zero_positive_pixels",
            "selection": "deterministic_nested_prefix",
        },
        "label_sha256": label_sha256,
        "split_sha256": split_sha256,
        "eligible_positive_count": len(positive_ids),
        "eligible_negative_count": len(negative_ids),
        "sets": sets,
    }
