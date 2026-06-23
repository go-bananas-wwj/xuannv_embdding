from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_train_task_module():
    path = Path(__file__).parents[1] / "scripts" / "train_task.py"
    spec = importlib.util.spec_from_file_location("train_task_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_for_early_stop_uses_configured_metric() -> None:
    module = _load_train_task_module()
    metrics = {"miou": 0.1, "f1_best": 0.4}
    assert module._score_for_early_stop(metrics, "f1_best") == pytest.approx(0.4)


def test_score_for_early_stop_raises_for_missing_metric() -> None:
    module = _load_train_task_module()
    with pytest.raises(KeyError, match="early_stop_metric"):
        module._score_for_early_stop({"miou": 0.1}, "f1_best")
