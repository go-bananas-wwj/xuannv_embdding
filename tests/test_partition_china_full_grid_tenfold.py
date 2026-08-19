from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "scripts/data/partition_china_full_grid_tenfold.py"


def load_module() -> object:
    assert MODULE_PATH.exists(), "The tenfold partition script must exist."
    spec = importlib.util.spec_from_file_location("partition_china_full_grid_tenfold", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_partition_assigns_every_shape_once_at_each_exact_capacity() -> None:
    """Catches a joint partition that loses, duplicates, or miscounts Shapes."""
    module = load_module()
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assignment = module.partition_equal_capacity(
        x=x,
        y=y,
        shard_capacities=[2, 2, 2, 2, 2, 2],
    )

    assert assignment.min() == 1
    assert assignment.max() == 6
    assert sorted(np.bincount(assignment, minlength=7)[1:].tolist()) == [2, 2, 2, 2, 2, 2]
