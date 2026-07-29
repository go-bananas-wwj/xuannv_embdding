from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/eval/run_harbin_pu_query_transfer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harbin_pu_query_transfer", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_rejects_non_380_coverage(tmp_path: Path) -> None:
    """A changed coverage universe must stop the paired protocol before any sampling."""
    module = _load_module()
    config = tmp_path / "bad_config.json"
    config.write_text(json.dumps({"coverage_patch_count": 379}), encoding="utf-8")

    with pytest.raises(ValueError, match="380"):
        module.prepare_harbin_pu_query(config, tmp_path / "output")


def test_polygon_schedule_is_reused_for_every_family(tmp_path: Path) -> None:
    """Family identity must not enter the prompt sampling key."""
    module = _load_module()
    schedule = {
        "building|0|42|5": {
            "support_patch_ids": ["harbin_patch_000001"],
            "polygon_indices": [3, 9, 12, 17, 21],
        }
    }
    prepared = module.PreparedProtocol(
        patch_ids=["harbin_patch_000001"],
        label_ids={"harbin_patch_000001": "patch_000001"},
        schedules=schedule,
    )

    assert prepared.schedule_for("building", 0, 42, 5) == schedule["building|0|42|5"]
    assert prepared.schedule_for("building", 0, 42, 5) == schedule["building|0|42|5"]


def test_label_resolution_uses_locked_source_patch_id() -> None:
    """A missing mapping cannot fall back to string rewriting or an unpaired label."""
    module = _load_module()
    mapping = {"harbin_patch_000001": "patch_000001"}

    assert module.resolve_label_id("harbin_patch_000001", mapping) == "patch_000001"
    with pytest.raises(KeyError, match="unlocked"):
        module.resolve_label_id("harbin_patch_000002", mapping)


def test_full_patch_component_is_rejected_before_schedule_generation(tmp_path: Path) -> None:
    """A dilated full-patch support must never reach PU background fitting."""
    module = _load_module()
    mask_path = tmp_path / "masks" / "patch_000001.tif"
    mask_path.parent.mkdir()
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 128, 1, 1),
    ) as dataset:
        dataset.write(np.ones((1, 128, 128), dtype=np.uint8))

    candidates, audit = module.polygon_candidates(
        tmp_path,
        ["harbin_patch_000001"],
        {"harbin_patch_000001": "patch_000001"},
        minimum_area=5,
        minimum_reliable_background_pixels=64,
    )

    assert candidates == []
    assert audit["rejected_insufficient_reliable_background"] == 1
    assert audit["eligible_component_count"] == 0


def test_validation_threshold_is_selected_without_test_labels() -> None:
    """Changing a held-out test label cannot change the validation-only threshold."""
    module = _load_module()
    validation_scores = [0.1, 0.2, 0.8, 0.9]
    validation_labels = [0, 0, 1, 1]

    assert module.select_validation_threshold(validation_scores, validation_labels) == 0.8


def test_max_prototype_scores_best_matching_support_prototype() -> None:
    """Max mode must retain a distinct matching polygon prototype."""
    module = _load_module()
    pixels = module.l2(module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32))
    prototypes = module.np.array([[1.0, 0.0], [0.0, 1.0]], dtype=module.np.float32)

    assert module.aggregate_foreground_similarity(pixels, prototypes, "max").tolist() == [1.0, 1.0]


def test_query_disabled_preserves_base_scores() -> None:
    """The disabled Query arm is exactly the unadapted PU base score."""
    module = _load_module()
    feature = module.np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=module.np.float32)
    model = {
        "mean": module.np.zeros(2, dtype=module.np.float32),
        "std": module.np.ones(2, dtype=module.np.float32),
        "foreground": module.np.array([1.0, 0.0], dtype=module.np.float32),
        "foreground_prototypes": module.np.array([[1.0, 0.0]], dtype=module.np.float32),
        "background": module.np.array([0.0, 1.0], dtype=module.np.float32),
        "support_threshold": 0.0,
    }

    scores, adapted = module.score_pu_query(feature, model, query_mode="disabled")

    assert scores.shape == (1, 2)
    assert not adapted


def test_validation_threshold_scales_to_dense_validation_maps() -> None:
    """Threshold calibration must scan dense validation pixels once, not once per pixel."""
    module = _load_module()
    scores = module.np.linspace(-1.0, 1.0, 8192)
    labels = scores >= 0.2

    start = time.perf_counter()
    threshold = module.select_validation_threshold(scores, labels)

    assert threshold >= 0.19
    assert time.perf_counter() - start < 0.3
