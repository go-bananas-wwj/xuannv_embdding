from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
POLICY = (
    ROOT
    / "configs/national/china_quarterly_2020_2021_sampling_policy_20260726.json"
)


def _policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def test_sampling_layers_sum_to_frozen_total() -> None:
    policy = _policy()
    sampling = policy["sampling"]
    assert sum(layer["count"] for layer in sampling["layers"]) == 62_000
    assert sampling["target_total"] == 62_000
    assert sampling["base_count"] == 57_405
    assert sampling["sampling_seed"] == 20260717
    assert sampling["final_land_fraction"] == 0.0106
    assert "planning estimate" in sampling["final_land_fraction_status"]


def test_semantic_supplement_subclasses_sum_to_layer_total() -> None:
    policy = _policy()
    semantic_layer = next(
        layer
        for layer in policy["sampling"]["layers"]
        if layer["id"] == "semantic_and_difficult_supplement"
    )
    assert sum(semantic_layer["subquotas"].values()) == semantic_layer["count"] == 1_500


def test_quarterly_products_cover_2020_and_2021() -> None:
    policy = _policy()
    assert policy["products"]["periods"] == [
        "2020Q1",
        "2020Q2",
        "2020Q3",
        "2020Q4",
        "2021Q1",
        "2021Q2",
        "2021Q3",
        "2021Q4",
    ]
    assert policy["products"]["shared_spatial_registry"] is True


def test_patch_geometry_matches_model_output() -> None:
    patch = _policy()["patch"]
    assert patch["side_meters"] == 1_280
    assert patch["pixels"] == 128
    assert patch["resolution_meters"] == 10
    assert patch["embedding_dimensions"] == 64


def test_development_split_and_final_fit_are_distinct() -> None:
    training = _policy()["training_use"]
    assert sum(training["development_split"]["fractions"].values()) == 1.0
    assert training["development_split"]["spatial_blocked"] is True
    assert training["final_fit"]["uses_all_accepted_or_partial_patches"] is True
    assert training["final_fit"]["allowed_after_protocol_freeze"] is True


def test_unknown_osm_is_not_a_negative_label() -> None:
    semantics = _policy()["weak_semantics"]
    assert semantics["blank_means"] == "unknown"
    assert semantics["uses_final_downstream_manual_labels"] is False


def test_policy_is_honest_about_pipeline_compatibility() -> None:
    status = _policy()["implementation_status"]
    assert status["end_to_end_materialization"] == "not_implemented"
    assert status["legacy_build_national_sampling_registry_compatible"] is False


def test_snow_is_preserved_as_a_surface_state() -> None:
    quality = _policy()["quality_gate"]
    assert "snow" not in quality["invalid_pixel_masks"]
    assert "snow" in quality["preserved_surface_state_masks"]
    assert quality["numeric_thresholds"]["minimum_passing_modalities_per_quarter"] == 2
    assert quality["maximum_partial_fraction_per_quarter"] == 0.25


def test_cross_zone_overlap_and_spatial_split_are_bounded() -> None:
    policy = _policy()
    deduplication = policy["sampling"]["deduplication"]
    development = policy["training_use"]["development_split"]
    assert deduplication["cross_zone_spatial_index_check"] is True
    assert deduplication["maximum_pairwise_overlap_fraction"] == 0.01
    assert development["block_grid"] == "50 km parent blocks in EPSG:6933"
    assert development["minimum_cross_split_buffer_meters"] == 25_600
