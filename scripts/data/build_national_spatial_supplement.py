#!/usr/bin/env python3
"""Top up the China V1 static preview with spatially stratified supplements.

This implements the first half of an AEF-style representative sampling design:
a broad spatial backbone followed by deterministic stratified random points.
It deliberately does not claim land-cover balancing before the nationwide OSM
and land-cover strata have been audited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd


PREVIEW_PATH = Path(__file__).with_name("visualize_national_sampling.py")
SPEC = importlib.util.spec_from_file_location("national_sampling_preview", PREVIEW_PATH)
assert SPEC is not None and SPEC.loader is not None
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _allocate(total: int, groups: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Allocate by sqrt(area) to preserve coverage while lifting small UTM zones."""
    weights = {name: math.sqrt(len(records)) for name, records in groups.items() if records}
    denominator = sum(weights.values())
    raw = {name: total * weight / denominator for name, weight in weights.items()}
    allocated = {name: min(len(groups[name]), int(math.floor(value))) for name, value in raw.items()}
    residual = total - sum(allocated.values())
    ranked = sorted(weights, key=lambda item: (raw[item] - allocated[item], item), reverse=True)
    while residual:
        changed = False
        for name in ranked:
            if residual == 0:
                break
            if allocated[name] < len(groups[name]):
                allocated[name] += 1
                residual -= 1
                changed = True
        if not changed:
            break
    if residual:
        raise ValueError("not enough unselected macrocells to reach requested target")
    return allocated


def build_supplement(
    inventory: list[dict[str, Any]],
    base_points: list[dict[str, Any]],
    country_geometry: Any,
    seed: int,
    target_total: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if target_total < len(base_points):
        raise ValueError("target_total cannot be below the existing base sample size")
    need = target_total - len(base_points)
    base_macro_ids = {str(point["macro_id"]) for point in base_points}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for macro in inventory:
        if str(macro["macro_id"]) not in base_macro_ids:
            groups[str(macro["grid_id"])].append(macro)
    allocations = _allocate(need, groups)
    selected: list[dict[str, Any]] = []
    selected_macros: set[str] = set()
    selected_by_group: Counter[str] = Counter()
    for group, quota in sorted(allocations.items()):
        candidates = sorted(
            groups[group],
            key=lambda macro: PREVIEW._unit_hash(seed, f"spatial-supplement:{macro['macro_id']}"),
        )
        for macro in candidates:
            if selected_by_group[group] >= quota:
                break
            # Reuse the boundary-safe deterministic candidate order from the base sampler.
            point = PREVIEW.select_static_preview(
                [{**macro, "estimated_patch_count": 100.0}], country_geometry, seed
            )
            if not point:
                continue
            record = point[0]
            record["status"] = "provisional_spatial_stratified_supplement_not_quality_eligible"
            record["sampling_tier"] = "spatial_stratified_supplement"
            selected.append(record)
            selected_macros.add(str(macro["macro_id"]))
            selected_by_group[group] += 1
    # Sliver/coastal macrocells can intersect the boundary while containing no
    # complete patch center. Fill their quota from valid candidates elsewhere.
    if len(selected) < need:
        fallback = sorted(
            (macro for records in groups.values() for macro in records if str(macro["macro_id"]) not in selected_macros),
            key=lambda macro: PREVIEW._unit_hash(seed, f"spatial-supplement-fallback:{macro['macro_id']}"),
        )
        for macro in fallback:
            if len(selected) == need:
                break
            point = PREVIEW.select_static_preview(
                [{**macro, "estimated_patch_count": 100.0}], country_geometry, seed
            )
            if not point:
                continue
            record = point[0]
            record["status"] = "provisional_spatial_stratified_supplement_not_quality_eligible"
            record["sampling_tier"] = "spatial_stratified_supplement_fallback"
            selected.append(record)
            selected_macros.add(str(macro["macro_id"]))
            selected_by_group[str(macro["grid_id"])] += 1
    if len(selected) != need:
        raise RuntimeError(f"selected {len(selected)} supplements, expected {need}")
    return selected, dict(sorted(selected_by_group.items()))


def render_map(country_path: Path, adm1_path: Path, base: list[dict[str, Any]], supplement: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    country = gpd.read_file(country_path).to_crs("EPSG:4326")
    admin1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
    fig, axis = plt.subplots(figsize=(15, 10), dpi=220)
    country.plot(ax=axis, color="#f4f1e8", edgecolor="#303030", linewidth=0.7, zorder=1)
    admin1.boundary.plot(ax=axis, color="#9a9a9a", linewidth=0.24, zorder=2)
    for records, color, label, zorder in (
        (base, "#c5221f", f"Base spatial sample ({len(base):,})", 3),
        (supplement, "#1464a0", f"Spatial-stratified supplement ({len(supplement):,})", 4),
    ):
        axis.scatter(
            [record["longitude"] for record in records],
            [record["latitude"] for record in records],
            s=0.48, c=color, alpha=0.72, label=label, linewidths=0, zorder=zorder,
        )
    min_x, min_y, max_x, max_y = country.total_bounds
    axis.set_xlim(min_x - 1.5, max_x + 1.5)
    axis.set_ylim(min_y - 1.5, max_y + 1.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("Xuannv China V1: 60,000-point provisional sampling plan", fontsize=15, pad=12)
    axis.legend(loc="lower left", fontsize=9, frameon=True, markerscale=8)
    axis.text(
        0.01, 0.015,
        "Static boundary-valid preview only; monthly source-quality and semantic-strata gates are pending.",
        transform=axis.transAxes, fontsize=9, color="#303030",
        bbox={"facecolor": "white", "edgecolor": "#808080", "pad": 4.0},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--base-points", type=Path, required=True)
    parser.add_argument("--country", type=Path, required=True)
    parser.add_argument("--adm1", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=60000)
    parser.add_argument("--points-output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    seed = int(policy["sampling"]["sampling_seed"])
    base = _read_jsonl(args.base_points)
    for record in base:
        record["sampling_tier"] = "base_spatial"
    inventory = _read_jsonl(args.inventory)
    country = gpd.read_file(args.country).to_crs("EPSG:4326").geometry.union_all()
    supplement, allocation = build_supplement(inventory, base, country, seed, args.target_total)
    all_points = base + supplement
    args.points_output.parent.mkdir(parents=True, exist_ok=True)
    args.points_output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_points), encoding="utf-8")
    render_map(args.country, args.adm1, base, supplement, args.map_output)
    summary = {
        "schema_version": "china_v1_spatial_supplement_v1",
        "base_points": len(base),
        "spatial_stratified_supplements": len(supplement),
        "total_points": len(all_points),
        "allocation_by_utm_grid": allocation,
        "semantic_supplement_status": "pending OSM and land-cover strata audit",
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
