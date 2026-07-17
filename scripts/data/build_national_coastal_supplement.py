#!/usr/bin/env python3
"""Add a small, explicitly ocean-adjacent supplement to China V1 sampling."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import transform as transform_geometry
from shapely.prepared import prep


SUPPLEMENT_PATH = Path(__file__).with_name("build_national_spatial_supplement.py")
SPEC = importlib.util.spec_from_file_location("national_spatial_supplement", SUPPLEMENT_PATH)
assert SPEC is not None and SPEC.loader is not None
SPATIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPATIAL)
PREVIEW = SPATIAL.PREVIEW


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def macro_wgs84(macro: dict[str, Any]) -> Any:
    left, bottom, right, top = (float(value) for value in macro["utm_bounds"])
    transformer = Transformer.from_crs(int(macro["grid_epsg"]), 4326, always_xy=True)
    return transform_geometry(transformer.transform, box(left, bottom, right, top))


def _coastal_candidates(inventory: list[dict[str, Any]], ocean: Any) -> list[dict[str, Any]]:
    prepared_ocean = prep(ocean)
    # A fully inland macro cannot intersect the ocean. This coarse filter keeps
    # the exact geometry checks inexpensive for a national inventory.
    return [
        macro for macro in inventory
        if float(macro["land_fraction"]) < 0.999 and prepared_ocean.intersects(macro_wgs84(macro))
    ]


def select_coastal_supplement(
    inventory: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    country: Any,
    ocean: Any,
    seed: int,
    target: int,
    max_coast_distance_degrees: float = 0.035,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates = _coastal_candidates(inventory, ocean)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for macro in candidates:
        groups[str(macro["grid_id"])].append(macro)
    allocation = SPATIAL._allocate(target, groups)
    existing_ids = {str(record["patch_id"]) for record in existing}
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()

    def try_macro(macro: dict[str, Any]) -> dict[str, Any] | None:
        for offset in range(1, 101):
            point = PREVIEW.select_static_preview(
                [{**macro, "estimated_patch_count": 100.0}], country, seed + offset
            )
            if not point or point[0]["patch_id"] in existing_ids:
                continue
            record = point[0]
            if ocean.distance(Point(record["longitude"], record["latitude"])) > max_coast_distance_degrees:
                continue
            record["status"] = "provisional_ocean_adjacent_supplement_not_quality_eligible"
            record["sampling_tier"] = "coastal_ocean_adjacent_supplement"
            record["sampling_seed"] = seed
            record["candidate_rank_seed_offset"] = offset
            record["coast_distance_degree_upper_bound"] = max_coast_distance_degrees
            return record
        return None

    for group, quota in sorted(allocation.items()):
        for macro in sorted(groups[group], key=lambda item: PREVIEW._unit_hash(seed, f"coast:{item['macro_id']}")):
            if selected_by_group[group] >= quota:
                break
            record = try_macro(macro)
            if record is None:
                continue
            selected.append(record)
            existing_ids.add(str(record["patch_id"]))
            selected_by_group[group] += 1

    if len(selected) < target:
        for macro in sorted(candidates, key=lambda item: PREVIEW._unit_hash(seed, f"coast-fallback:{item['macro_id']}")):
            if len(selected) == target:
                break
            record = try_macro(macro)
            if record is None:
                continue
            selected.append(record)
            existing_ids.add(str(record["patch_id"]))
            selected_by_group[str(record["grid_id"])] += 1
    if len(selected) != target:
        raise RuntimeError(f"selected {len(selected)} ocean-adjacent points, expected {target}")
    return selected, dict(sorted(selected_by_group.items()))


def render_map(country_path: Path, adm1_path: Path, existing: list[dict[str, Any]], coastal: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    country = gpd.read_file(country_path).to_crs("EPSG:4326")
    admin1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
    base = [record for record in existing if record.get("sampling_tier") == "base_spatial"]
    spatial = [record for record in existing if record.get("sampling_tier") != "base_spatial"]
    fig, axis = plt.subplots(figsize=(15, 10), dpi=220)
    country.plot(ax=axis, color="#f4f1e8", edgecolor="#303030", linewidth=0.7, zorder=1)
    admin1.boundary.plot(ax=axis, color="#9a9a9a", linewidth=0.24, zorder=2)
    for records, color, label, size, zorder in (
        (base, "#c5221f", f"Base spatial ({len(base):,})", 0.32, 3),
        (spatial, "#1464a0", f"Spatial supplement ({len(spatial):,})", 0.42, 4),
        (coastal, "#6b2f8a", f"Ocean-adjacent supplement ({len(coastal):,})", 2.6, 5),
    ):
        axis.scatter([item["longitude"] for item in records], [item["latitude"] for item in records], s=size, c=color, alpha=0.78, label=label, linewidths=0, zorder=zorder)
    min_x, min_y, max_x, max_y = country.total_bounds
    axis.set_xlim(min_x - 1.5, max_x + 1.5)
    axis.set_ylim(min_y - 1.5, max_y + 1.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("Xuannv China V1: 60,500-point plan with ocean-adjacent supplements", fontsize=15, pad=12)
    axis.legend(loc="upper left", fontsize=9, frameon=True, markerscale=6)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--existing-points", type=Path, required=True)
    parser.add_argument("--country", type=Path, required=True)
    parser.add_argument("--adm1", type=Path, required=True)
    parser.add_argument("--ocean", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--coastal-count", type=int, default=500)
    parser.add_argument("--points-output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    seed = int(policy["sampling"]["sampling_seed"])
    inventory = _read_jsonl(args.inventory)
    existing = _read_jsonl(args.existing_points)
    country = gpd.read_file(args.country).to_crs("EPSG:4326").geometry.union_all()
    ocean = gpd.read_file(args.ocean).to_crs("EPSG:4326").geometry.union_all()
    coastal, by_grid = select_coastal_supplement(inventory, existing, country, ocean, seed, args.coastal_count)
    all_points = existing + coastal
    args.points_output.parent.mkdir(parents=True, exist_ok=True)
    args.points_output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_points), encoding="utf-8")
    render_map(args.country, args.adm1, existing, coastal, args.map_output)
    summary = {
        "schema_version": "china_v1_coastal_supplement_v1",
        "existing_points": len(existing), "ocean_adjacent_supplements": len(coastal), "total_points": len(all_points),
        "coastal_by_utm_grid": by_grid, "ocean_source": str(args.ocean), "max_coast_distance_degrees": 0.035,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
