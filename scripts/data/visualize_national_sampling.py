#!/usr/bin/env python3
"""Create an auditable static 1% sampling preview and a national map.

This preview uses only the China boundary and macrocell inventory.  Points are
inside the national boundary but have *not* passed monthly source-quality
gates, so it must not be used as the final training registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATCH_SIDE_METERS = 1280
MACRO_SIDE_PATCHES = 10


def _unit_hash(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _candidate_order(seed: int, macro_id: str) -> list[tuple[int, int]]:
    candidates = [(col, row) for row in range(MACRO_SIDE_PATCHES) for col in range(MACRO_SIDE_PATCHES)]
    return sorted(candidates, key=lambda pair: _unit_hash(seed, f"{macro_id}:{pair[0]}:{pair[1]}"))


def select_static_preview(
    macrocells: list[dict[str, Any]],
    country_geometry: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Select boundary-valid points at the expected 1% static coverage rate."""
    selected: list[dict[str, Any]] = []
    transformers: dict[int, Transformer] = {}
    for macro in macrocells:
        estimated_count = float(macro["estimated_patch_count"])
        if _unit_hash(seed, f"preview-macro:{macro['macro_id']}") >= min(1.0, estimated_count / 100.0):
            continue
        epsg = int(macro["grid_epsg"])
        transformer = transformers.setdefault(epsg, Transformer.from_crs(epsg, 4326, always_xy=True))
        left, bottom, _, _ = (float(value) for value in macro["utm_bounds"])
        point_record: dict[str, Any] | None = None
        for patch_col, patch_row in _candidate_order(seed, str(macro["macro_id"])):
            x = left + (patch_col + 0.5) * PATCH_SIDE_METERS
            y = bottom + (patch_row + 0.5) * PATCH_SIDE_METERS
            lon, lat = transformer.transform(x, y)
            if not country_geometry.covers(Point(lon, lat)):
                continue
            point_record = {
                "schema_version": "china_v1_static_sampling_preview_v1",
                "status": "provisional_static_land_point_not_quality_eligible",
                "patch_id": f"preview_{macro['macro_id']}_c{patch_col}_r{patch_row}",
                "macro_id": macro["macro_id"],
                "grid_id": macro["grid_id"],
                "grid_epsg": epsg,
                "grid_col": int(macro["macro_col"]) * MACRO_SIDE_PATCHES + patch_col,
                "grid_row": int(macro["macro_row"]) * MACRO_SIDE_PATCHES + patch_row,
                "longitude": round(lon, 7),
                "latitude": round(lat, 7),
                "admin1": macro["admin1"],
                "sampling_seed": seed,
            }
            break
        if point_record is not None:
            selected.append(point_record)
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_map(country_path: Path, adm1_path: Path, points: list[dict[str, Any]], output: Path) -> None:
    country = gpd.read_file(country_path).to_crs("EPSG:4326")
    adm1 = gpd.read_file(adm1_path).to_crs("EPSG:4326")
    point_frame = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy([point["longitude"] for point in points], [point["latitude"] for point in points]),
        crs="EPSG:4326",
    )
    fig, axis = plt.subplots(figsize=(15, 10), dpi=220)
    country.plot(ax=axis, color="#f4f1e8", edgecolor="#303030", linewidth=0.7, zorder=1)
    adm1.boundary.plot(ax=axis, color="#9a9a9a", linewidth=0.24, zorder=2)
    point_frame.plot(ax=axis, color="#c5221f", markersize=0.48, alpha=0.72, zorder=3)
    min_x, min_y, max_x, max_y = country.total_bounds
    axis.set_xlim(min_x - 1.5, max_x + 1.5)
    axis.set_ylim(min_y - 1.5, max_y + 1.5)
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title("Xuannv China V1: provisional static 1% sampling preview", fontsize=15, pad=12)
    axis.text(
        0.01,
        0.015,
        f"{len(points):,} boundary-valid 1280 m patch centers; monthly quality filtering not applied",
        transform=axis.transAxes,
        fontsize=9,
        color="#303030",
        bbox={"facecolor": "white", "edgecolor": "#808080", "pad": 4.0},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--country", type=Path, required=True)
    parser.add_argument("--adm1", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--points-output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    seed = int(policy["sampling"]["sampling_seed"])
    country = gpd.read_file(args.country).to_crs("EPSG:4326").geometry.union_all()
    points = select_static_preview(_read_jsonl(args.inventory), country, seed)
    args.points_output.parent.mkdir(parents=True, exist_ok=True)
    args.points_output.write_text("".join(json.dumps(point, ensure_ascii=False) + "\n" for point in points), encoding="utf-8")
    render_map(args.country, args.adm1, points, args.map_output)
    print(json.dumps({"points": len(points), "points_output": str(args.points_output), "map_output": str(args.map_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
