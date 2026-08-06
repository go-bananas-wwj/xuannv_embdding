#!/usr/bin/env python3
"""Audit expected narrow overlaps between adjacent nationwide UTM parent grids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from china_full_grid import assess_utm_seam_overlap_policy, reconcile_utm_seam_audit
from pyproj import Transformer
from shapely import from_wkb
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree

SEAM_TOLERANCE_DEGREES = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--seam-audit-output", type=Path, required=True)
    parser.add_argument("--final-audit-output", type=Path, required=True)
    return parser.parse_args()


def _seam_candidates(output_root: Path) -> tuple[int, list[dict[str, object]]]:
    total = 0
    candidates: list[dict[str, object]] = []
    columns = ["parent_key", "grid_epsg", "longitude", "latitude", "geometry"]
    for path in sorted((output_root / "all").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
            for row in batch.to_pylist():
                total += 1
                zone = int(row["grid_epsg"]) - 32600
                geometry = from_wkb(row["geometry"])
                minx, _, maxx, _ = geometry.bounds
                west = -180 + (zone - 1) * 6
                east = west + 6
                if minx <= west <= maxx or minx <= east <= maxx:
                    candidates.append(
                        {
                            "parent_key": str(row["parent_key"]),
                            "zone": zone,
                            "longitude": float(row["longitude"]),
                            "latitude": float(row["latitude"]),
                            "geometry": geometry,
                        }
                    )
    if total == 0:
        raise ValueError(f"no canonical all-partition records below {output_root}")
    return total, candidates


def audit_utm_seams(output_root: Path) -> dict[str, object]:
    total, candidates = _seam_candidates(output_root)
    geometries = [row["geometry"] for row in candidates]
    tree = STRtree(geometries)
    to_equal_area = Transformer.from_crs(4326, 6933, always_xy=True)
    projected = [transform_geometry(to_equal_area.transform, geometry) for geometry in geometries]

    overlap_pair_count = 0
    overlap_area_m2 = 0.0
    non_adjacent_pair_count = 0
    owner_order_mismatch_count = 0
    off_seam_pair_count = 0
    max_pair_overlap_fraction = 0.0

    for left_index, left in enumerate(candidates):
        left_geometry = geometries[left_index]
        left_projected = projected[left_index]
        for right_index_value in tree.query(left_geometry, predicate="intersects"):
            right_index = int(right_index_value)
            if right_index <= left_index:
                continue
            right = candidates[right_index]
            left_zone = int(left["zone"])
            right_zone = int(right["zone"])
            if left_zone == right_zone:
                continue
            right_projected = projected[right_index]
            overlap = left_projected.intersection(right_projected)
            area = float(overlap.area)
            if area <= 0:
                continue

            overlap_pair_count += 1
            overlap_area_m2 += area
            max_pair_overlap_fraction = max(
                max_pair_overlap_fraction,
                area / min(float(left_projected.area), float(right_projected.area)),
            )
            low, high = (left, right) if left_zone < right_zone else (right, left)
            low_zone = int(low["zone"])
            high_zone = int(high["zone"])
            if high_zone - low_zone != 1:
                non_adjacent_pair_count += 1
                continue
            seam_longitude = -180 + low_zone * 6
            if not (float(low["longitude"]) < seam_longitude <= float(high["longitude"])):
                owner_order_mismatch_count += 1
            intersection_wgs84 = left_geometry.intersection(geometries[right_index])
            if abs(float(intersection_wgs84.centroid.x) - seam_longitude) > SEAM_TOLERANCE_DEGREES:
                off_seam_pair_count += 1

    audit = assess_utm_seam_overlap_policy(
        overlap_pair_count=overlap_pair_count,
        overlap_area_m2=overlap_area_m2,
        total_parent_count=total,
        non_adjacent_pair_count=non_adjacent_pair_count,
        owner_order_mismatch_count=owner_order_mismatch_count,
        off_seam_pair_count=off_seam_pair_count,
        max_pair_overlap_fraction=max_pair_overlap_fraction,
    )
    audit["seam_candidate_count"] = len(candidates)
    audit["seam_tolerance_degrees"] = SEAM_TOLERANCE_DEGREES
    return audit


def main() -> int:
    args = parse_args()
    base_audit = json.loads(args.base_audit.read_text(encoding="utf-8"))
    seam_audit = audit_utm_seams(args.output_root)
    final_audit = reconcile_utm_seam_audit(base_audit, seam_audit)
    args.seam_audit_output.write_text(
        json.dumps(seam_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.final_audit_output.write_text(
        json.dumps(final_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(seam_audit, ensure_ascii=False, sort_keys=True))
    return 0 if seam_audit["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
