#!/usr/bin/env python3
"""Build selected nationwide 1,280 m UTM grid partitions without materializing all records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
from china_full_grid import GridSpec, enumerate_macro_patch_records, write_zone_records


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sampled_keys(path: Path) -> set[str]:
    data = _read_json(path)
    if isinstance(data, dict):
        return set(data)
    if isinstance(data, list) and all(isinstance(key, str) for key in data):
        return set(data)
    raise ValueError("sampled parent-key JSON must be an object map or a list of strings")


def _macros(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list) or not all(isinstance(macro, dict) for macro in data):
        raise ValueError("macrocells JSON must be a list of macrocell objects")
    return data


def _zone_records(
    macros: Iterable[dict[str, Any]], boundary_wgs84: Any, spec: GridSpec, grid_id: str
):
    for macro in macros:
        if str(macro["grid_id"]) == grid_id:
            yield from enumerate_macro_patch_records(macro, boundary_wgs84, spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, required=True, help="WGS84 boundary vector file")
    parser.add_argument("--macrocells", type=Path, required=True, help="Task 1 macrocell JSON")
    parser.add_argument(
        "--sampled-parent-keys", type=Path, required=True, help="62k sampled key map JSON"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid-id", required=True, help="One UTM grid_id to write")
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    boundary = gpd.read_file(args.boundary).to_crs(4326).geometry.unary_union
    macros = _macros(args.macrocells)
    summary = write_zone_records(
        _zone_records(macros, boundary, GridSpec(), args.grid_id),
        _sampled_keys(args.sampled_parent_keys),
        args.output_root,
        args.batch_size,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, StopIteration) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
