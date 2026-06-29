#!/usr/bin/env python3
"""Audit OSM weak-label caches before using them for embedding training.

This script is intentionally conservative: it reports what the current OSM
cache can and cannot support, especially whether time-related tags are present.
It does not rewrite labels or treat OSM blanks as negatives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio


DEFAULT_LABEL_ROOT = Path("/data/xuannv_embedding/processed")
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/experiments/p3a_osm_audit_20260629")
DEFAULT_DOC_COPY = Path("docs/experiments/p3a_osm_weak_label_audit_20260629.md")

OSM_TIME_COLUMNS = [
    "start_date",
    "opening_date",
    "construction_date",
    "check_date",
    "survey:date",
    "source:date",
    "end_date",
    "date",
]

LIFECYCLE_PREFIXES = (
    "construction:",
    "proposed:",
    "planned:",
    "disused:",
    "abandoned:",
    "demolished:",
    "removed:",
    "razed:",
    "was:",
)

SEMANTIC_COLUMNS = [
    "building",
    "building:use",
    "construction",
    "highway",
    "waterway",
    "natural",
    "water",
    "landuse",
    "amenity",
    "man_made",
]


@dataclass(frozen=True)
class CacheSpec:
    region: str
    task: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-copy", type=Path, default=DEFAULT_DOC_COPY)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["haidian", "harbin"],
        help="Regions under label-root to audit.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["building_osm", "road_osm"],
        help="OSM label tasks under processed/{region}/labels/.",
    )
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def non_null_mask(series: gpd.GeoSeries) -> Any:
    values = series.astype("object")
    return values.notna() & (values.astype(str) != "") & (values.astype(str) != "nan")


def value_counts(gdf: gpd.GeoDataFrame, column: str, top_k: int) -> list[dict[str, Any]]:
    if column not in gdf.columns:
        return []
    values = gdf.loc[non_null_mask(gdf[column]), column].astype(str)
    counts = values.value_counts().head(top_k)
    return [{"value": str(idx), "count": int(count)} for idx, count in counts.items()]


def estimate_metric_summary(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    if gdf.empty:
        return {"area_km2": 0.0, "length_km": 0.0}
    try:
        metric_crs = gdf.estimate_utm_crs()
    except Exception:
        metric_crs = None
    if metric_crs is None:
        metric_crs = "EPSG:3857"
    metric = gdf.to_crs(metric_crs)
    geom_types = metric.geometry.geom_type
    area = float(metric.loc[geom_types.isin(["Polygon", "MultiPolygon"])].geometry.area.sum())
    length = float(
        metric.loc[geom_types.isin(["LineString", "MultiLineString"])].geometry.length.sum()
    )
    return {"area_km2": area / 1_000_000.0, "length_km": length / 1_000.0}


def audit_masks(task_root: Path) -> dict[str, Any]:
    mask_dir = task_root / "masks"
    mask_paths = sorted(mask_dir.glob("*.tif"))
    if not mask_paths:
        return {
            "num_masks": 0,
            "positive_masks": 0,
            "positive_ratio_mean": 0.0,
            "positive_ratio_median": 0.0,
            "positive_ratio_max": 0.0,
            "positive_ratio_p95": 0.0,
        }

    ratios: list[float] = []
    positives = 0
    for path in mask_paths:
        with rasterio.open(path) as src:
            arr = src.read(1)
        ratio = float((arr > 0).sum()) / float(arr.size)
        ratios.append(ratio)
        positives += int(ratio > 0)
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    return {
        "num_masks": len(mask_paths),
        "positive_masks": positives,
        "positive_patch_ratio": positives / float(len(mask_paths)),
        "positive_ratio_mean": float(ratio_arr.mean()),
        "positive_ratio_median": float(np.median(ratio_arr)),
        "positive_ratio_max": float(ratio_arr.max()),
        "positive_ratio_p95": float(np.percentile(ratio_arr, 95)),
    }


def audit_cache(spec: CacheSpec, top_k: int) -> dict[str, Any]:
    if not spec.path.exists():
        return {
            "region": spec.region,
            "task": spec.task,
            "path": str(spec.path),
            "exists": False,
            "error": "missing osm_cache.gpkg",
        }

    gdf = gpd.read_file(spec.path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()

    time_columns = [col for col in OSM_TIME_COLUMNS if col in gdf.columns]
    lifecycle_columns = [
        col
        for col in gdf.columns
        if any(col.startswith(prefix) for prefix in LIFECYCLE_PREFIXES)
    ]
    temporal_columns = time_columns + lifecycle_columns

    temporal_summary: dict[str, Any] = {}
    for col in temporal_columns:
        mask = non_null_mask(gdf[col])
        temporal_summary[col] = {
            "non_null": int(mask.sum()),
            "ratio": float(mask.mean()) if len(mask) else 0.0,
            "examples": [str(v) for v in gdf.loc[mask, col].astype(str).head(8).tolist()],
        }

    semantic_summary = {
        col: {
            "non_null": int(non_null_mask(gdf[col]).sum()),
            "top_values": value_counts(gdf, col, top_k),
        }
        for col in SEMANTIC_COLUMNS
        if col in gdf.columns
    }

    geom_counts = Counter(gdf.geometry.geom_type.astype(str))
    return {
        "region": spec.region,
        "task": spec.task,
        "path": str(spec.path),
        "exists": True,
        "num_features": int(len(gdf)),
        "num_columns": int(len(gdf.columns)),
        "crs": str(gdf.crs),
        "geometry_types": dict(sorted(geom_counts.items())),
        "metric_summary": estimate_metric_summary(gdf),
        "time_columns_present": time_columns,
        "lifecycle_columns_present": lifecycle_columns,
        "temporal_summary": temporal_summary,
        "semantic_summary": semantic_summary,
        "mask_summary": audit_masks(spec.path.parent),
    }


def discover_specs(label_root: Path, regions: list[str], tasks: list[str]) -> list[CacheSpec]:
    specs = []
    for region in regions:
        for task in tasks:
            specs.append(
                CacheSpec(
                    region=region,
                    task=task,
                    path=label_root / region / "labels" / task / "osm_cache.gpkg",
                )
            )
    return specs


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(results: list[dict[str, Any]], output_root: Path) -> str:
    lines = [
        "# P3A.0 OSM Weak Label Audit",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Output root: `{output_root}`",
        "- Scope: current OSM caches and raster masks for Haidian/Harbin building and road labels.",
        "- Rule: these labels are weak auxiliary signals only; OSM blanks are not strong negatives.",
        "",
        "## Executive Summary",
        "",
    ]

    any_time = False
    for item in results:
        for stat in item.get("temporal_summary", {}).values():
            if stat.get("non_null", 0) > 0:
                any_time = True
                break
    if any_time:
        lines.append(
            "Some current caches contain sparse OSM time tags. They can provide confidence hints, "
            "but not month-level ground truth change labels."
        )
    else:
        lines.append(
            "The current caches do not contain useful month-level OSM history. They are suitable "
            "for static/semi-static weak semantic priors, not for supervised monthly change."
        )

    lines.extend(
        [
            "",
            "## Cache and Mask Coverage",
            "",
            "| region | task | features | geometry | time columns with values | masks | positive patches | mean positive pixels | p95 positive pixels |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        if not item.get("exists"):
            lines.append(
                f"| {item['region']} | {item['task']} | missing | - | - | 0 | 0 | 0.00% | 0.00% |"
            )
            continue
        time_cols = [
            col
            for col, stat in item.get("temporal_summary", {}).items()
            if stat.get("non_null", 0) > 0
        ]
        mask = item["mask_summary"]
        geom = ", ".join(f"{k}:{v}" for k, v in item["geometry_types"].items())
        lines.append(
            "| {region} | {task} | {features} | {geom} | {time_cols} | {masks} | "
            "{positive} ({positive_ratio}) | {mean} | {p95} |".format(
                region=item["region"],
                task=item["task"],
                features=item["num_features"],
                geom=geom or "-",
                time_cols=", ".join(time_cols) if time_cols else "-",
                masks=mask["num_masks"],
                positive=mask["positive_masks"],
                positive_ratio=pct(mask.get("positive_patch_ratio", 0.0)),
                mean=pct(mask["positive_ratio_mean"]),
                p95=pct(mask["positive_ratio_p95"]),
            )
        )

    lines.extend(
        [
            "",
            "## Temporal Tag Detail",
            "",
        ]
    )
    for item in results:
        lines.append(f"### {item['region']} / {item['task']}")
        if not item.get("exists"):
            lines.extend(["", f"- Missing: `{item['path']}`", ""])
            continue
        temporal = item.get("temporal_summary", {})
        if not temporal:
            lines.extend(["", "- No configured time/lifecycle columns are present.", ""])
            continue
        lines.extend(["", "| column | non-null | ratio | examples |", "|---|---:|---:|---|"])
        for col, stat in temporal.items():
            examples = ", ".join(stat.get("examples", [])) or "-"
            lines.append(f"| `{col}` | {stat['non_null']} | {pct(stat['ratio'])} | {examples} |")
        lines.append("")

    lines.extend(
        [
            "## Semantic Tag Detail",
            "",
        ]
    )
    for item in results:
        if not item.get("exists"):
            continue
        lines.append(f"### {item['region']} / {item['task']}")
        semantic = item.get("semantic_summary", {})
        if not semantic:
            lines.extend(["", "- No configured semantic columns are present.", ""])
            continue
        for col, stat in semantic.items():
            top = ", ".join(f"{v['value']}:{v['count']}" for v in stat["top_values"]) or "-"
            lines.append(f"- `{col}` non-null {stat['non_null']}: {top}")
        lines.append("")

    lines.extend(
        [
            "## Training Decision",
            "",
            "1. Use current OSM building/road only as weak density or presence priors.",
            "2. Do not use current OSM as monthly change labels.",
            "3. Do not treat empty OSM areas as hard background negatives.",
            "4. For P3A monthly training, prefer density targets and confidence masks over hard masks.",
            "5. To use true OSM time information, download full-history `.osh.pbf` or monthly snapshots, "
            "then rerun this audit on history-derived GeoPackage layers.",
            "",
            "## Next Step",
            "",
            "Build P3A.1 index-reconstruction targets first, while preparing an OSM-history ingestion "
            "path in parallel. This avoids blocking the embedding upgrade on sparse OSM history.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    specs = discover_specs(args.label_root, args.regions, args.tasks)
    results = [audit_cache(spec, args.top_k) for spec in specs]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_root": str(args.label_root),
        "results": results,
    }
    json_path = args.output_root / "osm_weak_label_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_markdown(results, args.output_root)
    md_path = args.output_root / "osm_weak_label_audit.md"
    md_path.write_text(markdown, encoding="utf-8")
    if args.doc_copy:
        args.doc_copy.parent.mkdir(parents=True, exist_ok=True)
        args.doc_copy.write_text(markdown, encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.doc_copy:
        print(f"wrote {args.doc_copy}")


if __name__ == "__main__":
    main()
