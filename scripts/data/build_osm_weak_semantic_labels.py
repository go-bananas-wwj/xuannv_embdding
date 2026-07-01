#!/usr/bin/env python3
"""Rasterize broad OSM semantic groups into weak 128x128 training masks."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box


LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")
DEFAULT_CACHE_ROOT = Path("/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache")

ROAD_WIDTH_M = {
    "major": 18.0,
    "minor": 8.0,
    "path": 3.0,
    "rail": 8.0,
    "waterway": 12.0,
}
POINT_BUFFER_M = {
    "osm_activity_poi": 80.0,
    "osm_green": 25.0,
    "osm_water": 20.0,
    "default": 40.0,
}

TASK_RULES: dict[str, dict[str, Any]] = {
    "osm_building": {
        "themes": ["built"],
        "columns": {"building": "__any__"},
        "geometry": "polygon",
    },
    "osm_major_road": {
        "themes": ["transport"],
        "columns": {"highway": {"motorway", "trunk", "primary", "secondary", "tertiary"}},
        "geometry": "line",
        "buffer_m": ROAD_WIDTH_M["major"],
    },
    "osm_minor_road": {
        "themes": ["transport"],
        "columns": {"highway": {"residential", "unclassified", "service", "living_street"}},
        "geometry": "line",
        "buffer_m": ROAD_WIDTH_M["minor"],
    },
    "osm_path_walk": {
        "themes": ["transport"],
        "columns": {"highway": {"footway", "path", "cycleway", "steps", "pedestrian", "track"}},
        "geometry": "line",
        "buffer_m": ROAD_WIDTH_M["path"],
    },
    "osm_rail": {
        "themes": ["transport"],
        "columns": {"railway": {"rail", "subway", "light_rail", "tram", "station", "platform"}},
        "geometry": "line",
        "buffer_m": ROAD_WIDTH_M["rail"],
    },
    "osm_water": {
        "themes": ["natural"],
        "columns": {
            "natural": {"water", "wetland"},
            "water": "__any__",
            "waterway": {"river", "stream", "canal", "ditch", "drain", "riverbank"},
        },
        "geometry": "mixed",
        "buffer_m": ROAD_WIDTH_M["waterway"],
    },
    "osm_green": {
        "themes": ["landuse", "natural", "activity"],
        "columns": {
            "landuse": {"forest", "grass", "meadow", "recreation_ground", "allotments"},
            "natural": {"wood", "tree_row", "scrub", "grassland", "heath", "tree"},
            "leisure": {"park", "garden", "nature_reserve"},
        },
        "geometry": "mixed",
    },
    "osm_residential": {
        "themes": ["landuse"],
        "columns": {"landuse": {"residential"}},
        "geometry": "polygon",
    },
    "osm_commercial": {
        "themes": ["landuse", "activity"],
        "columns": {
            "landuse": {"commercial", "retail"},
            "shop": "__any__",
            "office": "__any__",
        },
        "geometry": "mixed",
    },
    "osm_industrial": {
        "themes": ["landuse", "built"],
        "columns": {
            "landuse": {"industrial", "brownfield", "quarry", "landfill"},
            "building": {"industrial", "warehouse", "factory"},
            "man_made": {"works", "storage_tank", "wastewater_plant", "silo", "chimney"},
        },
        "geometry": "mixed",
    },
    "osm_agriculture": {
        "themes": ["landuse"],
        "columns": {"landuse": {"farmland", "farmyard", "orchard", "vineyard", "greenhouse_horticulture"}},
        "geometry": "polygon",
    },
    "osm_construction": {
        "themes": ["landuse", "built", "transport"],
        "columns": {
            "landuse": {"construction", "brownfield"},
            "building": {"construction"},
            "highway": {"construction"},
            "railway": {"construction"},
            "construction": "__any__",
        },
        "geometry": "mixed",
    },
    "osm_activity_poi": {
        "themes": ["activity"],
        "columns": {
            "amenity": "__any__",
            "shop": "__any__",
            "office": "__any__",
            "tourism": "__any__",
            "leisure": {"park", "garden", "pitch", "sports_centre", "playground", "stadium"},
        },
        "geometry": "mixed",
    },
    "osm_playground": {
        "themes": ["activity", "landuse"],
        "columns": {
            "leisure": {"pitch", "track", "stadium", "sports_centre", "playground"},
            "landuse": {"recreation_ground"},
        },
        "geometry": "mixed",
        "point_buffer_m": 35.0,
    },
}


@dataclass(frozen=True)
class PatchRef:
    patch_id: str
    path: Path
    crs: Any
    transform: Any
    shape: tuple[int, int]
    bounds: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--regions", nargs="+", default=["haidian", "harbin"])
    parser.add_argument("--tasks", nargs="+", default=sorted(TASK_RULES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def patch_id_from_name(path: Path) -> str | None:
    match = re.search(r"(patch_\d{6})", path.stem)
    return match.group(1) if match else None


def choose_patch_refs(root: Path, limit: int | None = None) -> list[PatchRef]:
    by_patch: dict[str, Path] = {}
    for path in sorted(root.glob("*.tif")):
        if path.name.endswith("_mask.tif"):
            continue
        patch_id = patch_id_from_name(path)
        if patch_id is not None:
            by_patch[patch_id] = path

    refs: list[PatchRef] = []
    for patch_id, path in sorted(by_patch.items()):
        with rasterio.open(path) as src:
            refs.append(
                PatchRef(
                    patch_id=patch_id,
                    path=path,
                    crs=src.crs,
                    transform=src.transform,
                    shape=(src.height, src.width),
                    bounds=src.bounds,
                )
            )
        if limit is not None and len(refs) >= limit:
            break
    return refs


def load_theme(cache_root: Path, region: str, theme: str) -> gpd.GeoDataFrame:
    path = cache_root / f"{region}_{theme}.gpkg"
    if not path.exists():
        LOGGER.warning("missing OSM semantic cache: %s", path)
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def value_mask(gdf: gpd.GeoDataFrame, column: str, values: Any) -> np.ndarray:
    if column not in gdf.columns:
        return np.zeros(len(gdf), dtype=bool)
    series = gdf[column].astype("object")
    valid = series.notna() & (series.astype(str) != "") & (series.astype(str) != "nan")
    if values == "__any__":
        return valid.to_numpy()
    allowed = {str(value).lower() for value in values}
    return (valid & series.astype(str).str.lower().isin(allowed)).to_numpy()


def select_task_features(
    themes: dict[str, gpd.GeoDataFrame],
    task: str,
) -> gpd.GeoDataFrame:
    rule = TASK_RULES[task]
    selected = []
    for theme in rule["themes"]:
        gdf = themes.get(theme)
        if gdf is None or gdf.empty:
            continue
        keep = np.zeros(len(gdf), dtype=bool)
        for column, values in rule["columns"].items():
            keep |= value_mask(gdf, column, values)
        if keep.any():
            selected.append(gdf.loc[keep, ["geometry"]].copy())
    if not selected:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    merged = gpd.GeoDataFrame(
        {"geometry": list(gpd.GeoSeries(np.concatenate([df.geometry.to_numpy() for df in selected])))},
        geometry="geometry",
        crs=selected[0].crs,
    )
    return merged[merged.geometry.notna() & ~merged.geometry.is_empty].copy()


def prepare_geometries(
    gdf: gpd.GeoDataFrame,
    target_crs: Any,
    task: str,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)
    rule = TASK_RULES[task]
    target = gdf.to_crs(target_crs).copy()
    mode = rule.get("geometry", "mixed")
    buffer_m = float(rule.get("buffer_m", POINT_BUFFER_M.get(task, POINT_BUFFER_M["default"])))
    point_buffer_m = float(rule.get("point_buffer_m", POINT_BUFFER_M.get(task, POINT_BUFFER_M["default"])))

    out_geoms = []
    for geom in target.geometry:
        if geom is None or geom.is_empty:
            continue
        geom_type = geom.geom_type
        if geom_type in {"Polygon", "MultiPolygon"}:
            if mode in {"polygon", "mixed"}:
                out_geoms.append(geom)
        elif geom_type in {"LineString", "MultiLineString"}:
            if mode in {"line", "mixed"}:
                out_geoms.append(geom.buffer(buffer_m / 2.0, cap_style="round", join_style="round"))
        elif geom_type in {"Point", "MultiPoint"} and mode == "mixed":
            out_geoms.append(geom.buffer(point_buffer_m))
    return gpd.GeoDataFrame({"geometry": out_geoms}, geometry="geometry", crs=target_crs)


def rasterize_patch(geoms: gpd.GeoDataFrame, ref: PatchRef) -> np.ndarray:
    if geoms.empty:
        return np.zeros(ref.shape, dtype=np.uint8)
    patch_box = box(ref.bounds.left, ref.bounds.bottom, ref.bounds.right, ref.bounds.top)
    subset = geoms[geoms.intersects(patch_box)]
    if subset.empty:
        return np.zeros(ref.shape, dtype=np.uint8)
    shapes = ((geom, 1) for geom in subset.geometry if geom is not None and not geom.is_empty)
    return rasterize(
        shapes,
        out_shape=ref.shape,
        transform=ref.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )


def write_mask(path: Path, ref: PatchRef, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": ref.shape[0],
        "width": ref.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": ref.crs,
        "transform": ref.transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = np.array([record["positive_ratio"] for record in records], dtype=np.float64)
    positives = np.array([record["positive_pixels"] for record in records], dtype=np.int64)
    return {
        "num_patches": len(records),
        "positive_patches": int((positives > 0).sum()) if len(records) else 0,
        "positive_patch_ratio": float((positives > 0).mean()) if len(records) else 0.0,
        "positive_ratio_mean": float(ratios.mean()) if len(records) else 0.0,
        "positive_ratio_p95": float(np.percentile(ratios, 95)) if len(records) else 0.0,
        "positive_ratio_max": float(ratios.max()) if len(records) else 0.0,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def build_region(args: argparse.Namespace, region: str) -> dict[str, Any]:
    refs = choose_patch_refs(args.processed_root / region / "patches" / "s2", args.limit)
    if not refs:
        raise RuntimeError(f"no S2 patch refs found for {region}")
    themes = {
        theme: load_theme(args.cache_root, region, theme)
        for theme in sorted({theme for task in args.tasks for theme in TASK_RULES[task]["themes"]})
    }

    region_summary: dict[str, Any] = {"region": region, "tasks": {}}
    for task in args.tasks:
        output_root = args.processed_root / region / "labels" / task
        metadata_path = output_root / "metadata.json"
        if metadata_path.exists() and not args.overwrite:
            LOGGER.info("skip existing %s / %s", region, task)
            region_summary["tasks"][task] = json.loads(metadata_path.read_text(encoding="utf-8"))
            continue

        raw = select_task_features(themes, task)
        geoms = prepare_geometries(raw, refs[0].crs, task)
        records = []
        masks_dir = output_root / "masks"
        for idx, ref in enumerate(refs, start=1):
            mask = rasterize_patch(geoms, ref)
            mask_path = masks_dir / f"{ref.patch_id}.tif"
            write_mask(mask_path, ref, mask)
            positive = int((mask > 0).sum())
            records.append(
                {
                    "patch_id": ref.patch_id,
                    "mask": str(mask_path),
                    "positive_pixels": positive,
                    "positive_ratio": positive / float(mask.size),
                    "ref": str(ref.path),
                }
            )
            if idx % 100 == 0:
                LOGGER.info("%s / %s wrote %d/%d", region, task, idx, len(refs))

        metadata = {
            "region": region,
            "task": task,
            "label_kind": "osm_weak_semantic",
            "source_cache_root": str(args.cache_root),
            "raw_feature_count": int(len(raw)),
            "prepared_geometry_count": int(len(geoms)),
            "summary": summarize(records),
            "records": records,
            "rule": json_ready(TASK_RULES[task]),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        region_summary["tasks"][task] = metadata
        LOGGER.info("%s / %s summary: %s", region, task, metadata["summary"])
    return region_summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    invalid = sorted(set(args.tasks) - set(TASK_RULES))
    if invalid:
        raise ValueError(f"unknown tasks: {invalid}")
    summaries = [build_region(args, region) for region in args.regions]
    summary_path = args.processed_root / "osm_weak_semantic_summary_20260629.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
