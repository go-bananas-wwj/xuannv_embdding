#!/usr/bin/env python3
"""Build OSM-derived building and road masks for downstream tasks.

The script downloads OSM features for a region AOI, rasterizes them to the
existing 128x128 low-resolution patch grid for training, and optionally also
rasterizes to the high-resolution optical patch grid for visual QA.
"""

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
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, box, shape

LOGGER = logging.getLogger(__name__)

REGION_AOI = {
    "haidian": Path("configs/regions/haidian.geojson"),
    "harbin": Path("configs/regions/harbin.geojson"),
}

ROAD_WIDTH_M = {
    "motorway": 24.0,
    "trunk": 22.0,
    "primary": 18.0,
    "secondary": 14.0,
    "tertiary": 12.0,
    "residential": 8.0,
    "unclassified": 8.0,
    "service": 5.0,
    "living_street": 5.0,
    "footway": 2.0,
    "path": 2.0,
    "cycleway": 2.0,
    "track": 4.0,
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
    parser.add_argument("--region", choices=sorted(REGION_AOI), required=True)
    parser.add_argument(
        "--task",
        choices=("building_osm", "road_osm"),
        required=True,
        help="Which OSM label task to build.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed"),
    )
    parser.add_argument("--aoi", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to processed/{region}/labels/{task}.",
    )
    parser.add_argument(
        "--osm-cache",
        type=Path,
        default=None,
        help="GeoPackage cache path. Defaults under output-root/osm_cache.gpkg.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--include-minor-roads",
        action="store_true",
        help="Keep footway/path/cycleway/service roads for road_osm.",
    )
    parser.add_argument(
        "--highres",
        action="store_true",
        help="Also write high-resolution masks to masks_highres/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N patch ids, useful for smoke tests.",
    )
    parser.add_argument(
        "--patch-ids",
        nargs="+",
        default=None,
        help="Optional explicit patch ids to process, e.g. patch_000198.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_aoi_polygon(path: Path) -> Polygon:
    data = json.loads(path.read_text(encoding="utf-8"))
    geom = data["features"][0]["geometry"]
    polygon = shape(geom)
    if not isinstance(polygon, Polygon):
        polygon = polygon.convex_hull
    return polygon


def tags_for_task(task: str) -> dict[str, Any]:
    if task == "building_osm":
        return {"building": True}
    if task == "road_osm":
        return {"highway": True}
    raise ValueError(f"unknown task: {task}")


def read_or_download_osm(
    task: str,
    aoi_polygon: Polygon,
    cache_path: Path,
    overwrite: bool,
) -> gpd.GeoDataFrame:
    if cache_path.exists() and not overwrite:
        LOGGER.info("reading OSM cache: %s", cache_path)
        return gpd.read_file(cache_path)

    LOGGER.info("downloading OSM features for %s", task)
    gdf = ox.features_from_polygon(aoi_polygon, tags_for_task(task))
    if gdf.empty:
        LOGGER.warning("OSM query returned no features for %s", task)
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    else:
        gdf = gdf.reset_index()
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf.set_geometry("geometry")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GPKG")
    LOGGER.info("cached %d OSM features to %s", len(gdf), cache_path)
    return gdf


def patch_id_from_name(path: Path) -> str | None:
    match = re.search(r"(patch_\d{6})", path.stem)
    if match is None:
        return None
    return match.group(1)


def choose_patch_refs(
    root: Path,
    limit: int | None = None,
    patch_ids: list[str] | None = None,
) -> list[PatchRef]:
    by_patch: dict[str, Path] = {}
    for path in sorted(root.glob("*.tif")):
        if path.name.endswith("_mask.tif"):
            continue
        patch_id = patch_id_from_name(path)
        if patch_id is None:
            continue
        # Prefer later files so current-year imagery wins, but any month has the
        # same grid for a patch.
        by_patch[patch_id] = path

    refs: list[PatchRef] = []
    keep = set(patch_ids) if patch_ids else None
    for patch_id, path in sorted(by_patch.items()):
        if keep is not None and patch_id not in keep:
            continue
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


def road_width(value: Any, include_minor_roads: bool) -> float | None:
    if isinstance(value, list):
        values = [str(v) for v in value]
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        values = []
    else:
        values = [str(value)]

    if not values:
        return ROAD_WIDTH_M["unclassified"]

    widths = []
    for highway in values:
        if not include_minor_roads and highway in {"footway", "path", "cycleway", "service"}:
            continue
        widths.append(ROAD_WIDTH_M.get(highway, ROAD_WIDTH_M["unclassified"]))
    if not widths:
        return None
    return max(widths)


def prepare_geometries(
    gdf: gpd.GeoDataFrame,
    task: str,
    target_crs: Any,
    include_minor_roads: bool,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)

    target = gdf.to_crs(target_crs).copy()
    target = target[target.geometry.notna() & ~target.geometry.is_empty].copy()

    if task == "building_osm":
        target = target[target.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        return target[["geometry"]]

    buffered = []
    for _, row in target.iterrows():
        width = road_width(row.get("highway"), include_minor_roads)
        if width is None:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        buffered.append(geom.buffer(width / 2.0, cap_style="round", join_style="round"))
    return gpd.GeoDataFrame({"geometry": buffered}, geometry="geometry", crs=target_crs)


def rasterize_patch(
    geoms: gpd.GeoDataFrame,
    ref: PatchRef,
) -> np.ndarray:
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
    positives = np.array([r["positive_pixels"] for r in records], dtype=np.float64)
    ratios = np.array([r["positive_ratio"] for r in records], dtype=np.float64)
    return {
        "num_patches": len(records),
        "num_positive_patches": int((positives > 0).sum()) if len(records) else 0,
        "positive_pixels_total": int(positives.sum()) if len(records) else 0,
        "positive_ratio_mean": float(ratios.mean()) if len(records) else 0.0,
        "positive_ratio_max": float(ratios.max()) if len(records) else 0.0,
        "positive_ratio_min": float(ratios.min()) if len(records) else 0.0,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_root = args.output_root or args.processed_root / args.region / "labels" / args.task
    osm_cache = args.osm_cache or output_root / "osm_cache.gpkg"
    aoi_path = args.aoi or REGION_AOI[args.region]
    aoi_polygon = load_aoi_polygon(aoi_path)

    lowres_root = args.processed_root / args.region / "patches" / "s2"
    highres_root = args.processed_root / args.region / "patches" / "highres_optical"
    lowres_refs = choose_patch_refs(lowres_root, args.limit, args.patch_ids)
    if not lowres_refs:
        raise RuntimeError(f"no low-resolution patch refs found under {lowres_root}")
    highres_refs = (
        choose_patch_refs(highres_root, args.limit, args.patch_ids) if args.highres else []
    )

    osm = read_or_download_osm(args.task, aoi_polygon, osm_cache, args.overwrite_cache)
    lowres_geoms = prepare_geometries(
        osm,
        args.task,
        lowres_refs[0].crs,
        args.include_minor_roads,
    )
    highres_geoms = (
        prepare_geometries(osm, args.task, highres_refs[0].crs, args.include_minor_roads)
        if highres_refs
        else None
    )

    records = []
    masks_dir = output_root / "masks"
    for idx, ref in enumerate(lowres_refs, start=1):
        mask = rasterize_patch(lowres_geoms, ref)
        write_mask(masks_dir / f"{ref.patch_id}.tif", ref, mask)
        positive = int(mask.sum())
        records.append(
            {
                "patch_id": ref.patch_id,
                "mask": str(masks_dir / f"{ref.patch_id}.tif"),
                "positive_pixels": positive,
                "positive_ratio": positive / float(mask.size),
                "ref": str(ref.path),
            }
        )
        if idx % 50 == 0:
            LOGGER.info("wrote %d/%d low-res masks", idx, len(lowres_refs))

    if args.highres and highres_geoms is not None:
        highres_dir = output_root / "masks_highres"
        for idx, ref in enumerate(highres_refs, start=1):
            mask = rasterize_patch(highres_geoms, ref)
            write_mask(highres_dir / f"{ref.patch_id}.tif", ref, mask)
            if idx % 50 == 0:
                LOGGER.info("wrote %d/%d high-res masks", idx, len(highres_refs))

    metadata = {
        "region": args.region,
        "task": args.task,
        "aoi": str(aoi_path),
        "osm_cache": str(osm_cache),
        "num_osm_features": int(len(osm)),
        "include_minor_roads": bool(args.include_minor_roads),
        "lowres_summary": summarize(records),
        "records": records,
        "label_kind": "osm_gt",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("summary: %s", metadata["lowres_summary"])
    LOGGER.info("saved labels to %s", output_root)


if __name__ == "__main__":
    main()
