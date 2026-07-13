#!/usr/bin/env python3
"""为 18 类地物评测构建新增 OSM 标签层（海淀，离线 gpkg 缓存）。

新增图层：
- osm_river / osm_lake / osm_pond：水体三分。规则：
  1) water=* 子 tag 精确分派；
  2) 无 tag 的 natural=water 多边形：与 waterway=river/canal/stream 线要素相交 → river；
     否则按面积阈值（>= 3 ha → lake，< 3 ha → pond）；
  3) waterway 面要素（riverbank 等）与 waterway 主要线要素（buffer）并入 river。
- osm_bare：landuse=brownfield/greenfield + natural=bare_rock/sand/scree。
- osm_stadium：building=stadium + leisure=stadium。
- osm_train_station：railway=station（面）+ building=train_station。
- osm_wetland：natural=wetland（检索演示用真值）。
- osm_landfill：landuse=landfill（检索演示用真值）。

输出结构与既有标签层一致：masks/*.tif + metadata.json + split_5fold.json。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_weak = _load("osm_weak_labels", REPO_ROOT / "scripts/data/build_osm_weak_semantic_labels.py")
_split = _load("ds_split", REPO_ROOT / "downstreams/downstreams/data/split.py")
choose_patch_refs = _weak.choose_patch_refs
load_theme = _weak.load_theme
rasterize_patch = _weak.rasterize_patch
summarize = _weak.summarize
write_mask = _weak.write_mask
create_stratified_folds = _split.create_stratified_folds

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE = Path("/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache")
DEFAULT_PROCESSED = Path("/data/xuannv_embedding/processed")

RIVER_WATER_TAGS = {"river", "canal", "stream", "moat", "ditch", "drain"}
LAKE_WATER_TAGS = {"lake", "reservoir"}
POND_WATER_TAGS = {"pond", "basin", "fishpond", "reflecting_pool", "lagoon"}
WATERWAY_LINE_TAGS = {"river", "canal", "stream"}
LAKE_AREA_MIN_M2 = 30_000.0  # 3 ha：无 tag 水体按面积分派 lake/pond
RIVER_LINE_BUFFER_M = 15.0


def polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()


def col_eq(gdf: gpd.GeoDataFrame, col: str, values: set[str]) -> gpd.GeoDataFrame:
    if col not in gdf.columns:
        return gdf.iloc[0:0].copy()
    s = gdf[col].astype("object")
    keep = s.notna() & s.astype(str).str.lower().isin({v.lower() for v in values})
    return gdf[keep].copy()


def split_water(nat: gpd.GeoDataFrame, utm_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """返回 river/lake/pond 三个 GeoDataFrame（EPSG:4326）。"""
    water = polygons(nat[nat.get("natural").astype("object").astype(str).str.lower() == "water"])
    tag = water["water"].astype("object") if "water" in water.columns else None
    tagged = tag.notna() & (tag.astype(str) != "") & (tag.astype(str).str.lower() != "nan")

    river_parts, lake_parts, pond_parts = [], [], []
    tv = tag.astype(str).str.lower()
    river_parts.append(water[tagged & tv.isin(RIVER_WATER_TAGS)])
    lake_parts.append(water[tagged & tv.isin(LAKE_WATER_TAGS)])
    pond_parts.append(water[tagged & tv.isin(POND_WATER_TAGS)])
    other_tagged = water[tagged & ~tv.isin(RIVER_WATER_TAGS | LAKE_WATER_TAGS | POND_WATER_TAGS)]

    # waterway 主要线要素（用于兜底判断 + 直接并入 river）
    ww_lines = nat.iloc[0:0]
    if "waterway" in nat.columns:
        ww = nat[nat["waterway"].astype("object").astype(str).str.lower().isin(WATERWAY_LINE_TAGS)]
        ww_lines = ww[ww.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        ww_polys = polygons(ww)
        if len(ww_polys):
            river_parts.append(ww_polys)

    untagged = gpd.GeoDataFrame(
        __import__("pandas").concat([water[~tagged], other_tagged]), crs=water.crs
    )
    if len(untagged):
        utm = untagged.to_crs(utm_crs)
        areas = utm.area.to_numpy()
        if len(ww_lines):
            lines_union = ww_lines.to_crs(utm_crs).union_all()
            touches_river = utm.geometry.intersects(lines_union).to_numpy()
        else:
            touches_river = np.zeros(len(untagged), dtype=bool)
        river_parts.append(untagged[touches_river])
        rest = untagged[~touches_river]
        rest_areas = areas[~touches_river]
        lake_parts.append(rest[rest_areas >= LAKE_AREA_MIN_M2])
        pond_parts.append(rest[rest_areas < LAKE_AREA_MIN_M2])

    # waterway 线要素 buffer 后并入 river
    if len(ww_lines):
        buf = ww_lines.to_crs(utm_crs).buffer(RIVER_LINE_BUFFER_M).to_crs(water.crs)
        river_parts.append(gpd.GeoDataFrame({"geometry": buf}, crs=water.crs))

    import pandas as pd

    def merge(parts):
        parts = [p[["geometry"]] for p in parts if len(p)]
        if not parts:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        return out[out.geometry.notna() & ~out.geometry.is_empty]

    return {"osm_river": merge(river_parts), "osm_lake": merge(lake_parts), "osm_pond": merge(pond_parts)}


def build_simple_layers(themes: dict[str, gpd.GeoDataFrame]) -> dict[str, gpd.GeoDataFrame]:
    lu, tr, bu, nat = themes["landuse"], themes["transport"], themes["built"], themes["natural"]
    import pandas as pd

    def merge(parts):
        parts = [polygons(p)[["geometry"]] for p in parts if len(p)]
        if not parts:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
        return out[out.geometry.notna() & ~out.geometry.is_empty]

    return {
        "osm_bare": merge([
            col_eq(lu, "landuse", {"brownfield", "greenfield"}),
            col_eq(nat, "natural", {"bare_rock", "sand", "scree", "shingle"}),
        ]),
        "osm_stadium": merge([
            col_eq(bu, "building", {"stadium"}),
            col_eq(lu, "leisure", {"stadium"}),
        ]),
        "osm_train_station": merge([
            col_eq(tr, "railway", {"station"}),
            col_eq(bu, "building", {"train_station"}),
        ]),
        "osm_wetland": merge([col_eq(nat, "natural", {"wetland"})]),
        "osm_landfill": merge([col_eq(lu, "landuse", {"landfill"})]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    refs = choose_patch_refs(args.processed_root / args.region / "patches" / "s2")
    LOGGER.info("patch refs: %d, CRS=%s", len(refs), refs[0].crs)
    utm_crs = str(refs[0].crs)

    themes = {t: load_theme(args.cache_root, args.region, t) for t in ["natural", "landuse", "transport", "built"]}
    layers = split_water(themes["natural"], utm_crs)
    layers.update(build_simple_layers(themes))

    for task, gdf in layers.items():
        out_root = args.processed_root / args.region / "labels" / task
        meta_path = out_root / "metadata.json"
        if meta_path.exists() and not args.overwrite:
            LOGGER.info("skip existing %s", task)
            continue
        geoms = gdf.to_crs(utm_crs)
        records = []
        for ref in refs:
            mask = rasterize_patch(geoms, ref)
            write_mask(out_root / "masks" / f"{ref.patch_id}.tif", ref, mask)
            pos = int((mask > 0).sum())
            records.append({
                "patch_id": ref.patch_id,
                "positive_pixels": pos,
                "positive_ratio": pos / float(mask.size),
            })
        meta = {
            "region": args.region,
            "task": task,
            "label_kind": "osm_18class_20260713",
            "source_cache_root": str(args.cache_root),
            "raw_feature_count": int(len(gdf)),
            "summary": summarize(records),
            "records": records,
        }
        out_root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        split = create_stratified_folds(out_root / "masks", n_folds=5, val_ratio=0.1, seed=42)
        (out_root / "split_5fold.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        LOGGER.info("%s: %s", task, meta["summary"])

    print("done")


if __name__ == "__main__":
    main()
