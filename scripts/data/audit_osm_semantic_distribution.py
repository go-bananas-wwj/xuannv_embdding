#!/usr/bin/env python3
"""Download and audit broad OSM semantic distributions for weak supervision."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, shape


LOGGER = logging.getLogger(__name__)

REGION_AOI = {
    "haidian": Path("configs/regions/haidian.geojson"),
    "harbin": Path("configs/regions/harbin.geojson"),
}

DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629")
DEFAULT_DOC_COPY = Path("docs/experiments/p3a_osm_semantic_distribution_20260629_zh.md")

THEME_TAGS: dict[str, dict[str, Any]] = {
    "landuse": {"landuse": True},
    "natural": {"natural": True, "water": True, "waterway": True},
    "transport": {"highway": True, "railway": True, "aeroway": True},
    "built": {"building": True, "man_made": True, "construction": True},
    "activity": {
        "amenity": True,
        "shop": True,
        "office": True,
        "tourism": True,
        "leisure": True,
        "industrial": True,
    },
}

VALUE_GROUPS: dict[str, dict[str, set[str]]] = {
    "landuse": {
        "residential": {"residential", "apartments", "village_green"},
        "commercial": {"commercial", "retail", "business"},
        "industrial": {"industrial", "brownfield", "garages", "depot", "quarry", "landfill"},
        "agriculture": {"farmland", "farmyard", "orchard", "vineyard", "greenhouse_horticulture"},
        "forest_green": {"forest", "meadow", "grass", "recreation_ground", "allotments"},
        "education_public": {"education", "school", "university", "institutional"},
        "construction": {"construction"},
        "transport": {"railway", "highway"},
        "military": {"military"},
        "cemetery": {"cemetery"},
    },
    "natural": {
        "water_wetland": {"water", "wetland", "bay", "strait"},
        "forest_green": {"wood", "tree_row", "scrub", "grassland", "heath"},
        "bare_sparse": {"bare_rock", "sand", "scree", "shingle", "mud"},
    },
    "water": {
        "water_wetland": {
            "lake",
            "pond",
            "river",
            "reservoir",
            "basin",
            "canal",
            "ditch",
            "stream",
            "reflecting_pool",
        }
    },
    "waterway": {
        "water_wetland": {"river", "stream", "canal", "ditch", "drain", "riverbank"},
        "transport": {"dam", "weir", "lock_gate"},
    },
    "highway": {
        "major_road": {"motorway", "trunk", "primary", "secondary", "tertiary"},
        "minor_road": {"residential", "unclassified", "service", "living_street"},
        "path_walk": {"footway", "path", "cycleway", "steps", "pedestrian", "track"},
        "road_poi": {"crossing", "bus_stop", "traffic_signals", "turning_circle", "street_lamp"},
    },
    "railway": {
        "rail_transport": {"rail", "subway", "light_rail", "tram", "station", "halt"},
    },
    "building": {
        "residential_building": {"residential", "apartments", "house", "detached", "dormitory"},
        "commercial_building": {"commercial", "retail", "office", "hotel"},
        "industrial_building": {"industrial", "warehouse", "factory"},
        "education_public": {"school", "university", "college", "public", "hospital"},
        "generic_building": {"yes", "roof", "garages", "service", "hut", "shed"},
        "construction": {"construction"},
    },
    "amenity": {
        "education_public": {"school", "university", "college", "kindergarten", "library"},
        "health_public": {"hospital", "clinic", "doctors", "pharmacy"},
        "commercial_service": {"restaurant", "cafe", "fast_food", "fuel", "bank", "marketplace"},
        "public_service": {"police", "fire_station", "post_office", "townhall", "toilets"},
        "transport": {"parking", "bus_station", "bicycle_parking"},
    },
    "shop": {
        "commercial_service": set(),
    },
    "office": {
        "commercial_service": set(),
    },
    "tourism": {
        "commercial_service": {"hotel", "guest_house", "hostel"},
        "culture_recreation": {"museum", "attraction", "artwork", "viewpoint", "picnic_site"},
    },
    "leisure": {
        "culture_recreation": {"park", "garden", "pitch", "sports_centre", "playground", "stadium"},
        "forest_green": {"nature_reserve"},
    },
    "man_made": {
        "industrial": {"works", "storage_tank", "wastewater_plant", "silo", "chimney"},
        "infrastructure": {"tower", "pier", "bridge", "pipeline", "pumping_station"},
    },
}


@dataclass(frozen=True)
class ThemeResult:
    region: str
    theme: str
    cache_path: Path
    features: gpd.GeoDataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-copy", type=Path, default=DEFAULT_DOC_COPY)
    parser.add_argument("--regions", nargs="+", default=sorted(REGION_AOI))
    parser.add_argument("--themes", nargs="+", default=sorted(THEME_TAGS))
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_aoi_polygon(path: Path) -> Polygon:
    data = json.loads(path.read_text(encoding="utf-8"))
    polygon = shape(data["features"][0]["geometry"])
    if not isinstance(polygon, Polygon):
        polygon = polygon.convex_hull
    return polygon


def sanitize_for_gpkg(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = [
        "element",
        "id",
        "osmid",
        "landuse",
        "natural",
        "water",
        "waterway",
        "highway",
        "railway",
        "aeroway",
        "building",
        "building:use",
        "man_made",
        "construction",
        "amenity",
        "shop",
        "office",
        "tourism",
        "leisure",
        "industrial",
        "name",
        "geometry",
    ]
    out = gdf[[col for col in keep if col in gdf.columns]].copy()
    for col in out.columns:
        if col == "geometry":
            continue
        out[col] = out[col].map(
            lambda value: ",".join(map(str, value))
            if isinstance(value, list)
            else (None if value is None else str(value))
        )
    return out


def read_or_download(region: str, theme: str, output_root: Path, overwrite: bool) -> ThemeResult:
    cache_dir = output_root / "cache"
    cache_path = cache_dir / f"{region}_{theme}.gpkg"
    if cache_path.exists() and not overwrite:
        LOGGER.info("reading cache %s", cache_path)
        gdf = gpd.read_file(cache_path)
        return ThemeResult(region, theme, cache_path, gdf)

    LOGGER.info("downloading %s / %s", region, theme)
    polygon = load_aoi_polygon(REGION_AOI[region])
    gdf = ox.features_from_polygon(polygon, THEME_TAGS[theme])
    if gdf.empty:
        gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    else:
        gdf = gdf.reset_index()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf = gdf.set_geometry("geometry")
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = sanitize_for_gpkg(gdf)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GPKG")
    LOGGER.info("cached %d features to %s", len(gdf), cache_path)
    return ThemeResult(region, theme, cache_path, gdf)


def metric_frame(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    try:
        crs = gdf.estimate_utm_crs()
    except Exception:
        crs = None
    return gdf.to_crs(crs or "EPSG:3857")


def clip_to_region(gdf: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    polygon = load_aoi_polygon(REGION_AOI[region])
    aoi = gpd.GeoDataFrame({"geometry": [polygon]}, geometry="geometry", crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    aoi = aoi.to_crs(gdf.crs)
    try:
        clipped = gpd.clip(gdf, aoi)
    except Exception:
        clipped = gdf[gdf.intersects(aoi.geometry.iloc[0])].copy()
        clipped["geometry"] = clipped.geometry.intersection(aoi.geometry.iloc[0])
    return clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()


def non_null_values(gdf: gpd.GeoDataFrame, column: str) -> gpd.GeoSeries:
    if column not in gdf.columns:
        return gpd.GeoSeries([], dtype="object")
    values = gdf[column].astype("object")
    mask = values.notna() & (values.astype(str) != "") & (values.astype(str) != "nan")
    return values.loc[mask].astype(str)


def map_semantic_group(column: str, value: str) -> str | None:
    mappings = VALUE_GROUPS.get(column, {})
    value_norm = value.strip().lower()
    for group, values in mappings.items():
        if not values or value_norm in values:
            return group
    return None


def summarize_column(gdf: gpd.GeoDataFrame, metric: gpd.GeoDataFrame, column: str) -> dict[str, Any]:
    values = non_null_values(gdf, column)
    if values.empty:
        return {"non_null": 0, "top_values": [], "groups": {}}

    top_values = [
        {"value": str(index), "count": int(count)}
        for index, count in values.value_counts().head(20).items()
    ]
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"features": 0, "area_km2": 0.0, "length_km": 0.0})
    for idx, value in values.items():
        group = map_semantic_group(column, value)
        if group is None:
            continue
        geom = metric.geometry.loc[idx]
        groups[group]["features"] += 1
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            groups[group]["area_km2"] += float(geom.area) / 1_000_000.0
        elif geom.geom_type in {"LineString", "MultiLineString"}:
            groups[group]["length_km"] += float(geom.length) / 1_000.0
    return {
        "non_null": int(len(values)),
        "top_values": top_values,
        "groups": {key: dict(value) for key, value in sorted(groups.items())},
    }


def summarize_theme(result: ThemeResult) -> dict[str, Any]:
    gdf = clip_to_region(result.features, result.region)
    metric = metric_frame(gdf)
    geom_counts = gdf.geometry.geom_type.astype(str).value_counts().to_dict() if not gdf.empty else {}
    columns = sorted(set(VALUE_GROUPS).intersection(gdf.columns))
    column_summaries = {col: summarize_column(gdf, metric, col) for col in columns}
    return {
        "region": result.region,
        "theme": result.theme,
        "cache_path": str(result.cache_path),
        "num_features": int(len(gdf)),
        "num_columns": int(len(gdf.columns)),
        "geometry_types": {str(k): int(v) for k, v in geom_counts.items()},
        "columns": column_summaries,
    }


def aggregate_region(theme_summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = defaultdict(lambda: {"features": 0, "area_km2": 0.0, "length_km": 0.0})
    for summary in theme_summaries:
        for col_summary in summary["columns"].values():
            for group, stats in col_summary["groups"].items():
                merged[group]["features"] += int(stats["features"])
                merged[group]["area_km2"] += float(stats["area_km2"])
                merged[group]["length_km"] += float(stats["length_km"])
    return {key: dict(value) for key, value in sorted(merged.items())}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P3A OSM 语义分布调研",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据输出：`{payload['output_root']}`",
        "- 目的：把 OSM 作为静态/半静态弱语义先验，调研哪些标签能服务月度 embedding 训练。",
        "- 约束：OSM 标签不作为强真值；OSM 空白不作为强负样本；下游人工标注不进入主 embedding 训练。",
        "",
        "## 总体建议",
        "",
        "OSM 可以用于训练大概语义分布，尤其是建筑、道路、居住/商业/工业/农业/水体/绿地等粗粒度类别。",
        "建议使用密度图、面积比例图、距离图或 confidence mask，而不是逐像素硬标签。",
        "",
        "## 区域级语义覆盖",
        "",
    ]
    for region, groups in payload["region_groups"].items():
        lines.extend(
            [
                f"### {region}",
                "",
                "| 弱语义组 | 要素数 | 面积 km2 | 线长度 km | 建议用途 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for group, stats in sorted(
            groups.items(),
            key=lambda item: (item[1]["area_km2"] + item[1]["length_km"] / 10.0, item[1]["features"]),
            reverse=True,
        ):
            lines.append(
                f"| {group} | {int(stats['features'])} | {stats['area_km2']:.2f} | "
                f"{stats['length_km']:.2f} | {recommend_usage(group)} |"
            )
        lines.append("")

    lines.extend(["## 主题明细", ""])
    for item in payload["themes"]:
        lines.extend(
            [
                f"### {item['region']} / {item['theme']}",
                "",
                f"- 缓存：`{item['cache_path']}`",
                f"- 要素数：{item['num_features']}",
                f"- 几何类型：{json.dumps(item['geometry_types'], ensure_ascii=False)}",
                "",
            ]
        )
        for column, summary in item["columns"].items():
            top = ", ".join(f"{v['value']}:{v['count']}" for v in summary["top_values"][:12]) or "-"
            lines.append(f"- `{column}` 有值 {summary['non_null']}；Top values：{top}")
        lines.append("")

    lines.extend(
        [
            "## 可进入 P3A 的 OSM 弱标签",
            "",
            "优先级高：",
            "",
            "1. `building`：建筑密度、建筑面积比例、建筑边界/距离图。",
            "2. `highway`：主路/小路/步行路分层道路密度、道路中心线距离图。",
            "3. `landuse=residential/commercial/industrial/farmland/forest/grass`：粗土地利用分布。",
            "4. `natural=water/wood/wetland/grassland` 和 `waterway=*`：水体、湿地、绿地弱监督。",
            "",
            "优先级中：",
            "",
            "1. `amenity/shop/office/tourism/leisure`：功能区 POI 密度，例如商业服务、教育公共、医疗公共、文体休闲。",
            "2. `man_made`：工业设施和基础设施分布。",
            "3. `railway/aeroway`：交通设施先验。",
            "",
            "暂时不用作强监督：",
            "",
            "1. 过细的 POI 类别，例如具体店铺类型。",
            "2. 稀有或格式不稳定的标签。",
            "3. OSM 空白区域。",
            "",
            "## 训练接入方式",
            "",
            "P3A 中建议新增一个 `osm_weak_semantic` 辅助目标，输出 128x128 的多通道弱语义图：",
            "",
            "- `building_density`",
            "- `major_road_density`",
            "- `minor_road_density`",
            "- `water_density`",
            "- `green_density`",
            "- `residential_area`",
            "- `commercial_area`",
            "- `industrial_area`",
            "- `agriculture_area`",
            "- `poi_activity_density`",
            "",
            "loss 用 BCE/Huber/Dice 的轻权重组合，并配 confidence mask。它只帮助 embedding 学到大概语义分布，不让 OSM 主导模型。",
            "",
        ]
    )
    return "\n".join(lines)


def recommend_usage(group: str) -> str:
    if group in {"residential", "commercial", "industrial", "agriculture", "forest_green"}:
        return "土地利用面积比例/密度弱监督"
    if "road" in group or group in {"transport", "rail_transport"}:
        return "道路/交通线密度和距离图"
    if group in {"water_wetland"}:
        return "水体/湿地密度弱监督"
    if "building" in group or group == "generic_building":
        return "建筑密度和边界弱监督"
    if group in {"commercial_service", "education_public", "health_public", "culture_recreation"}:
        return "功能区 POI 密度弱监督"
    return "低权重语义先验"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ox.settings.timeout = args.timeout
    args.output_root.mkdir(parents=True, exist_ok=True)

    theme_summaries = []
    for region in args.regions:
        for theme in args.themes:
            result = read_or_download(region, theme, args.output_root, args.overwrite_cache)
            theme_summaries.append(summarize_theme(result))

    region_groups = {
        region: aggregate_region([item for item in theme_summaries if item["region"] == region])
        for region in args.regions
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "themes": theme_summaries,
        "region_groups": region_groups,
    }
    json_path = args.output_root / "osm_semantic_distribution.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(payload)
    md_path = args.output_root / "osm_semantic_distribution_zh.md"
    md_path.write_text(md, encoding="utf-8")
    if args.doc_copy:
        args.doc_copy.parent.mkdir(parents=True, exist_ok=True)
        args.doc_copy.write_text(md, encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if args.doc_copy:
        print(f"wrote {args.doc_copy}")


if __name__ == "__main__":
    main()
