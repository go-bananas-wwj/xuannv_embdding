"""生成哈尔滨新区与海淀的占位区域 GeoJSON（仅用于测试，后续替换为真实边界）。"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

# 哈尔滨新区：松北区附近小范围（WGS84）
HARBIN_BBOX = (126.45, 45.75, 126.55, 45.85)
# 北京海淀：中关村附近小范围（WGS84）
HAIDIAN_BBOX = (116.28, 39.95, 116.38, 40.05)


def bbox_to_polygon(bbox: tuple[float, float, float, float]) -> Polygon:
    """将 bbox 转为简单四边形多边形。"""
    minx, miny, maxx, maxy = bbox
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])


def save_placeholder(name: str, bbox: tuple[float, float, float, float], out_dir: Path) -> Path:
    """保存单个占位 GeoJSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = bbox_to_polygon(bbox)
    gdf = gpd.GeoDataFrame({"name": [name], "geometry": [geom]}, crs="EPSG:4326")
    out_path = out_dir / f"{name}.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def main() -> None:
    out_dir = Path("/root/workspace/xuannv/configs/regions")
    save_placeholder("harbin", HARBIN_BBOX, out_dir)
    save_placeholder("haidian", HAIDIAN_BBOX, out_dir)
    print(f"已生成占位区域文件到 {out_dir}")


if __name__ == "__main__":
    main()
