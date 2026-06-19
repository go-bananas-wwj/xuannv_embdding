"""自动扫描数据盘并生成 docs/data_inventory.md。

本脚本读取 /data/xuannv_embedding 下各区域、各类别的实际目录大小与文件数量，
生成一份 Markdown 数据清单，避免手工维护时与真实文件系统脱节。

用法：
    python scripts/data/update_inventory.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path("/data/xuannv_embedding")
DOCS_ROOT = Path(__file__).parent.parent.parent / "docs"
OUTPUT_PATH = DOCS_ROOT / "data_inventory.md"

REGIONS = ["harbin", "haidian"]

# 每个 region 需要扫描的类别与 source/target。
# 顺序即为表格输出顺序。
ENTRIES: dict[str, list[dict[str, str]]] = {
    "harbin": [
        {"category": "Raw 时序", "name": "Sentinel-1", "source": "s1", "raw": "raw/harbin/s1"},
        {"category": "Raw 时序", "name": "Sentinel-2", "source": "s2", "raw": "raw/harbin/s2"},
        {"category": "Raw 时序", "name": "Landsat", "source": "landsat", "raw": "raw/harbin/landsat"},
        {"category": "Raw 标签", "name": "ESRI 2023 LULC", "source": "esri_lulc_2023", "raw": "raw/harbin/esri_lulc_2023"},
        {"category": "Raw 高分辨率光学", "name": "百度 DOM", "source": "highres_optical", "raw": "raw/harbin/highres_optical"},
        {"category": "Raw 地理辅助", "name": "DEM", "source": "dem", "raw": "raw/harbin/dem"},
        {"category": "Raw 地理辅助", "name": "JRC Water", "source": "jrc_water", "raw": "raw/harbin/jrc_water"},
        {"category": "Raw 地理辅助", "name": "Dynamic World", "source": "dynamic_world", "raw": "raw/harbin/dynamic_world"},
        {"category": "Processed Patch", "name": "S2/S1/Landsat", "source": "patches", "processed": "processed/harbin/patches"},
        {"category": "Processed 标签", "name": "worldcover", "source": "worldcover", "processed": "processed/harbin/labels/worldcover"},
        {"category": "Processed 地理辅助", "name": "-", "source": "geography", "processed": "processed/harbin/geography"},
        {"category": "元数据", "name": "manifest", "source": "manifest", "path": "processed/harbin/manifest.json"},
        {"category": "统计量", "name": "mean/std", "source": "statistics", "path": "statistics/harbin"},
    ],
    "haidian": [
        {"category": "Raw 时序", "name": "Sentinel-1", "source": "s1", "raw": "raw/haidian/s1"},
        {"category": "Raw 时序", "name": "Sentinel-2", "source": "s2", "raw": "raw/haidian/s2"},
        {"category": "Raw 时序", "name": "Landsat", "source": "landsat", "raw": "raw/haidian/landsat"},
        {"category": "Raw 标签", "name": "ESRI 2023 LULC", "source": "esri_lulc_2023", "raw": "raw/haidian/esri_lulc_2023"},
        {"category": "Raw 高分辨率光学", "name": "PlanetScene", "source": "highres_optical", "raw": "raw/haidian/highres_optical"},
        {"category": "Raw 高分辨率 SAR", "name": "天仪 SAR", "source": "highres_sar", "raw": "raw/haidian/highres_sar"},
        {"category": "Raw 地理辅助", "name": "DEM", "source": "dem", "raw": "raw/haidian/dem"},
        {"category": "Raw 地理辅助", "name": "JRC Water", "source": "jrc_water", "raw": "raw/haidian/jrc_water"},
        {"category": "Raw 地理辅助", "name": "Dynamic World", "source": "dynamic_world", "raw": "raw/haidian/dynamic_world"},
        {"category": "Processed Patch", "name": "S2/S1/Landsat", "source": "patches", "processed": "processed/haidian/patches"},
        {"category": "Processed 标签", "name": "worldcover", "source": "worldcover", "processed": "processed/haidian/labels/worldcover"},
        {"category": "Processed 地理辅助", "name": "-", "source": "geography", "processed": "processed/haidian/geography"},
        {"category": "元数据", "name": "manifest", "source": "manifest", "path": "processed/haidian/manifest.json"},
        {"category": "统计量", "name": "mean/std", "source": "statistics", "path": "statistics/haidian"},
    ],
}

# 对无法从文件系统直接推断的状态，可在此添加备注。
NOTES: dict[str, str] = {
    "harbin:highres_optical": "百度 DOM 202512/202605 已下完；待预处理",
    "haidian:highres_optical": "PlanetScene 已下完并解压；待预处理",
    "haidian:highres_sar": "天仪 SAR 已下完；待解压/地理编码/配准",
}


def du(path: Path) -> str:
    """返回目录或文件的 human-readable 大小。"""
    if not path.exists():
        return "0"
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.split()[0]
    except Exception as exc:  # noqa: BLE001
        return f"err:{exc}"


def count_files(path: Path, pattern: str = "*") -> int:
    """递归统计匹配文件数量。"""
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def status_for(region: str, entry: dict[str, str], file_count: int) -> str:
    """根据文件数量与备注推断状态。"""
    key = f"{region}:{entry['source']}"
    note = NOTES.get(key, "")
    category = entry["category"]
    name = entry["name"]

    if file_count == 0:
        if "地理辅助" in category or name in ("DEM", "JRC Water", "Dynamic World", "-"):
            return "⏳ 预留"
        return "⏳ 等待生成"

    if note:
        return "🔄 " + note
    return "✅ 已存在"


def gather_row(region: str, entry: dict[str, str]) -> dict[str, str]:
    """收集表格一行的数据。"""
    if "path" in entry:
        p = DATA_ROOT / entry["path"]
        size = du(p)
        files = count_files(p) if p.is_dir() else (1 if p.exists() else 0)
        path_str = entry["path"]
    elif "processed" in entry:
        p = DATA_ROOT / entry["processed"]
        # 对 labels 统计 tif，其余统计所有文件
        pattern = "*.tif" if entry["category"] == "Processed 标签" else "*"
        size = du(p)
        files = count_files(p, pattern)
        path_str = entry["processed"]
    else:
        p = DATA_ROOT / entry["raw"]
        size = du(p)
        files = count_files(p)
        path_str = entry["raw"]

    return {
        "category": entry["category"],
        "name": entry["name"],
        "path": f"`{path_str}`",
        "size": size,
        "files": str(files),
        "status": status_for(region, entry, files),
    }


def render_markdown() -> str:
    """渲染完整 Markdown。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# xuannv_embedding 数据清单",
        "",
        f"> 本文档由 `scripts/data/update_inventory.py` 自动生成，最后更新：{now}",
        "> 如需查看最新状态，请直接运行该脚本。",
        "",
    ]

    for region in REGIONS:
        lines.append(f"## {region.capitalize()}")
        lines.append("")
        lines.append(
            "| 类别 | Source / Target | 路径 | 大小 | 文件数 | 状态 |"
        )
        lines.append(
            "|------|-----------------|------|------|--------|------|"
        )
        for entry in ENTRIES[region]:
            row = gather_row(region, entry)
            lines.append(
                f"| {row['category']} | {row['name']} | {row['path']} | {row['size']} | {row['files']} | {row['status']} |"
            )
        lines.append("")

    lines.extend([
        "## 图例",
        "",
        "- ✅ 已存在/已完成",
        "- 🔄 进行中或部分完成",
        "- ⏳ 等待前置条件/预留",
        "- ❌ 失败/缺失",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    if not DATA_ROOT.exists():
        print(f"数据根目录不存在: {DATA_ROOT}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_markdown(), encoding="utf-8")
    print(f"已生成数据清单: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
