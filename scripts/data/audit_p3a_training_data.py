#!/usr/bin/env python3
"""Audit P3A joint training data before launching a new run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


DEFAULT_MANIFEST = Path(
    "/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/manifest_v2_202512_202605.json"
)
DEFAULT_PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")
DEFAULT_STATS_ROOT = Path("/data/xuannv_embedding/statistics")
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/experiments/p3a_training_data_audit_20260629")
DEFAULT_DOC_COPY = Path("docs/experiments/p3a_training_data_audit_20260629_zh.md")

SOURCES = [
    "s2",
    "s1",
    "landsat",
    "worldcover",
    "highres_optical_haidian",
    "highres_optical_harbin",
    "highres_sar_haidian",
]

OSM_TASKS = [
    "osm_activity_poi",
    "osm_agriculture",
    "osm_building",
    "osm_commercial",
    "osm_construction",
    "osm_green",
    "osm_industrial",
    "osm_major_road",
    "osm_minor_road",
    "osm_path_walk",
    "osm_rail",
    "osm_residential",
    "osm_water",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--stats-root", type=Path, default=DEFAULT_STATS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-copy", type=Path, default=DEFAULT_DOC_COPY)
    parser.add_argument("--sample-limit-per-source", type=int, default=20)
    return parser.parse_args()


def resolve_path(manifest_root: Path, rel: str | Path) -> Path:
    return (manifest_root / Path(rel)).resolve()


def read_shape(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "count": int(src.count),
            "height": int(src.height),
            "width": int(src.width),
            "crs": str(src.crs),
            "transform": tuple(round(float(v), 6) for v in src.transform[:6]),
        }


def audit_manifest(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    root = args.manifest.parent
    region_counts = Counter(entry.get("region", "unknown") for entry in manifest)
    source_stats: dict[str, Any] = {}
    missing_paths: list[str] = []

    for source in SOURCES:
        by_region = defaultdict(lambda: {"patches_with_source": 0, "frames": 0})
        shape_counter: Counter[str] = Counter()
        sampled = 0
        for entry in manifest:
            region = entry.get("region", "unknown")
            paths = entry.get(source) or []
            if paths:
                by_region[region]["patches_with_source"] += 1
                by_region[region]["frames"] += len(paths)
            for rel in paths:
                path = resolve_path(root, rel)
                if not path.exists():
                    missing_paths.append(str(path))
                    continue
                if sampled < args.sample_limit_per_source:
                    try:
                        shape = read_shape(path)
                        shape_key = (
                            f"{shape['count']}x{shape['height']}x{shape['width']} "
                            f"{shape['crs']} {shape['transform']}"
                        )
                        shape_counter[shape_key] += 1
                        sampled += 1
                    except Exception as exc:
                        missing_paths.append(f"{path} read_error={exc}")
        source_stats[source] = {
            "by_region": {region: dict(stats) for region, stats in sorted(by_region.items())},
            "sampled_shapes": dict(shape_counter),
        }

    return {
        "num_patches": len(manifest),
        "region_counts": dict(region_counts),
        "source_stats": source_stats,
        "missing_paths": missing_paths[:100],
        "num_missing_paths": len(missing_paths),
    }


def audit_stats_files(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for region in ["haidian", "harbin"]:
        region_result = {}
        for source in ["s2", "s1", "landsat", "highres_optical", "highres_optical_harbin", "highres_sar"]:
            path = args.stats_root / region / f"{source}_stats.json"
            if not path.exists():
                region_result[source] = {"exists": False}
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            region_result[source] = {
                "exists": True,
                "channels": len(data.get("mean", [])),
                "count": data.get("count"),
            }
        result[region] = region_result
    return result


def audit_osm(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region in ["haidian", "harbin"]:
        region_result = {}
        for task in OSM_TASKS:
            meta_path = args.processed_root / region / "labels" / task / "metadata.json"
            if not meta_path.exists():
                region_result[task] = {"exists": False}
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary = meta.get("summary", {})
            records = meta.get("records", [])
            high_cover = sum(1 for record in records if float(record.get("positive_ratio", 0.0)) >= 0.8)
            empty = sum(1 for record in records if int(record.get("positive_pixels", 0)) == 0)
            region_result[task] = {
                "exists": True,
                **summary,
                "high_cover_patches_80pct": high_cover,
                "empty_patches": empty,
            }
        result[region] = region_result
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# P3A 训练数据审查",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- Manifest：`{payload['manifest_path']}`",
        f"- 输出目录：`{payload['output_root']}`",
        "",
        "## 总体结论",
        "",
    ]
    if manifest["num_missing_paths"] == 0:
        lines.append("联合训练数据路径检查通过：manifest 中抽查到的影像路径均存在。")
    else:
        lines.append(f"发现 {manifest['num_missing_paths']} 个缺失或读取失败路径，需要先修复。")
    lines.extend(
        [
            "",
            f"- 总 patch 数：{manifest['num_patches']}",
            f"- 区域分布：{json.dumps(manifest['region_counts'], ensure_ascii=False)}",
            "",
            "## Source 可用性",
            "",
            "| source | region | patch 有数据数 | frame 数 | 抽样 shape/CRS |",
            "|---|---|---:|---:|---|",
        ]
    )
    for source, stats in manifest["source_stats"].items():
        shape_text = "; ".join(stats["sampled_shapes"].keys()) or "-"
        for region, item in stats["by_region"].items():
            lines.append(
                f"| {source} | {region} | {item['patches_with_source']} | {item['frames']} | {shape_text} |"
            )

    lines.extend(["", "## 统计文件", "", "| region | source | exists | channels | count |", "|---|---|---:|---:|---:|"])
    for region, sources in payload["stats_files"].items():
        for source, item in sources.items():
            lines.append(
                f"| {region} | {source} | {item.get('exists', False)} | {item.get('channels', '-')} | {item.get('count', '-')} |"
            )

    lines.extend(
        [
            "",
            "## OSM 弱语义标签覆盖",
            "",
            "| region | task | exists | patch 数 | 正样本 patch | 平均正像素 | P95 正像素 | >=80% 满覆盖 patch | 空 patch |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for region, tasks in payload["osm"].items():
        for task, item in sorted(tasks.items()):
            lines.append(
                f"| {region} | {task} | {item.get('exists', False)} | "
                f"{item.get('num_patches', '-')} | {item.get('positive_patches', '-')} | "
                f"{float(item.get('positive_ratio_mean', 0))*100:.2f}% | "
                f"{float(item.get('positive_ratio_p95', 0))*100:.2f}% | "
                f"{item.get('high_cover_patches_80pct', '-')} | {item.get('empty_patches', '-')} |"
            )

    lines.extend(
        [
            "",
            "## 训练前注意事项",
            "",
            "1. `osm_green`、`osm_water`、`osm_activity_poi` 在部分 patch 覆盖很大，训练时必须低权重。",
            "2. OSM 只作为弱语义辅助，不参与 supervised change loss。",
            "3. 训练仍应联合海淀和哈尔滨，避免模型只记住单一区域。",
            "4. 下游人工标注不进入主 embedding 训练，只保留训练后测评。",
            "",
        ]
    )
    if manifest["missing_paths"]:
        lines.extend(["## 缺失路径样例", ""])
        lines.extend(f"- `{path}`" for path in manifest["missing_paths"][:30])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "output_root": str(args.output_root),
        "manifest": audit_manifest(args, manifest),
        "stats_files": audit_stats_files(args),
        "osm": audit_osm(args),
    }
    json_path = args.output_root / "p3a_training_data_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(payload)
    md_path = args.output_root / "p3a_training_data_audit_zh.md"
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
