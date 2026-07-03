#!/usr/bin/env python3
"""Merge cleaned OSM weak masks into one WorldCover-like land-cover label."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


@dataclass(frozen=True)
class LandcoverClass:
    class_id: int
    name: str
    tasks: tuple[str, ...]
    color: str


# Broad classes are written first; later fine classes override overlapping pixels.
CLASS_RULES: tuple[LandcoverClass, ...] = (
    LandcoverClass(1, "residential", ("osm_residential",), "#f9844a"),
    LandcoverClass(2, "commercial", ("osm_commercial",), "#ff70a6"),
    LandcoverClass(3, "industrial", ("osm_industrial",), "#6d6875"),
    LandcoverClass(4, "agriculture", ("osm_agriculture",), "#90be6d"),
    LandcoverClass(5, "green", ("osm_green",), "#43aa8b"),
    LandcoverClass(6, "recreation", ("osm_playground",), "#577590"),
    LandcoverClass(7, "construction", ("osm_construction",), "#9d4edd"),
    LandcoverClass(8, "water", ("osm_water",), "#4aa3df"),
    LandcoverClass(9, "building", ("osm_building",), "#d62828"),
    LandcoverClass(
        10,
        "transport",
        ("osm_path_walk", "osm_rail", "osm_minor_road", "osm_major_road"),
        "#f4d35e",
    ),
)
EXCLUDED_TASKS = ("osm_activity_poi",)


def _read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1) > 0


def _read_mask_optional(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(path) as src:
        return src.read(1) > 0


def _reference_mask(label_root: Path, patch_id: str) -> Path:
    for task in [task for rule in CLASS_RULES for task in rule.tasks]:
        path = label_root / task / "masks" / f"{patch_id}.tif"
        if path.exists():
            return path
    raise FileNotFoundError(f"no OSM mask found for {patch_id}")


def _patch_ids(label_root: Path) -> list[str]:
    ids: set[str] = set()
    for rule in CLASS_RULES:
        for task in rule.tasks:
            mask_dir = label_root / task / "masks"
            if not mask_dir.exists():
                continue
            for path in mask_dir.glob("patch_*.tif"):
                ids.add(path.stem)
    return sorted(ids)


def build_label(label_root: Path, patch_id: str) -> tuple[np.ndarray, dict[str, int]]:
    ref_path = _reference_mask(label_root, patch_id)
    with rasterio.open(ref_path) as src:
        shape = (src.height, src.width)
    label = np.zeros(shape, dtype=np.uint8)
    source_pixels: dict[str, int] = {}
    for rule in CLASS_RULES:
        merged = np.zeros(shape, dtype=bool)
        for task in rule.tasks:
            task_mask = _read_mask_optional(
                label_root / task / "masks" / f"{patch_id}.tif",
                shape,
            )
            source_pixels[task] = int(task_mask.sum())
            merged |= task_mask
        label[merged] = rule.class_id
    return label, source_pixels


def write_label(
    label_root: Path,
    output_root: Path,
    patch_id: str,
    label: np.ndarray,
    date: str,
) -> Path:
    ref_path = _reference_mask(label_root, patch_id)
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    output_path = output_root / "masks" / f"osm_landcover_{date}_{patch_id}.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(label, 1)
    return output_path


def _class_names() -> dict[int, str]:
    return {0: "background"} | {rule.class_id: rule.name for rule in CLASS_RULES}


def _class_colors() -> dict[int, str]:
    return {0: "#ffffff"} | {rule.class_id: rule.color for rule in CLASS_RULES}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_names = _class_names()
    total_pixels = sum(int(record["total_pixels"]) for record in records)
    class_pixels = {str(class_id): 0 for class_id in class_names}
    positive_patches = {str(class_id): 0 for class_id in class_names}
    source_pixels: dict[str, int] = {}

    for record in records:
        for class_id, count in record["class_pixels"].items():
            class_pixels[str(class_id)] += int(count)
            if int(count) > 0:
                positive_patches[str(class_id)] += 1
        for task, count in record["source_pixels"].items():
            source_pixels[task] = source_pixels.get(task, 0) + int(count)

    classes = []
    for class_id in sorted(class_names):
        pixels = int(class_pixels[str(class_id)])
        classes.append(
            {
                "class_id": class_id,
                "name": class_names[class_id],
                "pixels": pixels,
                "pixel_ratio": pixels / max(total_pixels, 1),
                "positive_patches": int(positive_patches[str(class_id)]),
                "positive_patch_ratio": int(positive_patches[str(class_id)]) / max(len(records), 1),
            }
        )

    return {
        "num_patches": len(records),
        "total_pixels": total_pixels,
        "class_names": class_names,
        "class_colors": _class_colors(),
        "class_rules": [
            {
                "class_id": rule.class_id,
                "name": rule.name,
                "tasks": list(rule.tasks),
                "color": rule.color,
            }
            for rule in CLASS_RULES
        ],
        "excluded_tasks": list(EXCLUDED_TASKS),
        "classes": classes,
        "source_task_pixels": source_pixels,
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# OSM Land-Cover Merge Summary",
        "",
        f"- Patches: {summary['num_patches']}",
        f"- Pixels: {summary['total_pixels']}",
        f"- Excluded noisy tasks: {', '.join(summary['excluded_tasks'])}",
        "",
        "## Merged Classes",
        "",
        "| ID | Class | Source OSM tasks | Pixel ratio | Positive patches |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    rules_by_id = {rule["class_id"]: rule for rule in summary["class_rules"]}
    for cls in summary["classes"]:
        rule = rules_by_id.get(cls["class_id"], {"tasks": []})
        tasks = ", ".join(rule["tasks"])
        lines.append(
            "| {class_id} | {name} | {tasks} | {pixel_ratio:.2%} | {positive_patches} ({positive_patch_ratio:.2%}) |".format(
                tasks=tasks,
                **cls,
            )
        )
    lines.extend(["", "## Source Task Pixels", "", "| OSM task | Pixels |", "| --- | ---: |"])
    for task, pixels in sorted(summary["source_task_pixels"].items()):
        lines.append(f"| {task} | {pixels} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest(
    manifest_path: Path,
    output_path: Path,
    label_subdir: str,
    date: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        patch_id = entry["patch_id"]
        path = f"labels/{label_subdir}/masks/osm_landcover_{date}_{patch_id}.tif"
        entry["worldcover"] = [path]
        entry["osm_landcover"] = [path]
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean.json"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"),
    )
    parser.add_argument("--label-subdir", default="osm_landcover")
    parser.add_argument("--label-date", default="20230101")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    label_root = args.processed_root / "labels"
    output_root = label_root / args.label_subdir
    patch_ids = _patch_ids(label_root)
    if args.limit is not None:
        patch_ids = patch_ids[: args.limit]

    records = []
    for patch_id in patch_ids:
        label, source_pixels = build_label(label_root, patch_id)
        out_path = write_label(label_root, output_root, patch_id, label, args.label_date)
        counts = np.bincount(label.reshape(-1), minlength=max(_class_names()) + 1)
        records.append(
            {
                "patch_id": patch_id,
                "path": str(out_path),
                "total_pixels": int(label.size),
                "class_pixels": {
                    int(class_id): int(counts[class_id]) for class_id in _class_names()
                },
                "source_pixels": source_pixels,
            }
        )

    summary = summarize(records)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_root / "summary.md")
    update_manifest(args.manifest, args.output_manifest, args.label_subdir, args.label_date)
    print(output_root / "metadata.json")
    print(output_root / "summary.md")
    print(args.output_manifest)


if __name__ == "__main__":
    main()
