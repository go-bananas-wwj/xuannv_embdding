#!/usr/bin/env python3
"""Build selected nationwide 1,280 m UTM grid partitions without materializing all records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from china_full_grid import (
    GridSpec,
    audit_grid_package,
    enumerate_macro_patch_records,
    read_sampled_registry_jsonl,
    write_grid_package_audit,
    write_zone_records,
)
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import box
from shapely.ops import transform as transform_geometry

ROOT = Path(__file__).resolve().parents[2]
DOCX_BUILDER_PATH = ROOT / "scripts/docs/build_china_full_grid_package_docx.py"
PACKAGE_README_SOURCE = ROOT / "docs/data/china_full_1280m_grid_package_readme_20260805.md"
PACKAGE_MANIFEST_NAME = "china_full_1280m_grid_package_manifest.json"
MACROCELL_INDEX_NAME = "china_full_1280m_macrocell_index.gpkg"
PACKAGE_PREVIEWS = {
    "national_density": "national_patch_density_by_utm_zone.png",
    "local_grid": "local_1280m_grid_sampled_unsampled.png",
    "utm_seams": "utm_owner_zone_seams.png",
}
MEMBERSHIP_AUDIT_SCHEMA_VERSION = "china_full_1280m_membership_audit_v1"
MEMBERSHIP_AUDIT_COUNT_FIELDS = (
    "all_count",
    "sampled_count",
    "unsampled_count",
    "matched",
    "missing_sampled_count",
    "duplicate_parent_key_count",
    "sampled_unsampled_intersection_count",
    "sampled_flag_mismatch_count",
    "partition_mismatch_count",
    "exact_partition_membership_mismatch_count",
    "child_partition_unknown_parent_key_count",
    "child_partition_metadata_mismatch_count",
    "child_partition_geometry_mismatch_count",
    "footprint_coordinate_mismatch_count",
    "stored_utm_bounds_mismatch_count",
    "stored_wgs84_bounds_mismatch_count",
    "invalid_geometry_count",
    "owner_zone_mismatch_count",
    "same_zone_positive_overlap_count",
    "cross_zone_overlap_violation_count",
)
MEMBERSHIP_AUDIT_FAILURE_COUNT_FIELDS = MEMBERSHIP_AUDIT_COUNT_FIELDS[4:]
MEMBERSHIP_AUDIT_HASH_FIELDS = (
    "identity_hash",
    "footprint_hash",
    "sampled_registry_footprint_hash",
)


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


def validate_membership_audit(audit: Any) -> dict[str, Any]:
    """Reject incomplete or failed membership audits before package publication starts."""
    if not isinstance(audit, dict):
        raise ValueError("membership audit must be a JSON object")
    if audit.get("schema_version") != MEMBERSHIP_AUDIT_SCHEMA_VERSION:
        raise ValueError("membership audit has an unsupported schema_version")
    if audit.get("passed") is not True:
        raise ValueError("membership audit passed must be exactly True")

    missing = [field for field in MEMBERSHIP_AUDIT_COUNT_FIELDS if field not in audit]
    if missing:
        raise ValueError(f"membership audit is missing required fields: {', '.join(missing)}")
    for field in MEMBERSHIP_AUDIT_COUNT_FIELDS:
        value = audit[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"membership audit {field} must be a non-negative integer")

    if audit["all_count"] != audit["sampled_count"] + audit["unsampled_count"]:
        raise ValueError("membership audit counts do not partition the parent atlas")
    if audit["matched"] != audit["sampled_count"]:
        raise ValueError("membership audit matched count must equal sampled_count")
    if any(audit[field] != 0 for field in MEMBERSHIP_AUDIT_FAILURE_COUNT_FIELDS):
        raise ValueError("membership audit passed status conflicts with non-zero failure counts")

    hash_mismatches = audit.get("hash_mismatches")
    if not isinstance(hash_mismatches, dict):
        raise ValueError("membership audit hash_mismatches must be an object")
    missing_hash_fields = [
        field for field in MEMBERSHIP_AUDIT_HASH_FIELDS if field not in hash_mismatches
    ]
    if missing_hash_fields:
        raise ValueError(
            "membership audit hash_mismatches is missing required fields: "
            f"{', '.join(missing_hash_fields)}"
        )
    for field in MEMBERSHIP_AUDIT_HASH_FIELDS:
        value = hash_mismatches[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"membership audit hash_mismatches.{field} must be zero")
    return audit


def _zone_records(
    macros: Iterable[dict[str, Any]], boundary_wgs84: Any, spec: GridSpec, grid_id: str
):
    for macro in macros:
        if str(macro["grid_id"]) == grid_id:
            yield from enumerate_macro_patch_records(macro, boundary_wgs84, spec)


def _iter_all_partition_rows(output_root: Path, columns: list[str]):
    """Yield selected fields from canonical GeoParquet without loading the atlas at once."""
    import pyarrow.parquet as pq

    paths = sorted((output_root / "all").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"no all-partition GeoParquet files found below {output_root}")
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        missing = set(columns) - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing package-index columns: {sorted(missing)}")
        for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
            yield from batch.to_pylist()


def _macrocell_wgs84_geometry(grid_epsg: int, macro_col: int, macro_row: int):
    side_m = GridSpec().side_m * GridSpec().macro_side_patches
    to_wgs84 = Transformer.from_crs(grid_epsg, 4326, always_xy=True)
    return transform_geometry(
        to_wgs84.transform,
        box(
            macro_col * side_m,
            macro_row * side_m,
            (macro_col + 1) * side_m,
            (macro_row + 1) * side_m,
        ),
    )


def summarize_grid_partitions(
    output_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exact macrocell and UTM counts from the written all GeoParquet partition."""
    macro_counts: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    zone_counts: dict[tuple[str, int], dict[str, Any]] = {}
    columns = ["macro_id", "grid_id", "grid_epsg", "grid_col", "grid_row", "sampled"]
    for row in _iter_all_partition_rows(Path(output_root), columns):
        grid_id = str(row["grid_id"])
        grid_epsg = int(row["grid_epsg"])
        macro_col = int(row["grid_col"]) // GridSpec().macro_side_patches
        macro_row = int(row["grid_row"]) // GridSpec().macro_side_patches
        macro_key = (str(row["macro_id"]), grid_epsg, macro_col, macro_row)
        macro = macro_counts.setdefault(
            macro_key,
            {
                "macro_id": macro_key[0],
                "grid_id": grid_id,
                "grid_epsg": grid_epsg,
                "macro_col": macro_col,
                "macro_row": macro_row,
                "all_count": 0,
                "sampled_count": 0,
                "unsampled_count": 0,
            },
        )
        zone = zone_counts.setdefault(
            (grid_id, grid_epsg),
            {
                "grid_id": grid_id,
                "grid_epsg": grid_epsg,
                "all_count": 0,
                "sampled_count": 0,
                "unsampled_count": 0,
            },
        )
        for summary in (macro, zone):
            summary["all_count"] += 1
            summary["sampled_count" if bool(row["sampled"]) else "unsampled_count"] += 1

    macros = sorted(
        macro_counts.values(),
        key=lambda row: (row["grid_epsg"], row["macro_row"], row["macro_col"]),
    )
    zones = sorted(zone_counts.values(), key=lambda row: row["grid_epsg"])
    if sum(row["all_count"] for row in macros) != sum(row["all_count"] for row in zones):
        raise ValueError("macrocell and UTM zone all counts disagree")
    return macros, zones


def write_macrocell_index(
    output_path: str | Path,
    macrocell_summaries: list[dict[str, Any]],
    zone_summaries: list[dict[str, Any]],
    boundary_wgs84: Any,
) -> Path:
    """Write macrocell and owner-zone count layers to a single WGS84 GeoPackage."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite macrocell index: {output_path}")
    if not macrocell_summaries or not zone_summaries:
        raise ValueError("macrocell index requires non-empty macrocell and UTM zone summaries")

    macro_rows = []
    for summary in macrocell_summaries:
        row = dict(summary)
        if row["all_count"] != row["sampled_count"] + row["unsampled_count"]:
            raise ValueError(f"invalid macrocell partition counts: {row['macro_id']}")
        row["geometry"] = _macrocell_wgs84_geometry(
            int(row["grid_epsg"]), int(row["macro_col"]), int(row["macro_row"])
        )
        macro_rows.append(row)

    zone_rows = []
    for summary in zone_summaries:
        row = dict(summary)
        if row["all_count"] != row["sampled_count"] + row["unsampled_count"]:
            raise ValueError(f"invalid UTM zone partition counts: {row['grid_id']}")
        zone_number = int(row["grid_epsg"]) - 32600
        west = -180 + (zone_number - 1) * 6
        zone_geometry = boundary_wgs84.intersection(box(west, -80, west + 6, 84))
        if zone_geometry.is_empty:
            raise ValueError(f"owner-zone boundary has no geometry for {row['grid_id']}")
        row["macrocell_count"] = sum(
            1
            for macro in macrocell_summaries
            if macro["grid_id"] == row["grid_id"] and macro["grid_epsg"] == row["grid_epsg"]
        )
        row["geometry"] = zone_geometry
        zone_rows.append(row)

    macro_frame = gpd.GeoDataFrame(macro_rows, geometry="geometry", crs="EPSG:4326")
    zone_frame = gpd.GeoDataFrame(zone_rows, geometry="geometry", crs="EPSG:4326")
    macro_frame.rename(
        columns={
            "macro_id": "MACRO_ID",
            "grid_id": "GRID_ID",
            "grid_epsg": "UTM_EPSG",
            "macro_col": "MACRO_COL",
            "macro_row": "MACRO_ROW",
            "all_count": "ALL_COUNT",
            "sampled_count": "SAMPLED",
            "unsampled_count": "UNSAMPLED",
        }
    ).to_file(output_path, layer="macrocells", driver="GPKG", index=False)
    zone_frame.rename(
        columns={
            "grid_id": "GRID_ID",
            "grid_epsg": "UTM_EPSG",
            "all_count": "ALL_COUNT",
            "sampled_count": "SAMPLED",
            "unsampled_count": "UNSAMPLED",
            "macrocell_count": "MACROCELLS",
        }
    ).to_file(output_path, layer="utm_zones", driver="GPKG", index=False)
    return output_path


def _preview_local_grid_records(output_root: Path, limit: int = 100) -> gpd.GeoDataFrame:
    """Select a bounded macrocell that contains both membership classes when available."""
    columns = ["macro_id", "sampled", "geometry"]
    current_macro_id: str | None = None
    records: list[dict[str, Any]] = []
    for row in _iter_all_partition_rows(output_root, columns):
        macro_id = str(row["macro_id"])
        if current_macro_id is None:
            current_macro_id = macro_id
        if macro_id != current_macro_id:
            if {bool(record["sampled"]) for record in records} == {False, True}:
                return gpd.GeoDataFrame(
                    records,
                    geometry=[from_wkb(record["geometry"]) for record in records],
                    crs="EPSG:4326",
                )
            current_macro_id = macro_id
            records = []
        if len(records) < limit:
            records.append(row)
        if len(records) == limit and {bool(record["sampled"]) for record in records} == {
            False,
            True,
        }:
            return gpd.GeoDataFrame(
                records,
                geometry=[from_wkb(record["geometry"]) for record in records],
                crs="EPSG:4326",
            )
    if {bool(record["sampled"]) for record in records} != {False, True}:
        raise ValueError("no bounded macrocell preview contains both sampled and unsampled patches")
    return gpd.GeoDataFrame(
        records,
        geometry=[from_wkb(record["geometry"]) for record in records],
        crs="EPSG:4326",
    )


def render_package_previews(output_root: str | Path, index_path: str | Path) -> dict[str, Path]:
    """Render the three package visual checks using bounded reads from canonical outputs."""
    output_root = Path(output_root)
    index_path = Path(index_path)
    zone_frame = gpd.read_file(index_path, layer="utm_zones")
    macro_frame = gpd.read_file(index_path, layer="macrocells")
    previews = {name: output_root / filename for name, filename in PACKAGE_PREVIEWS.items()}

    figure, axis = plt.subplots(figsize=(10, 7), dpi=180, facecolor="white")
    zone_frame.plot(ax=axis, column="ALL_COUNT", cmap="YlGnBu", edgecolor="black", linewidth=0.7)
    for _, row in zone_frame.iterrows():
        point = row.geometry.representative_point()
        axis.text(
            point.x,
            point.y,
            f"{row['GRID_ID']}\n{int(row['ALL_COUNT']):,}",
            ha="center",
            va="center",
            fontsize=8,
        )
    axis.set_title("National parent-patch density by UTM owner zone")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(previews["national_density"], facecolor="white")
    plt.close(figure)

    local_frame = _preview_local_grid_records(output_root)
    figure, axis = plt.subplots(figsize=(8, 8), dpi=180, facecolor="white")
    local_frame[~local_frame["sampled"]].plot(
        ax=axis, color="#d9d9d9", edgecolor="#777777", linewidth=0.35, label="Unsampled"
    )
    local_frame[local_frame["sampled"]].plot(
        ax=axis, color="#2166ac", edgecolor="#111111", linewidth=0.45, label="Sampled"
    )
    axis.legend(loc="upper right")
    axis.set_aspect("equal")
    axis.set_title("Local 1,280 m parent grid membership")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(previews["local_grid"], facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 7), dpi=180, facecolor="white")
    zone_frame.boundary.plot(ax=axis, color="#222222", linewidth=1.2)
    macro_frame.boundary.plot(ax=axis, color="#7f7f7f", linewidth=0.15, alpha=0.7)
    for _, row in zone_frame.iterrows():
        point = row.geometry.representative_point()
        axis.text(point.x, point.y, str(row["GRID_ID"]), ha="center", va="center", fontsize=8)
    axis.text(
        0.01,
        0.01,
        "Owner zone: center longitude in half-open UTM band; cross-zone overlap <= 1%",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "black", "pad": 3},
    )
    axis.set_title("UTM owner-zone seams and accepted overlap rule")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(previews["utm_seams"], facecolor="white")
    plt.close(figure)
    return previews


def _load_docx_builder():
    spec = importlib.util.spec_from_file_location("china_full_grid_package_docx", DOCX_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DOCX builder: {DOCX_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_package_artifacts(
    *,
    output_root: str | Path,
    boundary_wgs84: Any,
    membership_audit_path: str | Path,
    prior_sampling_source: str | Path,
    package_readme_source: str | Path = PACKAGE_README_SOURCE,
) -> Path:
    """Build index, previews, README, DOCX, and manifest after all partitions are written."""
    output_root = Path(output_root)
    audit_path = Path(membership_audit_path)
    audit = validate_membership_audit(_read_json(audit_path))
    macros, zones = summarize_grid_partitions(output_root)
    counts = {
        "all": sum(row["all_count"] for row in zones),
        "sampled": sum(row["sampled_count"] for row in zones),
        "unsampled": sum(row["unsampled_count"] for row in zones),
    }
    if counts["all"] != counts["sampled"] + counts["unsampled"]:
        raise ValueError("grid-package counts do not partition the parent atlas")
    for audit_key, count_key in (
        ("all_count", "all"),
        ("sampled_count", "sampled"),
        ("unsampled_count", "unsampled"),
    ):
        if audit_key in audit and int(audit[audit_key]) != counts[count_key]:
            raise ValueError(f"membership audit {audit_key} does not match generated partitions")
    if sum(row["all_count"] for row in macros) != counts["all"]:
        raise ValueError("macrocell all counts do not equal GeoParquet all count")

    index_path = write_macrocell_index(
        output_root / MACROCELL_INDEX_NAME, macros, zones, boundary_wgs84
    )
    previews = render_package_previews(output_root, index_path)
    readme_source = Path(package_readme_source)
    if not readme_source.exists():
        raise FileNotFoundError(f"package README source does not exist: {readme_source}")
    readme_path = output_root / "README.md"
    shutil.copyfile(readme_source, readme_path)
    docx_path = output_root / "先读我.docx"
    _load_docx_builder().build_package_docx(
        docx_path,
        package_metadata={
            "all_count": counts["all"],
            "sampled_count": counts["sampled"],
            "unsampled_count": counts["unsampled"],
            "membership_audit": audit,
        },
        prior_sampling_source=prior_sampling_source,
    )
    manifest_path = output_root / PACKAGE_MANIFEST_NAME
    manifest = {
        "schema_version": "china_full_1280m_grid_package_v1",
        "counts": counts,
        "macrocell_index": index_path.name,
        "membership_audit": audit,
        "zone_summaries": zones,
        "previews": {name: path.name for name, path in previews.items()},
        "documents": {"readme": readme_path.name, "docx": docx_path.name},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, help="WGS84 boundary vector file")
    parser.add_argument("--macrocells", type=Path, help="Task 1 macrocell JSON")
    parser.add_argument("--sampled-parent-keys", type=Path, help="62k sampled key map JSON")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grid-id", help="One UTM grid_id to write")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--build-package-artifacts",
        action="store_true",
        help="Build index, previews, documents, and manifest from already-written partitions",
    )
    parser.add_argument(
        "--membership-audit",
        type=Path,
        help="Membership audit JSON used by --build-package-artifacts",
    )
    parser.add_argument(
        "--prior-sampling-source",
        type=Path,
        default=ROOT / "docs/data/先读我_中国季度嵌入采样说明_20260727.md",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--sampled-registry", type=Path, help="Sampled registry JSONL for audit")
    parser.add_argument(
        "--audit-output",
        type=Path,
        help=(
            "Audit JSON destination "
            "(default: <output-root>/china_full_grid_membership_audit.json)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.build_package_artifacts:
        if args.boundary is None or args.membership_audit is None:
            raise ValueError("--build-package-artifacts requires --boundary and --membership-audit")
        boundary = gpd.read_file(args.boundary).to_crs(4326).geometry.unary_union
        manifest_path = build_package_artifacts(
            output_root=args.output_root,
            boundary_wgs84=boundary,
            membership_audit_path=args.membership_audit,
            prior_sampling_source=args.prior_sampling_source,
        )
        print(manifest_path)
        return 0
    if args.audit_only:
        if args.sampled_registry is None:
            raise ValueError("--audit-only requires --sampled-registry")
        audit = audit_grid_package(
            args.output_root,
            read_sampled_registry_jsonl(args.sampled_registry),
            batch_size=args.batch_size,
        )
        audit_path = write_grid_package_audit(
            audit,
            args.audit_output or args.output_root / "china_full_grid_membership_audit.json",
        )
        print(
            json.dumps({"audit_path": str(audit_path), **audit}, ensure_ascii=False, sort_keys=True)
        )
        return 0 if audit["passed"] else 2
    if None in (args.boundary, args.macrocells, args.sampled_parent_keys, args.grid_id):
        raise ValueError(
            "grid generation requires --boundary, --macrocells, --sampled-parent-keys, "
            "and --grid-id"
        )
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
