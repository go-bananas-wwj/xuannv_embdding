from __future__ import annotations

"""Build the Haidian P6A six-month manifest used before training.

The manifest keeps only 202512-202605 observations. For high-res optical, it
uses the explicit 20260501 alias files that point to 20260430 patches, so the
month-end optical image supervises the May embedding without adding hidden
date-remapping logic to the dataset.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")
HIGHRES_OPTICAL_DATES = (
    "20251201",
    "20260101",
    "20260201",
    "20260301",
    "20260401",
    "20260501",
)
DATE_RE = re.compile(r"_(20\d{6})_")


def _date_from_path(path: str) -> str | None:
    match = DATE_RE.search(Path(path).name)
    return match.group(1) if match else None


def _filter_months(paths: list[str] | None, months: set[str]) -> list[str] | None:
    if not paths:
        return None
    out = []
    for path in paths:
        date = _date_from_path(path)
        if date is not None and date[:6] in months:
            out.append(path)
    return sorted(out) or None


def _highres_optical_paths(patch_id: str) -> list[str]:
    return [
        f"patches/highres_optical/highres_optical_{date}_{patch_id}.tif"
        for date in HIGHRES_OPTICAL_DATES
    ]


def _validate_paths(root: Path, paths: list[str] | None, missing: list[str]) -> None:
    for rel in paths or []:
        full = root / rel
        if not full.exists():
            missing.append(str(full))
        mask = full.with_name(f"{full.stem}_mask.tif")
        if not mask.exists():
            missing.append(str(mask))


def build_manifest(root: Path, base_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with base_manifest.open("r", encoding="utf-8") as f:
        base = json.load(f)

    months = set(MONTHS)
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    coverage: dict[str, dict[str, int]] = {
        source: {month: 0 for month in MONTHS}
        for source in (
            "s2",
            "s1",
            "landsat",
            "highres_optical_haidian",
            "highres_sar_haidian",
        )
    }

    for entry in base:
        patch_id = entry["patch_id"]
        item: dict[str, Any] = {"patch_id": patch_id}
        item["s2"] = _filter_months(entry.get("s2"), months)
        item["s1"] = _filter_months(entry.get("s1"), months)
        item["landsat"] = _filter_months(entry.get("landsat"), months)
        item["worldcover"] = entry.get("worldcover")
        item["highres_optical_haidian"] = _highres_optical_paths(patch_id)
        item["highres_sar_haidian"] = _filter_months(
            entry.get("highres_sar_haidian"), months
        )

        for source, paths in item.items():
            if source in ("patch_id", "worldcover"):
                continue
            _validate_paths(root, paths, missing)
            seen_months = {
                date[:6] for path in (paths or []) if (date := _date_from_path(path))
            }
            for month in seen_months:
                if month in coverage[source]:
                    coverage[source][month] += 1
        _validate_paths(root, item.get("worldcover"), missing)
        out.append(item)

    meta = {
        "months": list(MONTHS),
        "highres_optical_dates": list(HIGHRES_OPTICAL_DATES),
        "num_entries": len(out),
        "coverage_patches_with_month": coverage,
        "missing_file_count": len(missing),
        "missing_files": missing[:200],
    }
    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian"),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("/data/xuannv_embedding/processed/haidian/manifest_stage2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/processed/haidian/"
            "manifest_p6a_202512_202605_pixelmask.json"
        ),
    )
    args = parser.parse_args()

    manifest, meta = build_manifest(args.root, args.base_manifest)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if meta["missing_file_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
