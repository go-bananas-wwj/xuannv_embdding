from __future__ import annotations

"""Filter cloudy low-resolution observations in a manifest by image quality.

The script keeps the original GeoTIFF files untouched and writes a new manifest
that includes only high-scoring S2/Landsat observations per patch-month.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

DATE_RE = re.compile(r"_(20\d{6})_")


def _date(path: str | Path) -> str | None:
    match = DATE_RE.search(Path(path).name)
    return match.group(1) if match else None


def _reflectance(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanpercentile(finite, 99)) > 10.0:
        return arr / 10000.0
    return arr


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask_path = path.with_name(f"{path.stem}_mask.tif")
    if not mask_path.exists():
        return np.ones(shape, dtype=bool)
    with rasterio.open(mask_path) as src:
        return src.read(1) > 0


def score_observation(root: Path, rel_path: str, source: str) -> dict[str, Any]:
    path = root / rel_path
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    mask = _read_mask(path, arr.shape[-2:])
    finite = np.all(np.isfinite(arr), axis=0)
    valid = mask & finite
    valid_ratio = float(valid.mean())
    if valid.sum() == 0:
        return {"path": rel_path, "score": -999.0, "valid_ratio": valid_ratio}

    if source == "s2":
        refl = _reflectance(arr[:11])
        blue, green, red = refl[0], refl[1], refl[2]
        nir, swir1 = refl[6], refl[9]
        scl = np.nan_to_num(arr[-1], nan=0).astype(np.uint8)
        clear_scl = np.isin(scl, [4, 5, 6])
        questionable_scl = np.isin(scl, [7, 11])
        bad_scl = np.isin(scl, [3, 8, 9, 10])
        strict_clear_ratio = float((clear_scl & valid).sum() / max(1, valid.sum()))
        questionable_ratio = float((questionable_scl & valid).sum() / max(1, valid.sum()))
        bad_scl_ratio = float((bad_scl & mask).sum() / max(1, mask.sum()))
    elif source == "landsat":
        refl = _reflectance(arr[:6])
        blue, green, red = refl[0], refl[1], refl[2]
        nir, swir1 = refl[3], refl[4]
        strict_clear_ratio = valid_ratio
        questionable_ratio = 0.0
        bad_scl_ratio = 0.0
    else:
        raise ValueError(f"Unsupported quality-filtered source: {source}")

    vis = (blue + green + red) / 3.0
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndsi = (green - swir1) / (green + swir1 + 1e-6)
    valid_vis = vis[valid]
    texture = float(np.nanstd(valid_vis))
    bright_ratio = float(((vis > 0.25) & valid).sum() / max(1, valid.sum()))
    cloud_like = float(
        (
            (blue > 0.18)
            & (green > 0.18)
            & (red > 0.18)
            & (ndvi < 0.35)
            & valid
        ).sum()
        / max(1, valid.sum())
    )
    snow_like = float(((ndsi > 0.35) & (green > 0.12) & valid).sum() / max(1, valid.sum()))

    score = (
        1.8 * valid_ratio
        + 1.2 * strict_clear_ratio
        + 1.5 * texture
        - 1.6 * cloud_like
        - 0.7 * bright_ratio
        - 1.2 * bad_scl_ratio
        - 0.4 * questionable_ratio
        - 0.25 * snow_like
    )
    return {
        "path": rel_path,
        "date": _date(rel_path),
        "score": float(score),
        "valid_ratio": valid_ratio,
        "strict_clear_ratio": strict_clear_ratio,
        "cloud_like": cloud_like,
        "bright_ratio": bright_ratio,
        "texture": texture,
        "bad_scl_ratio": bad_scl_ratio,
        "questionable_ratio": questionable_ratio,
        "snow_like": snow_like,
    }


def _filter_source_paths(
    root: Path,
    paths: list[str] | None,
    source: str,
    top_k: int,
    min_score: float,
) -> tuple[list[str] | None, list[dict[str, Any]]]:
    if not paths:
        return None, []
    grouped: dict[str, list[str]] = {}
    for path in paths:
        date = _date(path)
        if date is None:
            continue
        grouped.setdefault(date[:6], []).append(path)

    selected: list[str] = []
    records: list[dict[str, Any]] = []
    for month, month_paths in sorted(grouped.items()):
        scored = [score_observation(root, path, source) for path in month_paths]
        scored = sorted(scored, key=lambda item: item["score"], reverse=True)
        kept = [item for item in scored if item["score"] >= min_score][:top_k]
        if not kept and scored:
            kept = scored[:1]
        kept_paths = {item["path"] for item in kept}
        selected.extend(path for path in month_paths if path in kept_paths)
        for rank, item in enumerate(scored):
            record = dict(item)
            record.update(
                {
                    "source": source,
                    "month": month,
                    "rank": rank,
                    "kept": item["path"] in kept_paths,
                }
            )
            records.append(record)
    return sorted(selected) or None, records


def build_filtered_manifest(
    root: Path,
    input_manifest: Path,
    output_manifest: Path,
    sources: list[str],
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    all_records: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for entry in manifest:
        item = dict(entry)
        for source in sources:
            filtered, records = _filter_source_paths(
                root,
                entry.get(source),
                source,
                top_k=top_k,
                min_score=min_score,
            )
            item[source] = filtered
            for record in records:
                record["patch_id"] = entry["patch_id"]
            all_records.extend(records)
        out.append(item)

    output_manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "sources": sources,
        "top_k": top_k,
        "min_score": min_score,
        "num_entries": len(out),
        "num_scored_observations": len(all_records),
        "num_kept_observations": int(sum(1 for item in all_records if item["kept"])),
    }
    meta_path = output_manifest.with_suffix(".quality_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    records_path = output_manifest.with_suffix(".quality_records.json")
    records_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["s2", "landsat"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=1.35)
    args = parser.parse_args()
    meta = build_filtered_manifest(
        root=args.root,
        input_manifest=args.input,
        output_manifest=args.output,
        sources=args.sources,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
