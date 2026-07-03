from __future__ import annotations

"""Create lightweight date aliases for high-res optical patches.

This is useful when a month-end image should supervise the following month,
for example mapping PlanetScope 20260430 patches to training month 202605.
The script creates symlinks for both image and ``*_mask.tif`` files.
"""

import argparse
from pathlib import Path


def _replace_date(name: str, source_date: str, target_date: str) -> str:
    if source_date not in name:
        raise ValueError(f"{source_date} not found in {name}")
    return name.replace(source_date, target_date, 1)


def _link_pair(source_path: Path, target_path: Path, overwrite: bool) -> str:
    if target_path.exists() or target_path.is_symlink():
        if not overwrite:
            return "exists"
        target_path.unlink()
    target_path.symlink_to(source_path.name)
    return "linked"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-dir",
        type=Path,
        required=True,
        help="Directory containing highres_optical_YYYYMMDD_patch_xxx.tif files.",
    )
    parser.add_argument("--source-date", required=True, help="Existing YYYYMMDD date.")
    parser.add_argument("--target-date", required=True, help="Alias YYYYMMDD date.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    patch_dir = args.patch_dir
    source_date = args.source_date
    target_date = args.target_date
    image_files = sorted(
        p
        for p in patch_dir.glob(f"highres_optical_{source_date}_*.tif")
        if not p.name.endswith("_mask.tif")
    )
    if not image_files:
        raise FileNotFoundError(f"No highres optical patches found for {source_date}")

    counts = {"linked": 0, "exists": 0, "missing_mask": 0}
    for source_path in image_files:
        target_path = source_path.with_name(
            _replace_date(source_path.name, source_date, target_date)
        )
        status = _link_pair(source_path, target_path, args.overwrite)
        counts[status] += 1

        source_mask = source_path.with_name(f"{source_path.stem}_mask.tif")
        if not source_mask.exists():
            counts["missing_mask"] += 1
            continue
        target_mask = source_mask.with_name(
            _replace_date(source_mask.name, source_date, target_date)
        )
        status = _link_pair(source_mask, target_mask, args.overwrite)
        counts[status] += 1

    print(
        f"source_date={source_date} target_date={target_date} "
        f"patches={len(image_files)} counts={counts}"
    )


if __name__ == "__main__":
    main()
