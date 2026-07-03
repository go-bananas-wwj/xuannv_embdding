from __future__ import annotations

"""Audit preprocessed ``*_mask.tif`` files against their paired imagery."""

import argparse
from pathlib import Path

import numpy as np
import rasterio


def _image_files(source_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        path for path in source_dir.glob("*.tif") if not path.name.endswith("_mask.tif")
    )
    if limit is None or len(files) <= limit:
        return files
    step = max(1, len(files) // limit)
    return files[::step][:limit]


def _audit_source(patch_root: Path, source: str, limit: int | None) -> None:
    source_dir = patch_root / source
    files = _image_files(source_dir, limit)
    if not files:
        print(f"{source}: no images found in {source_dir}")
        return

    ratios: list[float] = []
    values: set[int] = set()
    missing = 0
    shape_bad = 0
    nonbinary = 0
    nodata_in_valid = 0
    zero_masks = 0
    full_masks = 0
    examples: list[str] = []

    for image_path in files:
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")
        if not mask_path.exists():
            missing += 1
            examples.append(f"missing:{image_path.name}")
            continue

        with rasterio.open(image_path) as src:
            ref = src.read(1)
            image_shape = (src.height, src.width)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if mask.shape != image_shape:
            shape_bad += 1
            examples.append(f"shape:{image_path.name}")
            continue

        unique = np.unique(mask)
        values.update(int(v) for v in unique[:8])
        if not np.isin(unique, [0, 1]).all():
            nonbinary += 1
            examples.append(f"nonbinary:{mask_path.name}")

        binary = mask > 0
        ratio = float(binary.mean())
        ratios.append(ratio)
        zero_masks += int(ratio == 0.0)
        full_masks += int(ratio == 1.0)
        if bool(((~np.isfinite(ref)) & binary).any()):
            nodata_in_valid += 1
            examples.append(f"nodata_valid:{image_path.name}")

    q = np.quantile(ratios, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    q_str = " ".join(f"{v:.3f}" for v in q)
    print(f"\n{source}")
    print(f"  sampled={len(files)} missing={missing} values={sorted(values)}")
    print(
        "  "
        f"shape_bad={shape_bad} nonbinary={nonbinary} "
        f"nodata_in_valid={nodata_in_valid}"
    )
    print(f"  valid_ratio_quantiles={q_str}")
    print(f"  zero_masks={zero_masks} full_masks={full_masks}")
    if examples:
        print(f"  examples={examples[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-root",
        type=Path,
        required=True,
        help="Directory containing source subdirectories, e.g. processed/haidian/patches.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["s2", "landsat", "s1", "highres_optical", "highres_sar"],
    )
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    for source in args.sources:
        _audit_source(args.patch_root, source, args.limit)


if __name__ == "__main__":
    main()
