#!/usr/bin/env python3
"""高分辨率 SAR patches 质量检查。

检查项：
- 每个日期的 patch 数量；
- patch 与 mask 一一配对；
- 有效像素比例分布；
- 归一化后数值范围是否在 [0, 1]；
- 与参考低分辨率 S2 patch 的地理范围一致性；
- 输出尺寸是否符合预期（427×427）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

EXPECTED_SIZE = 427


def parse_highres_filename(path: Path) -> tuple[str, str] | None:
    """解析 highres_sar_YYYYMMDD_{patch_id}.tif 文件名。"""
    if path.name.endswith("_mask.tif"):
        return None
    if not path.name.startswith("highres_sar_"):
        return None
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    date_str = parts[2]
    patch_id = "_".join(parts[3:])
    return date_str, patch_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA for high-res SAR patches")
    parser.add_argument(
        "--processed-root",
        default="/data/xuannv_embedding/processed",
        help="processed 数据根目录",
    )
    parser.add_argument(
        "--region",
        default="haidian",
        help="区域名称",
    )
    parser.add_argument(
        "--output",
        default="/data/xuannv_embedding/qa/qa_highres_sar_haidian.json",
        help="QA 报告输出路径",
    )
    args = parser.parse_args(argv)

    region = args.region
    patch_dir = Path(args.processed_root) / region / "patches" / "highres_sar"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "region": region,
        "patch_dir": str(patch_dir),
        "dates": {},
        "alignment_check": {},
        "issues": [],
    }

    if not patch_dir.exists():
        report["issues"].append(f"Patch directory does not exist: {patch_dir}")
        out_path.write_text(json.dumps(report, indent=2))
        print(f"QA report written to {out_path}")
        return 0

    # 按日期分组统计
    date_groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(patch_dir.glob("highres_sar_*.tif")):
        parsed = parse_highres_filename(p)
        if parsed is None:
            continue
        date_str, _ = parsed
        date_groups[date_str].append(p)

    for date_str, files in sorted(date_groups.items()):
        count = len(files)
        mask_paired = 0
        valid_ratios: list[float] = []
        sizes: set[tuple[int, int]] = set()
        value_mins: list[float] = []
        value_maxs: list[float] = []
        band_counts: set[int] = set()

        for p in files:
            mask_path = p.parent / f"{p.stem}_mask.tif"
            if mask_path.exists():
                mask_paired += 1
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    valid_ratios.append(float((mask == 1).mean()))
            else:
                report["issues"].append(f"Missing mask for {p.name}")

            with rasterio.open(p) as src:
                sizes.add((src.height, src.width))
                band_counts.add(src.count)
                if (src.height, src.width) != (EXPECTED_SIZE, EXPECTED_SIZE):
                    report["issues"].append(
                        f"Unexpected size for {p.name}: {src.height}x{src.width}"
                    )

                data = src.read()
                valid = data != 0
                if valid.any():
                    value_mins.append(float(data[valid].min()))
                    value_maxs.append(float(data[valid].max()))
                else:
                    report["issues"].append(f"No valid pixels in {p.name}")

        date_report: dict[str, Any] = {
            "count": count,
            "mask_paired": mask_paired,
            "sizes": [list(s) for s in sorted(sizes)],
            "band_counts": sorted(band_counts),
        }
        if valid_ratios:
            date_report["valid_ratio_min"] = float(np.min(valid_ratios))
            date_report["valid_ratio_max"] = float(np.max(valid_ratios))
            date_report["valid_ratio_mean"] = float(np.mean(valid_ratios))
            date_report["valid_ratio_median"] = float(np.median(valid_ratios))
        if value_mins:
            date_report["value_min"] = float(np.min(value_mins))
            date_report["value_max"] = float(np.max(value_maxs))
            date_report["value_min_list"] = [float(v) for v in value_mins]
            date_report["value_max_list"] = [float(v) for v in value_maxs]
            if all(m == 1.0 for m in value_mins) and all(m == 1.0 for m in value_maxs):
                report["issues"].append(
                    f"Date {date_str}: all sampled patches have value_min == value_max == 1.0 "
                    "(likely clipped dB normalization)"
                )
        report["dates"][date_str] = date_report

    # 与参考低分辨率 S2 patch 对齐检查
    ref_dir = Path(args.processed_root) / region / "patches" / "s2"
    if ref_dir.exists():
        sample_pairs: list[tuple[str, Path, Path]] = []
        for p in sorted(patch_dir.glob("highres_sar_*.tif"))[:10]:
            parsed = parse_highres_filename(p)
            if parsed is None:
                continue
            _, patch_id = parsed
            candidates = list(ref_dir.glob(f"s2_*_{patch_id}.tif"))
            candidates = [c for c in candidates if not c.name.endswith("_mask.tif")]
            if candidates:
                sample_pairs.append((patch_id, p, candidates[0]))

        aligned_count = 0
        mismatched: list[dict[str, Any]] = []
        for patch_id, hp, rp in sample_pairs:
            with rasterio.open(hp) as hsrc, rasterio.open(rp) as rsrc:
                hb = hsrc.bounds
                rb = rsrc.bounds
                # SAR patch 使用 427×427 像素覆盖 1280 m，可能与 S2 128×128 存在最多 1 像素（3 m）差异
                tol = 5.0
                ok = (
                    abs(hb.left - rb.left) <= tol
                    and abs(hb.bottom - rb.bottom) <= tol
                    and abs(hb.right - rb.right) <= tol
                    and abs(hb.top - rb.top) <= tol
                )
                if ok:
                    aligned_count += 1
                else:
                    mismatched.append(
                        {
                            "patch_id": patch_id,
                            "highres_bounds": list(hb),
                            "ref_bounds": list(rb),
                        }
                    )

        report["alignment_check"] = {
            "sampled": len(sample_pairs),
            "aligned": aligned_count,
            "mismatched": mismatched,
        }
    else:
        report["alignment_check"] = {
            "sampled": 0,
            "note": "No reference S2 patches found",
        }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"QA report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
