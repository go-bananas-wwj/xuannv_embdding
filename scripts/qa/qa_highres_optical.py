#!/usr/bin/env python3
"""高分辨率光学 patches 质量检查。

检查项：
- 每月 patch 数量；
- patch 与 mask 一一配对；
- 有效像素比例分布；
- 与参考低分辨率 S2 patch 的地理范围一致性；
- 输出尺寸是否符合预期。
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

EXPECTED_SIZES = {
    "harbin": 2560,
    "haidian": 427,
}


def parse_highres_filename(path: Path) -> tuple[str, str] | None:
    """解析 highres_optical_YYYYMMDD_{patch_id}.tif 文件名。"""
    if path.name.endswith("_mask.tif"):
        return None
    if not path.name.startswith("highres_optical_"):
        return None
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    date_str = parts[2]
    patch_id = "_".join(parts[3:])
    return date_str, patch_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA for high-res optical patches")
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--processed-root",
        default="/data/xuannv_embedding/processed",
        help="processed 数据根目录",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="QA 报告输出路径（默认 /data/xuannv_embedding/qa/qa_highres_optical_{region}.json）",
    )
    args = parser.parse_args(argv)

    region = args.region
    patch_dir = Path(args.processed_root) / region / "patches" / "highres_optical"
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"/data/xuannv_embedding/qa/qa_highres_optical_{region}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "region": region,
        "patch_dir": str(patch_dir),
        "months": {},
        "alignment_check": {},
        "issues": [],
    }

    if not patch_dir.exists():
        report["issues"].append(f"Patch directory does not exist: {patch_dir}")
        out_path.write_text(json.dumps(report, indent=2))
        print(f"QA report written to {out_path}")
        return 0

    # 按月份分组统计
    month_groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(patch_dir.glob("highres_optical_*.tif")):
        parsed = parse_highres_filename(p)
        if parsed is None:
            continue
        date_str, _ = parsed
        month_key = date_str[:6]
        month_groups[month_key].append(p)

    for month_key, files in sorted(month_groups.items()):
        count = len(files)
        mask_paired = 0
        valid_ratios: list[float] = []
        sizes: set[tuple[int, int]] = set()
        for p in files:
            mask_path = p.parent / f"{p.stem}_mask.tif"
            if mask_path.exists():
                mask_paired += 1
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    valid_ratios.append(float((mask == 1).mean()))
            else:
                report["issues"].append(
                    f"Missing mask for {p.name}"
                )

            with rasterio.open(p) as src:
                sizes.add((src.height, src.width))
                if (src.height, src.width) != (EXPECTED_SIZES[region], EXPECTED_SIZES[region]):
                    report["issues"].append(
                        f"Unexpected size for {p.name}: {src.height}x{src.width}"
                    )

        month_report: dict[str, Any] = {
            "count": count,
            "mask_paired": mask_paired,
            "sizes": [list(s) for s in sorted(sizes)],
        }
        if valid_ratios:
            month_report["valid_ratio_min"] = float(np.min(valid_ratios))
            month_report["valid_ratio_max"] = float(np.max(valid_ratios))
            month_report["valid_ratio_mean"] = float(np.mean(valid_ratios))
            month_report["valid_ratio_median"] = float(np.median(valid_ratios))
        report["months"][month_key] = month_report

    # 与参考低分辨率 S2 patch 对齐检查
    ref_dir = Path(args.processed_root) / region / "patches" / "s2"
    if ref_dir.exists():
        # 取一个高分辨率 patch 样本，找同 patch_id 的 S2 patch 比较 bounds
        sample_pairs: list[tuple[str, Path, Path]] = []
        for p in sorted(patch_dir.glob("highres_optical_*.tif"))[:10]:
            parsed = parse_highres_filename(p)
            if parsed is None:
                continue
            _, patch_id = parsed
            # S2 patch 文件名形如 s2_YYYYMMDD_{patch_id}.tif
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
                tol = 10.0  # 允许低分辨率网格与 AOI 网格存在半像素（5 m）原点差异
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
