"""QA: 检查时序影像 patch 与掩膜提取质量。"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS


def _date_from_patch_name(name: str) -> str | None:
    """从文件名中提取第一个 8 位连续数字作为日期。"""
    match = re.search(r"\d{8}", name)
    return match.group(0) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA patch extraction")
    parser.add_argument("--region", required=True, help="区域名，例如 harbin")
    parser.add_argument("--source", required=True, help="数据源，例如 s2")
    parser.add_argument(
        "--processed-root",
        default="/data/xuannv_embedding/processed",
        help="processed 根目录",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="随机抽检 patch 数量",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "QA 报告输出路径"
            "（默认：/data/xuannv_embedding/qa/qa_patch_extraction_{region}_{source}.json）"
        ),
    )
    args = parser.parse_args(argv)

    patch_dir = Path(args.processed_root) / args.region / "patches" / args.source
    if not patch_dir.exists():
        print(f"错误: patch 目录不存在: {patch_dir}", file=sys.stderr)
        return 1

    patch_files = sorted(p for p in patch_dir.glob("*.tif") if not p.name.endswith("_mask.tif"))
    report: dict = {
        "region": args.region,
        "source": args.source,
        "patch_dir": str(patch_dir),
        "total_patches": len(patch_files),
        "counts_per_date": {},
        "mask_coverage": {},
        "sample_checks": [],
        "missing_masks": [],
        "warnings": [],
        "issues": [],
        "ok": True,
    }

    expected_crs: CRS | None = None

    counts_per_date: dict[str, int] = defaultdict(int)
    for p in patch_files:
        date = _date_from_patch_name(p.stem)
        if date is None:
            report["warnings"].append(f"无法从文件名解析日期: {p.name}")
            date = "unknown"
        counts_per_date[date] += 1
    report["counts_per_date"] = dict(counts_per_date)

    # 检查掩膜一一对应
    for p in patch_files:
        mask_path = p.parent / f"{p.stem}_mask.tif"
        if not mask_path.exists():
            report["missing_masks"].append(str(p))
            report["issues"].append(f"缺少掩膜: {p.name}")
            report["ok"] = False

    # 随机抽检
    sample_size = min(args.max_samples, len(patch_files))
    sample_files = random.sample(patch_files, sample_size) if sample_size > 0 else []

    for p in sample_files:
        mask_path = p.parent / f"{p.stem}_mask.tif"
        check = {"patch": str(p), "ok": True, "issues": []}
        try:
            with rasterio.open(p) as src:
                patch = src.read()
                patch_nodata = src.nodata
                patch_crs = src.crs
                patch_shape = patch.shape[-2:]

            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask_crs = src.crs

            # 记录期望 CRS
            if expected_crs is None and patch_crs is not None:
                expected_crs = patch_crs

            if patch_shape != (128, 128):
                check["ok"] = False
                check["issues"].append(f"patch 尺寸 {patch_shape} 不是 128x128")

            if mask.shape != patch_shape:
                check["ok"] = False
                check["issues"].append(f"mask shape {mask.shape} != patch shape {patch_shape}")

            if patch_crs is not None and expected_crs is not None and patch_crs != expected_crs:
                check["ok"] = False
                check["issues"].append(f"patch CRS {patch_crs} 与期望 {expected_crs} 不一致")

            if mask_crs is not None and expected_crs is not None and mask_crs != expected_crs:
                check["ok"] = False
                check["issues"].append(f"mask CRS {mask_crs} 与期望 {expected_crs} 不一致")

            valid_ratio = float(mask.mean())
            check["valid_ratio"] = valid_ratio
            report["mask_coverage"][p.stem] = valid_ratio

            if valid_ratio > 0.0:
                valid_pixels = patch[:, mask > 0]
                if valid_pixels.size > 0:
                    if patch_nodata is not None and np.all(valid_pixels == patch_nodata):
                        check["ok"] = False
                        check["issues"].append("mask 显示有效但 patch 全为 nodata")
                    if not np.isfinite(valid_pixels).any():
                        check["ok"] = False
                        check["issues"].append("mask 显示有效但 patch 无有限值")

            if not check["ok"]:
                report["ok"] = False
                report["issues"].extend(f"{p.name}: {issue}" for issue in check["issues"])
        except Exception as exc:  # noqa: BLE001
            check["ok"] = False
            check["issues"].append(f"读取异常: {exc}")
            report["ok"] = False
            report["issues"].append(f"{p.name}: {exc}")

        report["sample_checks"].append(check)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = (
            Path("/data/xuannv_embedding/qa")
            / f"qa_patch_extraction_{args.region}_{args.source}.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"QA 报告: {out_path}")
    print(f"总 patch 数: {report['total_patches']}")
    print(f"日期数: {len(report['counts_per_date'])}")
    print(f"抽检数: {len(report['sample_checks'])}")
    print(f"缺失掩膜: {len(report['missing_masks'])}")
    print(f"警告数: {len(report['warnings'])}")
    print(f"问题数: {len(report['issues'])}")
    print(f"结果: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
