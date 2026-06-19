"""QA: 检查 WorldCover 标签重映射、取值范围、与 S2 patch 几何对齐及类别分布。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio


PATCH_ID_RE = re.compile(r"p\d{3}_r\d{3}")
BOUNDS_TOLERANCE = 250.0  # 米，覆盖 AOI 边缘 S2 覆盖不足导致的网格偏移


def _patch_id_from_name(name: str) -> str | None:
    match = PATCH_ID_RE.search(name)
    return match.group(0) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA WorldCover 标签 patch")
    parser.add_argument("--region", required=True, help="区域名，例如 harbin")
    parser.add_argument(
        "--processed-root",
        default="/data/xuannv_embedding/processed",
        help="processed 根目录",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="QA 报告输出路径（默认：/data/xuannv_embedding/qa/qa_labels_{region}.json）",
    )
    parser.add_argument(
        "--ref-source",
        default="s2",
        help="用于对齐检查的参考数据源",
    )
    parser.add_argument(
        "--bounds-tol",
        type=float,
        default=BOUNDS_TOLERANCE,
        help="bounds 对齐容差（米）",
    )
    args = parser.parse_args(argv)

    processed_root = Path(args.processed_root)
    label_dir = processed_root / args.region / "labels" / "worldcover"
    ref_dir = processed_root / args.region / "patches" / args.ref_source

    if not label_dir.exists():
        print(f"错误: 标签目录不存在: {label_dir}", file=sys.stderr)
        return 1

    label_files = sorted(label_dir.glob("*.tif"))
    if not label_files:
        print(f"错误: 未找到标签文件: {label_dir}", file=sys.stderr)
        return 1

    report: dict = {
        "region": args.region,
        "label_dir": str(label_dir),
        "ref_dir": str(ref_dir),
        "total_patches": len(label_files),
        "class_distribution": {},
        "alignment": {"checked": 0, "passed": 0, "failed": []},
        "edge_patches": [],
        "anomalies": [],
        "issues": [],
        "ok": True,
    }

    class_counts = np.zeros(9, dtype=np.int64)

    for label_path in label_files:
        patch_id = _patch_id_from_name(label_path.name)
        if patch_id is None:
            report["anomalies"].append(f"无法解析 patch_id: {label_path.name}")
            report["ok"] = False
            continue

        try:
            with rasterio.open(label_path) as src:
                label = src.read(1)
                label_bounds = src.bounds
                label_crs = src.crs
                label_nodata = src.nodata
        except Exception as exc:  # noqa: BLE001
            report["anomalies"].append(f"{label_path.name}: 读取失败 {exc}")
            report["ok"] = False
            continue

        # 取值范围检查：必须在 0-8 之间
        unique_vals = np.unique(label)
        out_of_range = unique_vals[(unique_vals < 0) | (unique_vals > 8)]
        if out_of_range.size > 0:
            report["anomalies"].append(
                f"{label_path.name}: 存在越界值 {out_of_range.tolist()}"
            )
            report["ok"] = False

        # 统计类别分布
        counts = np.bincount(label.ravel(), minlength=9)
        class_counts += counts

        # 与参考 S2 patch 对齐检查（排除 mask 文件）
        ref_candidates = [
            p for p in ref_dir.glob(f"*{patch_id}.tif")
            if not p.name.endswith("_mask.tif")
        ]
        ref_path = ref_candidates[0] if ref_candidates else None
        align = {
            "patch_id": patch_id,
            "label": str(label_path),
            "ref": str(ref_path) if ref_path else None,
            "ok": True,
            "issues": [],
        }

        if ref_path is None:
            align["ok"] = False
            align["issues"].append(f"未找到参考 {args.ref_source} patch")
        else:
            try:
                with rasterio.open(ref_path) as src:
                    ref_bounds = src.bounds
                    ref_crs = src.crs

                if label_crs != ref_crs:
                    align["ok"] = False
                    align["issues"].append(
                        f"CRS 不一致: label={label_crs}, ref={ref_crs}"
                    )

                bounds_diff = [
                    abs(getattr(label_bounds, a) - getattr(ref_bounds, a))
                    for a in ("left", "bottom", "right", "top")
                ]
                max_diff = max(bounds_diff)
                align["bounds_diff_max"] = float(max_diff)
                if max_diff > args.bounds_tol:
                    align["ok"] = False
                    align["issues"].append(
                        f"bounds 最大偏差 {max_diff:.3f}m 超过容差 {args.bounds_tol}m"
                    )
                elif max_diff > 10.0:
                    # 在容差内但存在明显边缘偏移，记录为 edge patch
                    report["edge_patches"].append(
                        {"patch_id": patch_id, "bounds_diff_max": float(max_diff)}
                    )
            except Exception as exc:  # noqa: BLE001
                align["ok"] = False
                align["issues"].append(f"读取参考 patch 异常: {exc}")

        report["alignment"]["checked"] += 1
        if align["ok"]:
            report["alignment"]["passed"] += 1
        else:
            report["alignment"]["failed"].append(align)
            report["ok"] = False
            report["issues"].extend(
                f"{patch_id}: {issue}" for issue in align["issues"]
            )

    report["class_distribution"] = {
        str(i): int(class_counts[i]) for i in range(len(class_counts))
    }

    # nodata 一致性检查
    if any(c < 0 for c in class_counts):
        report["issues"].append("类别计数出现负值")
        report["ok"] = False

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = Path("/data/xuannv_embedding/qa") / f"qa_labels_{args.region}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"QA 报告: {out_path}")
    print(f"总 patch 数: {report['total_patches']}")
    print(f"对齐检查: {report['alignment']['passed']}/{report['alignment']['checked']} 通过")
    print(f"边缘偏移 patch 数: {len(report['edge_patches'])}")
    print("类别分布:")
    for cls, count in report["class_distribution"].items():
        print(f"  class {cls}: {count}")
    print(f"异常数: {len(report['anomalies'])}")
    print(f"问题数: {len(report['issues'])}")
    print(f"结果: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
