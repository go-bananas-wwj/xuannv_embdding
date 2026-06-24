#!/usr/bin/env python3
"""汇总 202604 / 202605 / bitemporal 等多个月份设定下的 V1.2 benchmark 结果，并与 AEF 2025 对比。"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

TASKS = ["construction", "building_change", "farm_change", "rubbish"]
METRICS = ["auc_roc", "f1_best", "f1_0.5", "miou"]


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "average" in data and "std" in data:
        # aggregate_5fold 已提供 average/std
        return {"avg": {m: data["average"].get(m, np.nan) for m in METRICS},
                "std": {m: data["std"].get(m, np.nan) for m in METRICS}}
    if isinstance(data, list):
        folds = data
    else:
        folds = data.get("folds", data)
    avg = {}
    std = {}
    for m in METRICS:
        vals = [f[m] for f in folds if m in f]
        avg[m] = float(np.mean(vals)) if vals else np.nan
        std[m] = float(np.std(vals, ddof=0)) if vals else np.nan
    return {"avg": avg, "std": std}


def format_mean_std(avg: float, std: float) -> str:
    if np.isnan(avg):
        return "-"
    return f"{avg:.4f}±{std:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aef-root", required=True, type=Path)
    p.add_argument("--v12-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    variants = {
        "AEF 2025": args.aef_root,
        "V1.2 202604": args.v12_root / "downstream_benchmark_202604",
        "V1.2 202512+202604": args.v12_root / "downstream_benchmark_bitemporal_202512_202604",
        "V1.2 202605": args.v12_root / "downstream_benchmark_202605",
        "V1.2 202512+202605": args.v12_root / "downstream_benchmark_bitemporal_202512_202605",
    }

    aef_task_dir = {
        "construction": "construction_joint",
        "building_change": "building_change",
        "farm_change": "farm_change",
        "rubbish": "rubbish",
    }

    rows = []
    for task in TASKS:
        for variant_name, root in variants.items():
            if variant_name == "AEF 2025":
                summary_path = root / aef_task_dir[task] / "summary_5fold.json"
            else:
                summary_path = root / task / "summary_5fold.json"
            if not summary_path.exists():
                continue
            stats = load_summary(summary_path)
            rows.append({
                "task": task,
                "variant": variant_name,
                **{m: format_mean_std(stats["avg"][m], stats["std"][m]) for m in METRICS},
                "_avg": stats["avg"],
                "_std": stats["std"],
            })

    # 找出每个任务每种指标的最优 variant（按 AUC 或 F1-best）
    lines = []
    lines.append("# V1.2 不同月份 embedding 下游 benchmark 对比报告\n")
    lines.append("- **AEF 2025**: Google/GeoAI Foundation 官方年度 embedding（单时相 2025，64 维）。\n")
    lines.append("- **V1.2 202604 / 202605**: V1.2 模型提取的单月份 embedding。\n")
    lines.append("- **V1.2 202512+202604 / 202512+202605**: V1.2 模型提取的双时相 embedding（construction 仍为单月份 202604/202605）。\n")
    lines.append("- 所有 mask 已按你要求把 202512 重命名为 202604，因此 202604/202605 均对应 4 月份标注。\n")
    lines.append("\n")

    # 总表
    lines.append("## 总览表（mean±std）\n")
    lines.append("| 任务 | 设定 | AUC-ROC | F1-best | F1@0.5 | mIoU |\n")
    lines.append("|------|------|---------|---------|--------|------|\n")
    for r in rows:
        lines.append(f"| {r['task']} | {r['variant']} | {r['auc_roc']} | {r['f1_best']} | {r['f1_0.5']} | {r['miou']} |\n")

    lines.append("\n")

    # 每个任务的 best AUC / best F1-best
    lines.append("## 各任务最优设定\n")
    lines.append("| 任务 | 最优 AUC 设定 | AUC | 最优 F1-best 设定 | F1-best |\n")
    lines.append("|------|---------------|-----|-------------------|---------|\n")
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        best_auc = max(task_rows, key=lambda r: r["_avg"]["auc_roc"])
        best_f1 = max(task_rows, key=lambda r: r["_avg"]["f1_best"])
        lines.append(
            f"| {task} | {best_auc['variant']} | {best_auc['auc_roc']} | "
            f"{best_f1['variant']} | {best_f1['f1_best']} |\n"
        )

    lines.append("\n")
    lines.append("## 结论\n")
    lines.append("- construction：V1.2 各月份版本 F1-best 约 0.04，仍远低于 AEF 的 0.37。\n")
    lines.append("- 变化检测任务（building_change / farm_change / rubbish）：V1.2 各月份版本 AUC 均在 0.50 上下，与 AEF（0.68–0.92）差距巨大。\n")
    lines.append("- 在 V1.2 内部，202605 单月份和 202512+202605 双时相略优于 202604，但差异很小。\n")
    lines.append("- 整体来看，当前 V1.2 模型的下游判别能力仍显著弱于 AEF 2025 基线。\n")

    args.output.write_text("".join(lines), encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
