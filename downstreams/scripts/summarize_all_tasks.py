#!/usr/bin/env python3
"""汇总 5 个下游任务的 5-fold 结果到一个总表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    tasks = {
        "construction": "construction_upernet_harbin_5fold_v3",
        "building_change": "building_change_diff_unet_harbin_5fold",
        "farm_change": "farm_change_diff_unet_harbin_5fold",
        "rubbish": "rubbish_diff_unet_harbin_5fold",
        "construction_joint": "construction_upernet_joint_5fold_v3",
    }

    summary = {}
    for task_name, exp_dir in tasks.items():
        summary_path = args.experiment_root / exp_dir / "summary_5fold.json"
        if not summary_path.exists():
            summary[task_name] = {"status": "not_found", "path": str(summary_path)}
            continue
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        avg = data.get("average", {})
        std = data.get("std", {})
        summary[task_name] = {
            "n_folds": data.get("n_folds", 0),
            "auc_roc_mean": avg.get("auc_roc", 0.0),
            "auc_roc_std": std.get("auc_roc", 0.0),
            "f1_best_mean": avg.get("f1_best", 0.0),
            "f1_best_std": std.get("f1_best", 0.0),
            "f1_0.5_mean": avg.get("f1_0.5", 0.0),
            "miou_mean": avg.get("miou", 0.0),
            "status": "ok",
        }

    out_path = args.output or args.experiment_root / "all_tasks_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
