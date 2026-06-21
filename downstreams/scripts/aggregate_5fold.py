#!/usr/bin/env python3
"""汇总 5-fold 实验结果并输出 summary_5fold.json。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--out-path", type=Path, default=None)
    args = p.parse_args()

    metrics_keys = [
        "miou",
        "f1_0.5",
        "f1_best",
        "precision",
        "recall",
        "ap",
        "auprc",
        "auc_roc",
    ]

    fold_results: list[dict[str, float]] = []
    for fold_dir in sorted(args.experiment_root.glob("fold*")):
        summary_path = fold_dir / "summary_5fold.json"
        if not summary_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        fold_results.append(data)

    if not fold_results:
        print("No fold results found.")
        return

    avg = {k: sum(fold.get(k, 0.0) for fold in fold_results) / len(fold_results) for k in metrics_keys}
    std = {
        k: (sum((fold.get(k, 0.0) - avg[k]) ** 2 for fold in fold_results) / len(fold_results)) ** 0.5
        for k in metrics_keys
    }

    summary = {
        "n_folds": len(fold_results),
        "average": avg,
        "std": std,
        "folds": fold_results,
    }

    out_path = args.out_path or args.experiment_root / "summary_5fold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {len(fold_results)} folds -> {out_path}")
    print(f"AUC: {avg['auc_roc']:.4f} +/- {std['auc_roc']:.4f}")
    print(f"F1_best: {avg['f1_best']:.4f} +/- {std['f1_best']:.4f}")


if __name__ == "__main__":
    main()
