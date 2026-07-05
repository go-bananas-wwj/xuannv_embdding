#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASKS = {
    "building": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
    "road": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
    "water": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate simple Haidian downstream probes on production embeddings."
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task in args.tasks:
        if task not in TASKS:
            raise KeyError(f"Unknown task: {task}. Available: {sorted(TASKS)}")
        cmd = [
            sys.executable,
            "scripts/eval/train_aef_downstream_probe.py",
            "--embedding-root",
            str(args.embedding_root),
            "--label-root",
            TASKS[task],
            "--region",
            "haidian",
            "--month",
            args.month,
            "--output-root",
            str(args.output_root / task),
            "--fold",
            str(args.fold),
            "--device",
            args.device,
            "--head",
            args.head,
            "--epochs",
            str(args.epochs),
            "--cache-device",
            "npu" if args.device.startswith("npu") else "cpu",
            "--eval-cache-device",
            "npu" if args.device.startswith("npu") else "cpu",
        ]
        if args.save_predictions:
            cmd.append("--save-predictions")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
