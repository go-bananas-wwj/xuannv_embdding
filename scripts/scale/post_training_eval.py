#!/usr/bin/env python3
"""Post-training quick evaluation, AEF comparison, and visualization workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path

TASKS = (
    "construction",
    "building_change",
    "farm_change",
    "rubbish",
    "water",
    "construction_joint",
)

TASK_INFO = {
    "construction": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/construction",
        "region": "haidian",
    },
    "building_change": {
        "label_root": "/data/xuannv_embedding/processed/harbin/labels/building_change",
        "region": "harbin",
    },
    "farm_change": {
        "label_root": "/data/xuannv_embedding/processed/harbin/labels/farm_change",
        "region": "harbin",
    },
    "rubbish": {
        "label_root": "/data/xuannv_embedding/processed/harbin/labels/rubbish",
        "region": "harbin",
    },
    "water": {
        "label_root": "/data/xuannv_embedding/processed/harbin/labels/osm_water",
        "region": "harbin",
    },
    "construction_joint": {
        "label_root": "/data/xuannv_embedding/processed/construction_joint_v2",
        "region": "construction_joint",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--benchmark-base",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks"),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Existing or explicit benchmark directory. Overrides benchmark-base/run-id.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("downstreams/configs/v2_acceptance_quick_concat_diff.yaml"),
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--fold", type=int, default=0, help="Default quick mode runs fold 0.")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--npu", default="0")
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--skip-v1-comparison", action="store_true")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def build_env(npu: str) -> dict[str, str]:
    env = os.environ.copy()
    src = "/root/workspace/xuannv/src"
    downstreams = "/root/workspace/xuannv/downstreams"
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{downstreams}:{old_pythonpath}"
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    return env


def write_report(benchmark_root: Path, run_id: str, tasks: list[str], mode: str) -> None:
    lines = [
        f"# Post Training Evaluation - {run_id}",
        "",
        f"Benchmark root: `{benchmark_root}`",
        f"Evaluation mode: `{mode}`",
        "",
        "## Outputs",
        "",
        "- `comparison_vs_aef.md`",
        "- `comparison_vs_v1.0.md` when enabled",
        "- `threshold_calibration.md`",
        "- `visualizations/index.md` when prediction export is enabled",
        "",
        "## Task Summaries",
        "",
    ]
    for task in tasks:
        summary = benchmark_root / task / "summary.json"
        meta = benchmark_root / task / "summary_meta.json"
        legacy = benchmark_root / task / "summary_5fold.json"
        lines.append(
            f"- {task}: `{summary}` (`{legacy.name}` is a compatibility alias; meta `{meta}`)"
        )
    (benchmark_root / "POST_TRAINING_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{timestamp}"
    benchmark_root = args.benchmark_root or args.benchmark_base / run_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    env = build_env(args.npu)

    if not args.skip_training:
        for task in args.tasks:
            info = TASK_INFO[task]
            cmd = [
                "python",
                "downstreams/scripts/train_task.py",
                "--task",
                task,
                "--config",
                str(args.config),
                "--embedding-root",
                str(args.embedding_root),
                "--label-root",
                info["label_root"],
                "--region",
                info["region"],
                "--output-root",
                str(benchmark_root / task),
                "--save-predictions",
            ]
            if not args.all_folds:
                cmd.extend(["--fold", str(args.fold)])
            run(cmd, env)

    run(
        [
            "python",
            "scripts/report/compare_benchmark_to_aef.py",
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(benchmark_root / "comparison_vs_aef.json"),
            "--name",
            run_id,
            "--tasks",
            *args.tasks,
        ],
        env,
    )

    run(
        [
            "python",
            "scripts/report/summarize_threshold_calibration.py",
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(benchmark_root / "threshold_calibration.json"),
            "--tasks",
            *args.tasks,
        ],
        env,
    )

    if not args.skip_v1_comparison:
        run(
            [
                "python",
                "scripts/report/compare_v2_acceptance.py",
                "--benchmark-root",
                str(benchmark_root),
                "--output",
                str(benchmark_root / "comparison_vs_v1.0.json"),
            ],
            env,
        )

    if not args.skip_visualization:
        run(
            [
                "python",
                "scripts/report/visualize_downstream_outputs.py",
                "--benchmark-root",
                str(benchmark_root),
                "--embedding-root",
                str(args.embedding_root),
                "--output-root",
                str(benchmark_root / "visualizations"),
                "--tasks",
                *args.tasks,
                "--samples-per-task",
                str(args.samples_per_task),
            ],
            env,
        )

    mode = "full_5fold" if args.all_folds else f"quick_fold_{args.fold}"
    write_report(benchmark_root, run_id, args.tasks, mode)
    print(f"post-training evaluation complete: {benchmark_root}")


if __name__ == "__main__":
    main()
