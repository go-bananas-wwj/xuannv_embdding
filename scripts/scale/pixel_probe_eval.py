#!/usr/bin/env python3
"""Run the unified pixel-probe downstream benchmark for embedding maps."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
from pathlib import Path
from statistics import mean


TASK_INFO = {
    "haidian_building_osm": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
        "region": "haidian",
    },
    "haidian_road_osm": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
        "region": "haidian",
    },
    "haidian_water_osm": {
        "label_root": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
        "region": "haidian",
    },
}

METRICS = ("auc_roc", "ap", "f1_best", "f1_at_threshold", "miou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_INFO))
    parser.add_argument("--month", default="202604")
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-patches", type=int, default=8)
    parser.add_argument("--max-pixels-per-patch", type=int, default=4096)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--npu", default="0,1,2")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument(
        "--aef-root",
        type=Path,
        default=Path(
            "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/"
            "aef_haidian_2025_pixel_mlp_probe_20260703"
        ),
    )
    parser.add_argument("--skip-aef-comparison", action="store_true")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    parts = [
        "/root/workspace/xuannv/src",
        "/root/workspace/xuannv/downstreams",
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(part for part in parts if part)
    return env


def parse_npus(raw: str) -> list[str]:
    npus = [part.strip() for part in raw.split(",") if part.strip()]
    if not npus:
        raise ValueError("--npu must contain at least one device id")
    return npus


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=build_env())


def train_task(args: argparse.Namespace, task: str, npu: str) -> None:
    if task not in TASK_INFO:
        raise KeyError(f"Unknown task: {task}")
    info = TASK_INFO[task]
    cmd = [
        "python",
        "scripts/eval/train_aef_downstream_probe.py",
        "--embedding-root",
        str(args.embedding_root),
        "--label-root",
        info["label_root"],
        "--region",
        info["region"],
        "--month",
        args.month,
        "--output-root",
        str(args.output_root / task),
        "--device",
        f"npu:{npu}",
        "--head",
        args.head,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-patches",
        str(args.batch_patches),
        "--max-pixels-per-patch",
        str(args.max_pixels_per_patch),
        "--positive-fraction",
        str(args.positive_fraction),
        "--eval-every",
        str(args.eval_every),
        "--cache-device",
        "npu",
        "--eval-cache-device",
        "cpu",
    ]
    if args.fold is not None:
        cmd.extend(["--fold", str(args.fold)])
    if args.save_predictions:
        cmd.append("--save-predictions")
    run(cmd)


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list summary: {path}")
    return data


def summarize_task(task_root: Path) -> dict[str, float | int | None]:
    rows = load_rows(task_root / "summary.json")
    out: dict[str, float | int | None] = {"num_folds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        out[metric] = mean(values) if values else None
    return out


def write_report(args: argparse.Namespace) -> None:
    report: dict[str, object] = {
        "name": args.name,
        "embedding_root": str(args.embedding_root),
        "output_root": str(args.output_root),
        "trainer": "pixel_probe",
        "head": args.head,
        "month": args.month,
        "tasks": {},
    }
    lines = [
        f"# Pixel Probe Benchmark - {args.name}",
        "",
        f"Embedding root: `{args.embedding_root}`",
        f"Output root: `{args.output_root}`",
        f"Head: `{args.head}`",
        f"Month: `{args.month}`",
        "",
        "| Task | Folds | AUC | AP | F1 best | F1 @ val threshold | mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    task_reports: dict[str, object] = {}
    for task in args.tasks:
        summary = summarize_task(args.output_root / task)
        task_reports[task] = summary
        lines.append(
            "| {task} | {folds} | {auc} | {ap} | {f1_best} | {f1_thr} | {miou} |".format(
                task=task,
                folds=summary["num_folds"],
                auc=fmt(summary["auc_roc"]),
                ap=fmt(summary["ap"]),
                f1_best=fmt(summary["f1_best"]),
                f1_thr=fmt(summary["f1_at_threshold"]),
                miou=fmt(summary["miou"]),
            )
        )
    report["tasks"] = task_reports
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "PIXEL_PROBE_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "PIXEL_PROBE_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def compare_to_aef(args: argparse.Namespace) -> None:
    if args.skip_aef_comparison:
        return
    run(
        [
            "python",
            "scripts/report/compare_benchmark_to_aef.py",
            "--benchmark-root",
            str(args.output_root),
            "--aef-root",
            str(args.aef_root),
            "--output",
            str(args.output_root / "comparison_vs_aef.json"),
            "--name",
            args.name,
            "--tasks",
            *args.tasks,
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_training:
        npus = parse_npus(args.npu)
        with ThreadPoolExecutor(max_workers=min(len(npus), len(args.tasks))) as pool:
            futures = {
                pool.submit(train_task, args, task, npus[idx % len(npus)]): task
                for idx, task in enumerate(args.tasks)
            }
            for future in as_completed(futures):
                future.result()
    write_report(args)
    compare_to_aef(args)


if __name__ == "__main__":
    main()
