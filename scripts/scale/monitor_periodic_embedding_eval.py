#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASKS = {
    "building": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
    "road": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
    "water": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config: Path
    output_dir: Path


def parse_spec(raw: str) -> ExperimentSpec:
    parts = raw.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--spec must be NAME|CONFIG|OUTPUT_DIR, got: " + raw
        )
    return ExperimentSpec(parts[0], Path(parts[1]), Path(parts[2]))


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def find_embedding_dir(
    embedding_root: Path,
    checkpoint_stem: str,
    suffix: str,
) -> Path:
    candidates = sorted(embedding_root.glob(f"*_{checkpoint_stem[:8]}_{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No embedding dir found for {checkpoint_stem} {suffix}"
        )
    return candidates[-1]


def summarize_task_metrics(bench_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in TASKS:
        path = bench_root / task / "summary.json"
        if not path.exists():
            summary[task] = {"error": f"missing {path}"}
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        row = rows[0] if rows else {}
        summary[task] = {
            "auc": row.get("auc"),
            "ap": row.get("ap"),
            "f1_best": row.get("f1_best"),
            "threshold_best": row.get("threshold_best"),
            "miou_at_best": row.get("miou_at_best"),
        }
    vals = [
        metrics["f1_best"]
        for metrics in summary.values()
        if isinstance(metrics.get("f1_best"), (int, float))
    ]
    summary["macro"] = {"f1_best": sum(vals) / len(vals) if vals else None}
    return summary


def evaluate_checkpoint(
    spec: ExperimentSpec,
    epoch: int,
    args: argparse.Namespace,
    state: dict[str, Any],
) -> None:
    key = f"{spec.name}:epoch_{epoch}"
    checkpoint = spec.output_dir / f"epoch_{epoch}.pt"
    if key in state["completed"]:
        return
    if not checkpoint.exists():
        return

    suffix = f"{args.suffix_prefix}_{spec.name}_epoch{epoch}_{args.month}"
    log_path = args.log_dir / f"{spec.name}_epoch{epoch}.log"
    bench_root = args.benchmark_root / f"{spec.name}_epoch{epoch}_{args.month}_fold0"
    embedding_dir: Path | None = None

    try:
        try:
            embedding_dir = find_embedding_dir(args.embedding_root, checkpoint.stem, suffix)
            print(f"[monitor] reuse embedding {embedding_dir}", flush=True)
        except FileNotFoundError:
            print(f"[monitor] export {spec.name} epoch {epoch}", flush=True)
            run_cmd(
                [
                    sys.executable,
                    "downstreams/scripts/precompute_embeddings.py",
                    "--config",
                    str(spec.config),
                    "--regions",
                    args.region,
                    "--output-root",
                    str(args.embedding_root),
                    "--checkpoint",
                    str(checkpoint),
                    "--suffix",
                    suffix,
                    "--months",
                    args.month,
                    "--device",
                    args.export_device,
                ],
                log_path,
            )
            embedding_dir = find_embedding_dir(args.embedding_root, checkpoint.stem, suffix)

        for task, label_root in TASKS.items():
            print(f"[monitor] eval {spec.name} epoch {epoch} task {task}", flush=True)
            run_cmd(
                [
                    sys.executable,
                    "scripts/eval/train_aef_downstream_probe.py",
                    "--embedding-root",
                    str(embedding_dir),
                    "--label-root",
                    label_root,
                    "--region",
                    args.region,
                    "--month",
                    args.month,
                    "--output-root",
                    str(bench_root / task),
                    "--fold",
                    "0",
                    "--device",
                    args.eval_device,
                    "--head",
                    "mlp",
                    "--hidden-dim",
                    str(args.hidden_dim),
                    "--epochs",
                    str(args.probe_epochs),
                    "--cache-device",
                    args.cache_device,
                    "--eval-cache-device",
                    args.eval_cache_device,
                ],
                log_path,
            )

        summary = summarize_task_metrics(bench_root)
        report = {
            "experiment": spec.name,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "embedding_deleted": str(embedding_dir),
            "benchmark_root": str(bench_root),
            "metrics": summary,
        }
        (bench_root / "periodic_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state["completed"][key] = report
        state["failed"].pop(key, None)
        save_state(args.state_path, state)
        print(f"[monitor] done {spec.name} epoch {epoch}: {summary}", flush=True)
    except Exception as exc:  # noqa: BLE001
        state["failed"][key] = {"error": repr(exc), "log": str(log_path)}
        save_state(args.state_path, state)
        print(f"[monitor] failed {spec.name} epoch {epoch}: {exc!r}", flush=True)
    finally:
        if embedding_dir is not None and embedding_dir.exists():
            shutil.rmtree(embedding_dir)
            print(f"[monitor] deleted embedding {embedding_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export temporary embeddings at checkpoints, run quick probes, then delete embeddings."
    )
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=[200, 400, 600, 800])
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--month", default="202604")
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/v2_202512_202605_periodic"),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p10_periodic"),
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p10_periodic/monitor_state.json"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/data/xuannv_embedding/logs/p10_periodic_eval"),
    )
    parser.add_argument("--suffix-prefix", default="periodic")
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cache-device", choices=["none", "cpu", "npu"], default="cpu")
    parser.add_argument("--eval-cache-device", choices=["none", "cpu", "npu"], default="cpu")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state_path)
    args.embedding_root.mkdir(parents=True, exist_ok=True)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    while True:
        for spec in args.spec:
            for epoch in args.epochs:
                evaluate_checkpoint(spec, epoch, args, state)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
