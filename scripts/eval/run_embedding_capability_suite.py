#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TASKS = {
    "building": "/data/xuannv_embedding/processed/haidian/labels/building_osm",
    "road": "/data/xuannv_embedding/processed/haidian/labels/road_osm",
    "water": "/data/xuannv_embedding/processed/haidian/labels/osm_water",
    "park_green": "/data/xuannv_embedding/processed/haidian/labels/merged_park_green",
    "construction": "/data/xuannv_embedding/processed/haidian/labels/construction",
    "green": "/data/xuannv_embedding/processed/haidian/labels/osm_green",
    "residential": "/data/xuannv_embedding/processed/haidian/labels/osm_residential",
    "commercial": "/data/xuannv_embedding/processed/haidian/labels/osm_commercial",
    "industrial": "/data/xuannv_embedding/processed/haidian/labels/osm_industrial",
    "agriculture": "/data/xuannv_embedding/processed/haidian/labels/osm_agriculture",
    "rail": "/data/xuannv_embedding/processed/haidian/labels/osm_rail",
    "playground": "/data/xuannv_embedding/processed/haidian/labels/osm_playground",
    "path_walk": "/data/xuannv_embedding/processed/haidian/labels/osm_path_walk",
    "park": "/data/xuannv_embedding/processed/haidian/labels/osm_park",
    "garden": "/data/xuannv_embedding/processed/haidian/labels/osm_garden",
    "pitch": "/data/xuannv_embedding/processed/haidian/labels/osm_pitch",
    "sports": "/data/xuannv_embedding/processed/haidian/labels/osm_sports",
    "school": "/data/xuannv_embedding/processed/haidian/labels/osm_school",
    "university": "/data/xuannv_embedding/processed/haidian/labels/osm_university",
    "education": "/data/xuannv_embedding/processed/haidian/labels/merged_education",
    "education_single": "/data/xuannv_embedding/processed/haidian/labels/osm_education",
    "hospital": "/data/xuannv_embedding/processed/haidian/labels/osm_hospital",
    "parking": "/data/xuannv_embedding/processed/haidian/labels/osm_parking",
    "forest": "/data/xuannv_embedding/processed/haidian/labels/osm_forest",
    "grass": "/data/xuannv_embedding/processed/haidian/labels/osm_grass",
    "retail": "/data/xuannv_embedding/processed/haidian/labels/osm_retail",
    "research_gov": "/data/xuannv_embedding/processed/haidian/labels/osm_research_gov",
    "sports_pitch": "/data/xuannv_embedding/processed/haidian/labels/merged_sports_pitch",
}


@dataclass(frozen=True)
class EvalJob:
    model_name: str
    embedding_root: Path
    task: str
    label_root: Path
    head: str
    shot: str
    fold: int | None
    device: str
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fair Haidian embedding capability suite. AEF and Xuannv use the "
            "same downstream probe, split, train schedule, threshold calibration, and metrics."
        )
    )
    parser.add_argument(
        "--xuannv-root",
        type=Path,
        required=True,
        help="Embedding root containing haidian/patch_xxx/<month>_embedding_map.pt.",
    )
    parser.add_argument(
        "--aef-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--heads", nargs="+", choices=["linear", "mlp", "mlp_deep"], default=["linear", "mlp"])
    parser.add_argument(
        "--shots",
        nargs="+",
        default=["5", "10", "20", "50", "full"],
        help="Positive training patch budgets. Use 'full' for the complete train split.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-pixels-per-patch", type=int, default=4096)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", nargs="+", default=["npu:0"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_shot(shot: str) -> str:
    if shot == "full":
        return shot
    value = int(shot)
    if value <= 0:
        raise ValueError(f"shot must be positive or 'full', got {shot}")
    return str(value)


def build_jobs(args: argparse.Namespace) -> list[EvalJob]:
    tasks: dict[str, Path] = {}
    for task in args.tasks:
        if task not in DEFAULT_TASKS:
            raise KeyError(f"Unknown task {task}. Available: {sorted(DEFAULT_TASKS)}")
        label_root = Path(DEFAULT_TASKS[task])
        split_path = label_root / "split_5fold.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Task {task} has no fixed split: {split_path}")
        tasks[task] = label_root

    models = {
        "xuannv_haidian_v1": args.xuannv_root,
        "aef_annual_2025": args.aef_root,
    }
    shots = [normalize_shot(shot) for shot in args.shots]
    jobs: list[EvalJob] = []
    device_idx = 0
    for model_name, embedding_root in models.items():
        for task, label_root in tasks.items():
            for head in args.heads:
                for shot in shots:
                    for fold in args.folds:
                        device = args.devices[device_idx % len(args.devices)]
                        device_idx += 1
                        jobs.append(
                            EvalJob(
                                model_name=model_name,
                                embedding_root=embedding_root,
                                task=task,
                                label_root=label_root,
                                head=head,
                                shot=shot,
                                fold=fold,
                                device=device,
                                output_root=args.output_root
                                / model_name
                                / task
                                / head
                                / f"shot_{shot}"
                                / f"fold_{fold}",
                            )
                        )
    return jobs


def command_for_job(args: argparse.Namespace, job: EvalJob) -> list[str]:
    month = args.aef_month if job.model_name.startswith("aef_") else args.month
    cmd = [
        sys.executable,
        "scripts/eval/train_aef_downstream_probe.py",
        "--embedding-root",
        str(job.embedding_root),
        "--label-root",
        str(job.label_root),
        "--region",
        args.region,
        "--month",
        month,
        "--output-root",
        str(job.output_root),
        "--fold",
        str(job.fold),
        "--device",
        job.device,
        "--head",
        job.head,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(args.hidden_dim),
        "--max-pixels-per-patch",
        str(args.max_pixels_per_patch),
        "--positive-fraction",
        str(args.positive_fraction),
        "--eval-every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--cache-device",
        "npu" if job.device.startswith("npu") else "cpu",
        "--eval-cache-device",
        "npu" if job.device.startswith("npu") else "cpu",
    ]
    if job.shot != "full":
        cmd += ["--train-positive-patches", job.shot]
    if args.save_predictions:
        cmd.append("--save-predictions")
    return cmd


def run_job(args: argparse.Namespace, job: EvalJob) -> dict[str, Any]:
    job.output_root.mkdir(parents=True, exist_ok=True)
    cmd = command_for_job(args, job)
    log_path = job.output_root / "run.log"
    if args.dry_run:
        log_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
        return {
            "model": job.model_name,
            "task": job.task,
            "head": job.head,
            "shot": job.shot,
            "fold": job.fold,
            "device": job.device,
            "status": "dry_run",
            "output_root": str(job.output_root),
            "command": cmd,
        }
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    status = "ok" if proc.returncode == 0 else "failed"
    record: dict[str, Any] = {
        "model": job.model_name,
        "task": job.task,
        "head": job.head,
        "shot": job.shot,
        "fold": job.fold,
        "device": job.device,
        "status": status,
        "returncode": proc.returncode,
        "output_root": str(job.output_root),
        "log": str(log_path),
    }
    metrics_path = job.output_root / f"fold_{job.fold}" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ["ap", "auc_roc", "f1_0.5", "f1_at_threshold", "f1_best", "miou", "val_threshold"]:
            if key in metrics:
                record[key] = metrics[key]
        record["metrics_path"] = str(metrics_path)
    if proc.returncode != 0:
        raise RuntimeError(f"Job failed: {' '.join(cmd)}. See {log_path}")
    return record


def write_summary(output_root: Path, records: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: (r["task"], r["head"], str(r["shot"]), r["model"], int(r["fold"])))
    (output_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "model",
        "task",
        "head",
        "shot",
        "fold",
        "status",
        "ap",
        "auc_roc",
        "f1_0.5",
        "f1_at_threshold",
        "f1_best",
        "miou",
        "val_threshold",
        "output_root",
    ]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Haidian Embedding Capability Suite",
        "",
        "| Task | Head | Shot | Fold | Model | AP | AUC | F1@val-thr | F1-best | mIoU |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {head} | {shot} | {fold} | {model} | {ap:.4f} | {auc_roc:.4f} | "
            "{f1_at_threshold:.4f} | {f1_best:.4f} | {miou:.4f} |".format(
                task=row["task"],
                head=row["head"],
                shot=row["shot"],
                fold=row["fold"],
                model=row["model"],
                ap=float(row.get("ap", 0.0)),
                auc_roc=float(row.get("auc_roc", 0.0)),
                f1_at_threshold=float(row.get("f1_at_threshold", 0.0)),
                f1_best=float(row.get("f1_best", 0.0)),
                miou=float(row.get("miou", 0.0)),
            )
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fairness_contract": {
            "same_probe_script": "scripts/eval/train_aef_downstream_probe.py",
            "same_splits": True,
            "same_heads": args.heads,
            "same_epochs": args.epochs,
            "same_optimizer": "AdamW",
            "same_threshold_rule": "validation best F1 threshold applied to test",
            "only_embedding_root_differs": True,
        },
        "args": jsonable_args(args),
        "jobs": [
            {
                "model": job.model_name,
                "embedding_root": str(job.embedding_root),
                "task": job.task,
                "label_root": str(job.label_root),
                "head": job.head,
                "shot": job.shot,
                "fold": job.fold,
                "device": job.device,
                "output_root": str(job.output_root),
                "command": command_for_job(args, job),
            }
            for job in jobs
        ],
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_job = {executor.submit(run_job, args, job): job for job in jobs}
        for future in as_completed(future_to_job):
            record = future.result()
            records.append(record)
            write_summary(args.output_root, records)
            print(
                f"[{len(records)}/{len(jobs)}] {record['status']} "
                f"{record['model']} {record['task']} {record['head']} "
                f"shot={record['shot']} fold={record['fold']}",
                flush=True,
            )
    write_summary(args.output_root, records)


if __name__ == "__main__":
    main()
