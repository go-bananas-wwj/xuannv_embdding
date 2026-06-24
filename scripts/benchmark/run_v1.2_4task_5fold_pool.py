#!/usr/bin/env python3
"""V1.2 4-task × 5-fold 下游 benchmark 的 NPU 池化调度器。

每张 NPU 跑一个 fold，所有任务的所有 fold 一起调度，最大化 8 卡利用率。
用法示例：
    python scripts/benchmark/run_v1.2_4task_5fold_pool.py --months 202604
    python scripts/benchmark/run_v1.2_4task_5fold_pool.py --months 202512 202604 --out-suffix bitemporal
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NATIONAL_OUTPUT = Path("/data2/xuannv_embedding/national/outputs")
EMB_ROOT = Path("/data2/xuannv_embedding/national/embeddings_v1.2")
LABEL_ROOT = Path("/data/xuannv_embedding/processed")
CONFIG_ROOT = ROOT / "downstreams" / "configs" / "aef_benchmark"

sys.path.insert(0, str(ROOT))
from downstreams.data.split import create_combined_stratified_folds, create_stratified_folds


def latest_embedding_subdir() -> Path:
    dirs = sorted(EMB_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise RuntimeError(f"no embedding dirs under {EMB_ROOT}")
    return dirs[0]


def ensure_splits() -> None:
    tasks = {
        "construction": {"regions": ["haidian", "harbin"]},
        "building_change": {"region": "harbin"},
        "farm_change": {"region": "harbin"},
        "rubbish": {"region": "harbin"},
    }
    for task, spec in tasks.items():
        task_label = LABEL_ROOT / task
        task_label.mkdir(parents=True, exist_ok=True)
        if "regions" in spec:
            split_path = task_label / "split_joint_5fold.json"
            if split_path.exists():
                continue
            mask_dirs = {r: LABEL_ROOT / r / "labels" / task / "masks" for r in spec["regions"]}
            split = create_combined_stratified_folds(mask_dirs, seed=42)
            split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2))
            print(f"wrote {split_path}")
        else:
            split_path = task_label / "split_5fold.json"
            if split_path.exists():
                continue
            mask_dir = LABEL_ROOT / spec["region"] / "labels" / task / "masks"
            split = create_stratified_folds(mask_dir, seed=42)
            split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2))
            print(f"wrote {split_path}")


def build_fold_config(
    base_cfg: Path, gpu: int, tmp_dir: Path, task: str, fold: int, batch_size: int | None = None
) -> Path:
    data = yaml.safe_load(open(base_cfg))
    data.setdefault("experiment", {})["device"] = f"npu:{gpu}"
    if batch_size is not None:
        data.setdefault("training", {})["batch_size"] = batch_size
    out = tmp_dir / f"{task}_fold{fold}.yaml"
    yaml.safe_dump(data, open(out, "w"), allow_unicode=True, sort_keys=False)
    return out


def run_benchmark(months: list[str], out_suffix: str, max_gpus: int, batch_size: int | None = None) -> None:
    emb_subdir = latest_embedding_subdir()
    bench_root = Path(f"/data2/xuannv_embedding/national/downstream_benchmark_{out_suffix}")
    bench_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = bench_root / ".tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> embedding root: {emb_subdir}")
    print(f"==> months: {months}")
    print(f"==> bench root: {bench_root}")
    print(f"==> max gpus: {max_gpus}")

    ensure_splits()

    task_configs = {
        "construction": CONFIG_ROOT / "construction.yaml",
        "building_change": CONFIG_ROOT / "building_change.yaml",
        "farm_change": CONFIG_ROOT / "farm_change.yaml",
        "rubbish": CONFIG_ROOT / "rubbish.yaml",
    }

    # 提交所有 fold job，按 NPU 池调度
    for task, cfg_path in task_configs.items():
        out_dir = bench_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "summary_5fold.json").exists():
            print(f"==> {task} already done, skipping")
            continue

        task_months = months.copy()
        if task == "construction" and len(months) > 1:
            task_months = [months[-1]]
            print(f"==> {task} uses single month: {task_months}")

        print(f"==> scheduling {task} (months={task_months})")
        running: list[tuple[subprocess.Popen, int, Path, int]] = []
        completed = 0
        for fold in range(5):
            # 等待空闲 NPU
            while len(running) >= max_gpus:
                for i, (proc, gpu, fold_out, f) in list(enumerate(running)):
                    ret = proc.poll()
                    if ret is not None:
                        if ret != 0:
                            raise RuntimeError(f"fold {task}/fold_{f} failed with code {ret}")
                        completed += 1
                        running.pop(i)
                        print(f"  completed {task} fold {f} on NPU {gpu} ({completed}/5)")
                        break
                else:
                    time.sleep(2)

            gpu = fold % max_gpus if len(running) == 0 else next(g for g in range(max_gpus) if g not in {x[1] for x in running})
            fold_out = out_dir / f"fold_{fold}"
            fold_out.mkdir(parents=True, exist_ok=True)
            fold_cfg = build_fold_config(cfg_path, gpu, tmp_dir, task, fold, batch_size)
            months_arg = " ".join(task_months)

            if task == "construction":
                cmd = [
                    sys.executable, "-m", "downstreams.scripts.train_task",
                    "--task", task,
                    "--config", str(fold_cfg),
                    "--embedding-root", str(emb_subdir),
                    "--label-root", str(LABEL_ROOT),
                    "--regions", "haidian", "harbin",
                    "--output-root", str(fold_out),
                    "--months", *task_months,
                    "--fold", str(fold),
                ]
            else:
                cmd = [
                    sys.executable, "-m", "downstreams.scripts.train_task",
                    "--task", task,
                    "--config", str(fold_cfg),
                    "--embedding-root", str(emb_subdir),
                    "--label-root", str(LABEL_ROOT / "harbin" / "labels"),
                    "--region", "harbin",
                    "--output-root", str(fold_out),
                    "--months", *task_months,
                    "--fold", str(fold),
                ]
            log = open(fold_out / "train.log", "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((proc, gpu, fold_out, fold))
            print(f"  launched {task} fold {fold} on NPU {gpu}")

        # 等待剩余 folds
        while running:
            for i, (proc, gpu, fold_out, f) in list(enumerate(running)):
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        raise RuntimeError(f"fold {task}/fold_{f} failed with code {ret}")
                    completed += 1
                    running.pop(i)
                    print(f"  completed {task} fold {f} on NPU {gpu} ({completed}/5)")
                    break
            else:
                time.sleep(2)

        # 汇总
        subprocess.run(
            [
                sys.executable, "-m", "downstreams.scripts.aggregate_5fold",
                "--experiment-root", str(out_dir),
                "--out-path", str(out_dir / "summary_5fold.json"),
            ],
            check=True,
            cwd=ROOT,
        )
        print(f"==> aggregated {task}")

    print(f"==> Benchmark complete. Results under {bench_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--months", nargs="+", required=True, help="月份列表，例如 202604 或 202512 202604")
    p.add_argument("--out-suffix", required=True, help="输出目录后缀")
    p.add_argument("--max-gpus", type=int, default=8, help="同时使用的最大 NPU 数")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 downstream config 的 batch_size")
    args = p.parse_args()
    run_benchmark(args.months, args.out_suffix, args.max_gpus, args.batch_size)


if __name__ == "__main__":
    main()
