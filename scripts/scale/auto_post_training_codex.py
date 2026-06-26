#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for training completion, run post-training eval, "
            "then ask Codex for upgrades."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--train-session", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument(
        "--embedding-output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/v2_202512_202605"),
    )
    parser.add_argument(
        "--benchmark-base",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks"),
    )
    parser.add_argument("--regions", nargs="+", default=["haidian", "harbin"])
    parser.add_argument("--npu", default="0")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--skip-codex", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(
    cmd: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, env=env)


def run_capture(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def tmux_alive(session: str) -> bool:
    code, _ = run_capture(["tmux", "has-session", "-t", session])
    return code == 0


def latest_epoch_from_status(output_dir: Path) -> int | None:
    status_path = output_dir / "monitor_status_latest.md"
    if not status_path.exists():
        return None
    text = status_path.read_text(encoding="utf-8", errors="replace")
    marker = "'epoch': "
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else None


def build_env(npu: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"/root/workspace/xuannv/src:/root/workspace/xuannv:"
        f"/root/workspace/xuannv/downstreams:{pythonpath}"
    )
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    return env


def wait_for_training(args: argparse.Namespace, state_path: Path) -> None:
    while tmux_alive(args.train_session):
        latest_epoch = latest_epoch_from_status(args.train_output_dir)
        write_state(
            state_path,
            {
                "stage": "waiting_for_training",
                "time_utc": now_utc(),
                "latest_epoch": latest_epoch,
                "train_session": args.train_session,
            },
        )
        time.sleep(args.poll_seconds)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def newest_child(root: Path, before: set[Path]) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and p not in before]
    if not candidates:
        candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"没有找到 embedding 输出目录: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_precompute(args: argparse.Namespace, env: dict[str, str], state_path: Path) -> Path:
    checkpoint = args.train_output_dir / args.checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    args.embedding_output_root.mkdir(parents=True, exist_ok=True)
    before = {p for p in args.embedding_output_root.iterdir() if p.is_dir()}
    write_state(state_path, {"stage": "precompute_embeddings", "time_utc": now_utc()})
    cmd = [
        "python",
        "downstreams/scripts/precompute_embeddings.py",
        "--config",
        str(args.config),
        "--regions",
        *args.regions,
        "--output-root",
        str(args.embedding_output_root),
        "--checkpoint",
        str(checkpoint),
        "--suffix",
        args.run_name,
    ]
    if not args.dry_run:
        run_command(cmd, env=env)
    return newest_child(args.embedding_output_root, before)


def run_post_training_eval(
    args: argparse.Namespace,
    embedding_root: Path,
    env: dict[str, str],
    state_path: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark_root = args.benchmark_base / f"{args.run_name}_{timestamp}"
    write_state(
        state_path,
        {
            "stage": "post_training_eval",
            "time_utc": now_utc(),
            "embedding_root": str(embedding_root),
            "benchmark_root": str(benchmark_root),
        },
    )
    cmd = [
        "python",
        "scripts/scale/post_training_eval.py",
        "--embedding-root",
        str(embedding_root),
        "--run-name",
        args.run_name,
        "--benchmark-root",
        str(benchmark_root),
        "--npu",
        args.npu,
        "--fold",
        "0",
        "--samples-per-task",
        str(args.samples_per_task),
    ]
    if not args.dry_run:
        run_command(cmd, env=env)
    return benchmark_root


def run_codex_review(
    args: argparse.Namespace,
    benchmark_root: Path,
    state_path: Path,
) -> Path | None:
    if args.skip_codex:
        return None
    output_path = benchmark_root / "CODEX_UPGRADE_RECOMMENDATION.md"
    prompt = f"""
你是 xuannv_embedding 项目的训练结果分析 Codex。请只读分析，不要修改代码或配置。

请读取以下训练后评估目录，并给出下一轮模型升级建议：
- benchmark_root: {benchmark_root}
- post_training_report: {benchmark_root / "POST_TRAINING_REPORT.md"}
- aef_comparison: {benchmark_root / "comparison_vs_aef.json"}
- v1_comparison: {benchmark_root / "comparison_vs_v1.0.json"}
- visualizations: {benchmark_root / "visualizations"}

请输出中文 Markdown，包含：
1. 当前结果是否能作为 bug-fixed baseline；
2. 与 AEF/V1 相比的主要差距；
3. 视觉结果是否显示 202512/202605 的时序敏感性；
4. 下一轮最多 3 个优先升级实验，明确改哪些模块/损失/配置；
5. 每个实验的预期收益、风险和验收指标。

最终答案写入这个文件：{output_path}
""".strip()
    write_state(
        state_path,
        {
            "stage": "codex_upgrade_review",
            "time_utc": now_utc(),
            "benchmark_root": str(benchmark_root),
            "output": str(output_path),
        },
    )
    if args.dry_run:
        return output_path
    run_command(
        [
            "codex",
            "exec",
            "--cd",
            "/root/workspace/xuannv",
            "--sandbox",
            "read-only",
            "--output-last-message",
            str(output_path),
            prompt,
        ]
    )
    return output_path


def main() -> None:
    args = parse_args()
    state_path = args.train_output_dir / "auto_post_training_state.json"
    env = build_env(args.npu)
    write_state(state_path, {"stage": "started", "time_utc": now_utc()})
    wait_for_training(args, state_path)

    latest_epoch = latest_epoch_from_status(args.train_output_dir)
    if latest_epoch is not None and latest_epoch + 1 < args.total_epochs:
        write_state(
            state_path,
            {
                "stage": "training_ended_early",
                "time_utc": now_utc(),
                "latest_epoch": latest_epoch,
            },
        )
        raise RuntimeError(f"训练提前结束: latest_epoch={latest_epoch}")

    embedding_root = run_precompute(args, env, state_path)
    benchmark_root = run_post_training_eval(args, embedding_root, env, state_path)
    codex_output = run_codex_review(args, benchmark_root, state_path)
    write_state(
        state_path,
        {
            "stage": "complete",
            "time_utc": now_utc(),
            "embedding_root": str(embedding_root),
            "benchmark_root": str(benchmark_root),
            "codex_output": str(codex_output) if codex_output else None,
        },
    )


if __name__ == "__main__":
    main()
