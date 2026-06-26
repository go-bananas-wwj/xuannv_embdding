#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.scale.auto_post_training_codex import (
    build_env,
    latest_epoch_from_status,
    run_codex_review,
    run_post_training_eval,
    run_precompute,
    tmux_alive,
    write_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop train/eval/Codex-upgrade controller for Xuannv embedding."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--train-session", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--npu", default="0")
    parser.add_argument("--regions", nargs="+", default=["haidian", "harbin"])
    parser.add_argument("--samples-per-task", type=int, default=4)
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
    parser.add_argument(
        "--target-macro-delta",
        type=float,
        default=0.05,
        help="Required macro improvement over AEF for the Codex agent to stop.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_epoch_from_train_log(output_dir: Path) -> int | None:
    latest: int | None = None
    for log_path in sorted((output_dir / "logs").glob("train*.log")):
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                marker = " INFO epoch "
                if marker not in line:
                    continue
                tail = line.split(marker, 1)[1]
                epoch_text = tail.split(":", 1)[0]
                if epoch_text.isdigit():
                    latest = max(latest or 0, int(epoch_text))
    return latest


def latest_epoch_from_checkpoints(output_dir: Path, total_epochs: int) -> int | None:
    if (output_dir / f"epoch_{total_epochs}.pt").exists():
        return total_epochs - 1
    latest: int | None = None
    for path in output_dir.glob("epoch_*.pt"):
        epoch_text = path.stem.removeprefix("epoch_")
        if epoch_text.isdigit():
            latest = max(latest or 0, int(epoch_text) - 1)
    return latest


def latest_epoch(output_dir: Path, total_epochs: int) -> int | None:
    values = [
        latest_epoch_from_status(output_dir),
        latest_epoch_from_train_log(output_dir),
        latest_epoch_from_checkpoints(output_dir, total_epochs),
    ]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, text=True)


def wait_for_training(
    train_session: str,
    output_dir: Path,
    total_epochs: int,
    poll_seconds: int,
    state_path: Path,
    round_idx: int,
) -> None:
    while tmux_alive(train_session):
        write_state(
            state_path,
            {
                "stage": "waiting_for_training",
                "round": round_idx,
                "time_utc": now_utc(),
                "train_session": train_session,
                "latest_epoch": latest_epoch(output_dir, total_epochs),
            },
        )
        time.sleep(poll_seconds)

    observed_epoch = latest_epoch(output_dir, total_epochs)
    if observed_epoch is not None and observed_epoch + 1 < total_epochs:
        raise RuntimeError(
            f"训练提前结束: session={train_session}, latest_epoch={observed_epoch}"
        )


def load_decision(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Codex 未写入闭环决策文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"闭环决策文件必须是 JSON object: {path}")
    return data


def closed_loop_prompt(
    benchmark_root: Path,
    decision_path: Path,
    round_idx: int,
    max_rounds: int,
    target_macro_delta: float,
) -> str:
    return f"""
你是 xuannv_embedding 项目的自动升级 Codex，正在执行闭环训练第 {round_idx} 轮。

目标：让下游任务指标明显超过 AEF。停止条件是：
- 与 AEF 对比的 macro 指标整体为正；
- 关键 macro 指标至少达到约 +{target_macro_delta:.3f} 的提升，或你判断已经“远远超过 AEF”；
- 可视化显示 202512/202605 embedding 对变化有明显时序敏感性。

请先只读分析：
- benchmark_root: {benchmark_root}
- comparison_vs_aef: {benchmark_root / "comparison_vs_aef.json"}
- comparison_vs_v1: {benchmark_root / "comparison_vs_v1.0.json"}
- visualizations: {benchmark_root / "visualizations"}
- post_training_report: {benchmark_root / "POST_TRAINING_REPORT.md"}

如果已经达到停止条件：
1. 写中文总结到 {benchmark_root / "CODEX_CLOSED_LOOP_FINAL.md"}
2. 写 JSON 到 {decision_path}：
   {{"action":"stop","reason":"...","benchmark_root":"{benchmark_root}"}}

如果没有达到停止条件，并且 round < {max_rounds}：
1. 选择一个最优先升级，不要同时做多个大改。
2. 直接修改代码/配置/文档，遵守 AGENTS.md；完成后测试、commit、push。
3. 新建一个自包含训练配置，输出目录必须带时间戳，建议 batch size 合理。
4. 用 tmux 启动下一轮 6 卡训练，NPU 0~5。
5. 写 JSON 到 {decision_path}：
   {{
     "action":"continue",
     "reason":"...",
     "next_config":"configs/...",
     "next_run_name":"...",
     "next_train_session":"...",
     "next_train_output_dir":"/data/xuannv_embedding/outputs/...",
     "next_total_epochs":100
   }}

约束：
- 不要删除数据或旧实验。
- 每轮只做一个清晰实验，方便归因。
- 所有实质代码/配置/文档改动都要 commit 并 push。
- 如果缺少必要指标或脚本失败，写 action=stop 并说明 blocker。
""".strip()


def run_upgrade_codex(
    benchmark_root: Path,
    decision_path: Path,
    round_idx: int,
    max_rounds: int,
    target_macro_delta: float,
) -> dict[str, object]:
    output_path = benchmark_root / "CODEX_CLOSED_LOOP_ANALYSIS.md"
    prompt = closed_loop_prompt(
        benchmark_root,
        decision_path,
        round_idx,
        max_rounds,
        target_macro_delta,
    )
    if decision_path.exists():
        decision_path.unlink()
    run_command(
        [
            "codex",
            "exec",
            "--cd",
            "/root/workspace/xuannv",
            "--sandbox",
            "danger-full-access",
            "--output-last-message",
            str(output_path),
            prompt,
        ]
    )
    return load_decision(decision_path)


def main() -> None:
    args = parse_args()
    env = build_env(args.npu)
    state_path = args.train_output_dir / "closed_loop_state.json"

    current_config = args.config
    current_output_dir = args.train_output_dir
    current_session = args.train_session
    current_run_name = args.run_name
    current_total_epochs = args.total_epochs

    for round_idx in range(1, args.max_rounds + 1):
        wait_for_training(
            current_session,
            current_output_dir,
            current_total_epochs,
            args.poll_seconds,
            state_path,
            round_idx,
        )
        write_state(
            state_path,
            {
                "stage": "evaluating",
                "round": round_idx,
                "time_utc": now_utc(),
                "run_name": current_run_name,
            },
        )
        eval_args = argparse.Namespace(
            config=current_config,
            train_output_dir=current_output_dir,
            checkpoint_name="best.pt",
            embedding_output_root=args.embedding_output_root,
            benchmark_base=args.benchmark_base,
            regions=args.regions,
            run_name=current_run_name,
            npu=args.npu,
            samples_per_task=args.samples_per_task,
            dry_run=False,
        )
        embedding_root = run_precompute(eval_args, env, state_path)
        benchmark_root = run_post_training_eval(eval_args, embedding_root, env, state_path)
        run_codex_review(eval_args, benchmark_root, state_path)

        decision_path = benchmark_root / "CLOSED_LOOP_DECISION.json"
        decision = run_upgrade_codex(
            benchmark_root,
            decision_path,
            round_idx,
            args.max_rounds,
            args.target_macro_delta,
        )
        write_state(
            state_path,
            {
                "stage": "codex_decision",
                "round": round_idx,
                "time_utc": now_utc(),
                "decision": decision,
                "benchmark_root": str(benchmark_root),
            },
        )
        if decision.get("action") != "continue":
            break
        current_config = Path(str(decision["next_config"]))
        current_output_dir = Path(str(decision["next_train_output_dir"]))
        current_session = str(decision["next_train_session"])
        current_run_name = str(decision["next_run_name"])
        current_total_epochs = int(decision.get("next_total_epochs", 100))

    write_state(
        state_path,
        {
            "stage": "complete",
            "time_utc": now_utc(),
            "round": round_idx,
        },
    )


if __name__ == "__main__":
    main()
