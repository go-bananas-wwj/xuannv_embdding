#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EPOCH_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO epoch "
    r"(?P<epoch>\d+): train_loss=(?P<train_loss>[0-9.]+)"
    r"(?:, val_loss=(?P<val_loss>[0-9.]+))?"
)
BEST_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO "
    r"新的最佳 checkpoint: epoch=(?P<epoch>\d+), loss=(?P<loss>[0-9.]+)"
)
NPU_PROCESS_RE = re.compile(r"^\|\s*\d+\s+\d+\s+\|\s+\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodically summarize a training run.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmux-session", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def tmux_alive(session: str) -> bool:
    code, _ = run_command(["tmux", "has-session", "-t", session])
    return code == 0


def parse_log(log_path: Path) -> dict[str, object]:
    if not log_path.exists():
        return {"epochs": [], "latest_epoch": None, "best": None}

    epochs: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                item: dict[str, object] = {
                    "time": epoch_match.group("time"),
                    "epoch": int(epoch_match.group("epoch")),
                    "train_loss": float(epoch_match.group("train_loss")),
                }
                if epoch_match.group("val_loss") is not None:
                    item["val_loss"] = float(epoch_match.group("val_loss"))
                epochs.append(item)
                continue
            best_match = BEST_RE.search(line)
            if best_match:
                best = {
                    "time": best_match.group("time"),
                    "epoch": int(best_match.group("epoch")),
                    "loss": float(best_match.group("loss")),
                }
    return {
        "epochs": epochs[-20:],
        "latest_epoch": epochs[-1] if epochs else None,
        "best": best,
    }


def checkpoint_inventory(output_dir: Path) -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*.pt")):
        stat = path.stat()
        checkpoints.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    return checkpoints


def npu_summary() -> str:
    code, output = run_command(["npu-smi", "info"])
    if code != 0:
        return output[-4000:]
    lines = output.splitlines()
    process_lines = [line for line in lines if NPU_PROCESS_RE.search(line)]
    if process_lines:
        return "\n".join(process_lines)
    return output[-4000:]


def write_status(args: argparse.Namespace) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    log_info = parse_log(args.log)
    alive = tmux_alive(args.tmux_session)
    latest = log_info["latest_epoch"]
    progress = None
    if isinstance(latest, dict):
        progress = round((int(latest["epoch"]) + 1) / args.total_epochs, 4)

    status: dict[str, object] = {
        "time_utc": now,
        "experiment": args.experiment,
        "tmux_session": args.tmux_session,
        "tmux_alive": alive,
        "log": str(args.log),
        "latest_epoch": latest,
        "progress": progress,
        "best": log_info["best"],
        "recent_epochs": log_info["epochs"],
        "checkpoints": checkpoint_inventory(args.output_dir),
        "npu_summary": npu_summary(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "monitor_status.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(status, ensure_ascii=False) + "\n")

    latest_path = args.output_dir / "monitor_status_latest.md"
    with latest_path.open("w", encoding="utf-8") as f:
        f.write(f"# {args.experiment}\n\n")
        f.write(f"- time_utc: `{now}`\n")
        f.write(f"- tmux_alive: `{alive}`\n")
        f.write(f"- progress: `{progress}`\n")
        f.write(f"- latest_epoch: `{latest}`\n")
        f.write(f"- best: `{log_info['best']}`\n")
        f.write(f"- checkpoints: `{status['checkpoints']}`\n\n")
        f.write("## NPU\n\n```text\n")
        f.write(str(status["npu_summary"]))
        f.write("\n```\n")

    return status


def main() -> None:
    args = parse_args()
    while True:
        status = write_status(args)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if args.once:
            return
        if not status["tmux_alive"]:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
