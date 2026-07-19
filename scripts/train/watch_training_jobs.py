#!/usr/bin/env python3
"""Terminate stalled torchrun jobs so their owning queue can retry quickly."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def _torchrun_jobs() -> dict[int, Path]:
    jobs: dict[int, Path] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [value.decode("utf-8", errors="replace") for value in args if value]
        if not any("torchrun" in value for value in decoded):
            continue
        if "scripts/train/train.py" not in decoded or "--config" not in decoded:
            continue
        config_index = decoded.index("--config") + 1
        if config_index >= len(decoded):
            continue
        jobs[int(proc.name)] = Path(decoded[config_index])
    return jobs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=int, default=600)
    parser.add_argument("--startup-grace-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if min(args.stall_seconds, args.startup_grace_seconds, args.poll_seconds) <= 0:
        raise ValueError("watchdog timing values must be positive")

    first_seen: dict[int, float] = {}
    terminated: set[int] = set()
    while True:
        now = time.time()
        jobs = _torchrun_jobs()
        active = []
        for pid, config in jobs.items():
            first_seen.setdefault(pid, now)
            log_path = args.log_root / f"{config.stem}.log"
            progress_at = log_path.stat().st_mtime if log_path.exists() else first_seen[pid]
            stalled_for = max(0.0, now - progress_at)
            age = max(0.0, now - first_seen[pid])
            should_terminate = (
                pid not in terminated
                and age >= args.startup_grace_seconds
                and stalled_for >= args.stall_seconds
            )
            if should_terminate:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                else:
                    terminated.add(pid)
                    event = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "pid": pid,
                        "config": str(config),
                        "log": str(log_path),
                        "stalled_seconds": round(stalled_for, 1),
                        "action": "sigterm_torchrun",
                    }
                    args.events.parent.mkdir(parents=True, exist_ok=True)
                    with args.events.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            active.append({
                "pid": pid,
                "experiment": config.stem,
                "age_seconds": round(age, 1),
                "stalled_seconds": round(stalled_for, 1),
                "terminated": pid in terminated,
            })
        first_seen = {pid: first_seen[pid] for pid in jobs}
        terminated.intersection_update(jobs)
        _write_json(args.heartbeat, {
            "at": datetime.now(timezone.utc).isoformat(),
            "stall_seconds": args.stall_seconds,
            "jobs": active,
        })
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
