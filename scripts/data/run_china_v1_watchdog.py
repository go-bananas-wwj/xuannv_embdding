#!/usr/bin/env python3
"""Restart a resumable China V1 worker when its COG heartbeat stalls.

The child keeps its partial Zarr shard.  A restart only refreshes process-local
SAS tokens and retries unfinished source/month records; completed records stay
protected by the shard's ``done`` matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _terminate(process: subprocess.Popen[bytes], grace_seconds: int = 20) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--restart-delay", type=int, default=15)
    parser.add_argument("--max-restarts", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("supply a worker command after --")
    if args.stall_seconds < 30 or args.max_restarts < 0:
        raise ValueError("invalid watchdog limits")

    restarts = 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        args.heartbeat.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment["CHINA_V1_HEARTBEAT_PATH"] = str(args.heartbeat)
        environment["PYTHONUNBUFFERED"] = "1"
        with args.log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] watchdog start={restarts} command={' '.join(args.command)}\n")
            log.flush()
            child = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            reason = "completed"
            while child.poll() is None:
                time.sleep(10)
                if args.heartbeat.exists() and time.time() - args.heartbeat.stat().st_mtime <= args.stall_seconds:
                    continue
                reason = "no_heartbeat"
                _terminate(child)
                break
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(), "restarts": restarts,
            "reason": reason, "returncode": child.returncode, "command": args.command,
        }
        _write_state(args.state, state)
        if reason == "completed" and child.returncode == 0:
            return
        if restarts >= args.max_restarts:
            raise RuntimeError(f"watchdog exceeded {args.max_restarts} restarts")
        restarts += 1
        time.sleep(args.restart_delay)


if __name__ == "__main__":
    main()
