#!/usr/bin/env python3
"""Run an assigned, restartable subset of China V1 Zarr shard jobs."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from materialize_china_v1_shard import _patch_from_record, _read_jsonl, load_catalogs, materialize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count or args.passes <= 0:
        raise ValueError("invalid worker assignment or pass count")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = sorted(args.points_dir.glob(f"{args.prefix}_[0-9][0-9][0-9].jsonl"))
    assigned = [job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index]
    if not assigned:
        raise ValueError("worker received no jobs")
    catalogs = load_catalogs(args.catalog_root, args.months)
    pending = list(assigned)
    reports: list[dict[str, object]] = []
    for pass_number in range(1, args.passes + 1):
        retry: list[Path] = []
        for job in pending:
            output = args.output_root / f"{job.stem}.zarr"
            if output.exists():
                logging.info("already complete: %s", output)
                continue
            points = [_patch_from_record(record) for record in _read_jsonl(job)]
            try:
                report = materialize(points, args.catalog_root, output, args.months, args.top_k, catalogs)
            except Exception as exc:
                logging.exception("pass %s failed: %s", pass_number, job.name)
                retry.append(job)
                reports.append({"job": job.name, "pass": pass_number, "status": "partial", "error": f"{type(exc).__name__}: {exc}"})
            else:
                reports.append({"job": job.name, "pass": pass_number, "status": "complete", **report})
        if not retry:
            break
        pending = retry
        time.sleep(min(60, 2 ** (pass_number - 1)))
    print(json.dumps({"worker_index": args.worker_index, "assigned": len(assigned), "remaining_partial": [job.name for job in pending], "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
