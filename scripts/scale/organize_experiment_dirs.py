#!/usr/bin/env python3
"""Create a non-destructive timestamped index for experiment directories."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

TIMESTAMP_RE = re.compile(r"(20\d{6})(?:[_-]?(\d{6}))?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments"),
    )
    parser.add_argument("--index-name", default="by_date")
    return parser.parse_args()


def infer_timestamp(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.name)
    if match:
        date = match.group(1)
        time = match.group(2) or "000000"
        return f"{date}_{time}"
    mtime = dt.datetime.utcfromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%Y%m%d_%H%M%S")


def safe_link_name(name: str, timestamp: str) -> str:
    suffix = f"_{timestamp}"
    if name.endswith(suffix):
        return name
    return f"{name}{suffix}"


def collect_summary_paths(path: Path) -> list[str]:
    return [str(p) for p in sorted(path.glob("**/summary_5fold.json"))[:20]]


def collect_record(path: Path, root: Path) -> dict[str, Any]:
    timestamp = infer_timestamp(path)
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "timestamp": timestamp,
        "mtime_utc": dt.datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(),
        "summary_5fold": collect_summary_paths(path),
    }


def write_inventory(records: list[dict[str, Any]], index_dir: Path) -> None:
    records = sorted(records, key=lambda item: (item["timestamp"], item["name"]))
    (index_dir / "inventory.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Experiment Directory Inventory",
        "",
        "This directory contains timestamp-suffixed symlinks. Original experiment",
        "directories are not moved.",
        "",
        "| Timestamp | Name | Original Path | Summary Count |",
        "|---|---|---|---:|",
    ]
    for record in records:
        lines.append(
            "| {timestamp} | {name} | `{path}` | {count} |".format(
                timestamp=record["timestamp"],
                name=record["name"],
                path=record["path"],
                count=len(record["summary_5fold"]),
            )
        )
    (index_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root
    index_dir = root / args.index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child == index_dir:
            continue
        record = collect_record(child, root)
        records.append(record)
        link = index_dir / safe_link_name(child.name, record["timestamp"])
        if link.exists() or link.is_symlink():
            if link.resolve() == child.resolve():
                continue
            link.unlink()
        link.symlink_to(child)

    write_inventory(records, index_dir)
    print(f"indexed {len(records)} experiment directories under {index_dir}")


if __name__ == "__main__":
    main()
