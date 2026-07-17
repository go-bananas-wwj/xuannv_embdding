#!/usr/bin/env python3
"""Download a pinned OSM extract atomically and write a SHA-256 lock record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path, lock_path: Path) -> dict[str, str]:
    if not url.startswith("https://"):
        raise ValueError("only HTTPS source URLs are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "xuannv-china-v1-data-prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    digest = _sha256(partial)
    os.replace(partial, destination)
    lock = {
        "schema_version": "xuannv_osm_snapshot_lock_v1",
        "url": url,
        "artifact_path": str(destination.resolve()),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=lock_path.parent, delete=False) as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    os.replace(temp_path, lock_path)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch(args.url, args.destination, args.lock), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
