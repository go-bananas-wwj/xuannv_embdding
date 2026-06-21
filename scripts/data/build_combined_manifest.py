#!/usr/bin/env python3
"""Build a combined manifest containing all Harbin + Haidian patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Build combined 744-patch manifest")
    p.add_argument("--harbin-manifest", type=Path, required=True)
    p.add_argument("--haidian-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    combined: list[dict] = []
    for region, manifest_path in [
        ("harbin", args.harbin_manifest),
        ("haidian", args.haidian_manifest),
    ]:
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            entry = dict(entry)
            entry["region"] = region
            for key, value in entry.items():
                if key == "patch_id" or not isinstance(value, list):
                    continue
                entry[key] = [f"{region}/{v}" for v in value]
            combined.append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(combined)} patches to {args.output}")


if __name__ == "__main__":
    main()
