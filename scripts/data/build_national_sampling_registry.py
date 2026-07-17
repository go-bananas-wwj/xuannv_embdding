#!/usr/bin/env python3
"""Build a reproducible nationwide training-chip registry from an atlas JSONL.

The atlas is deliberately a lightweight, pre-download artifact: each input line
describes one 1280 m candidate chip and the quality/semantic summaries used to
select it.  Raster acquisition happens only after this registry is frozen.

Required atlas fields:
  patch_id, grid_id, grid_row, grid_col, eligible

Recommended fields:
  admin1, ecoregion, strata (list or comma-separated string), and
  stratum_scores (mapping from stratum name to [0, 1] coverage/priority).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_unit_hash(seed: int, value: str) -> float:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _parse_strata(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("strata must be a comma-separated string or a list of strings")


def _read_atlas(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on atlas line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"atlas line {line_number} must be an object")
            yield record


def _validate_record(record: dict[str, Any]) -> None:
    required = ("patch_id", "grid_id", "grid_row", "grid_col", "eligible")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"atlas record {record.get('patch_id', '<unknown>')} missing {missing}")
    if not isinstance(record["patch_id"], str) or not record["patch_id"]:
        raise ValueError("patch_id must be a non-empty string")
    if not isinstance(record["grid_id"], str) or not record["grid_id"]:
        raise ValueError(f"{record['patch_id']}: grid_id must be a non-empty string")
    if not isinstance(record["grid_row"], int) or not isinstance(record["grid_col"], int):
        raise ValueError(f"{record['patch_id']}: grid_row and grid_col must be integers")
    _parse_strata(record.get("strata"))


def _macrocell(record: dict[str, Any], macro_side: int) -> tuple[str, int, int]:
    return (
        record["grid_id"],
        record["grid_row"] // macro_side,
        record["grid_col"] // macro_side,
    )


def _stratum_score(record: dict[str, Any], stratum: str, seed: int) -> tuple[float, float]:
    scores = record.get("stratum_scores", {})
    if scores is None:
        scores = {}
    if not isinstance(scores, dict):
        raise ValueError(f"{record['patch_id']}: stratum_scores must be an object")
    raw_score = scores.get(stratum, 1.0)
    if not isinstance(raw_score, (int, float)):
        raise ValueError(f"{record['patch_id']}: score for {stratum} must be numeric")
    # Larger coverage/priority wins; the deterministic hash only breaks ties.
    return float(raw_score), -_stable_unit_hash(seed, f"{stratum}:{record['patch_id']}")


def _parse_quota_overrides(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        name, separator, count = value.partition("=")
        if not separator or not name or not count.isdigit() or int(count) <= 0:
            raise ValueError(f"invalid quota override {value!r}; use stratum=count")
        quotas[name] = int(count)
    return quotas


def _load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    sampling = policy.get("sampling", {})
    quotas = sampling.get("supplement_quotas", {})
    if not isinstance(quotas, dict) or not all(
        isinstance(name, str) and isinstance(count, int) and count > 0
        for name, count in quotas.items()
    ):
        raise ValueError("policy sampling.supplement_quotas must be a positive integer mapping")
    return policy


def _push_candidate(
    heap: list[tuple[tuple[float, float], int, dict[str, Any]]],
    rank: tuple[float, float],
    counter: int,
    record: dict[str, Any],
    limit: int,
) -> None:
    item = (rank, counter, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank > heap[0][0]:
        heapq.heapreplace(heap, item)


def build_registry(
    atlas_path: Path,
    policy_path: Path,
    macro_side: int | None,
    seed: int | None,
    quota_overrides: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = _load_policy(policy_path)
    sampling = policy["sampling"]
    macro_side = macro_side or int(sampling["macro_side_patches"])
    seed = seed if seed is not None else int(sampling["sampling_seed"])
    if macro_side <= 0:
        raise ValueError("macro_side must be positive")

    quotas = {str(name): int(count) for name, count in sampling["supplement_quotas"].items()}
    quotas.update(quota_overrides)
    reservoir_multiplier = int(sampling.get("supplement_reservoir_multiplier", 5))
    if reservoir_multiplier < 2:
        raise ValueError("supplement_reservoir_multiplier must be at least 2")

    base_winners: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}
    reservoirs: dict[str, list[tuple[tuple[float, float], int, dict[str, Any]]]] = {
        name: [] for name in quotas
    }
    seen_patch_ids: set[str] = set()
    total_records = 0
    eligible_records = 0
    counter = 0

    for record in _read_atlas(atlas_path):
        _validate_record(record)
        total_records += 1
        patch_id = record["patch_id"]
        if patch_id in seen_patch_ids:
            raise ValueError(f"duplicate patch_id in atlas: {patch_id}")
        seen_patch_ids.add(patch_id)
        if not record["eligible"]:
            continue
        eligible_records += 1
        counter += 1
        key = _macrocell(record, macro_side)
        base_rank = _stable_unit_hash(seed, patch_id)
        incumbent = base_winners.get(key)
        if incumbent is None or base_rank < incumbent[0]:
            base_winners[key] = (base_rank, record)

        strata = set(_parse_strata(record.get("strata")))
        for stratum, quota in quotas.items():
            if stratum not in strata:
                continue
            _push_candidate(
                reservoirs[stratum],
                _stratum_score(record, stratum, seed),
                counter,
                record,
                limit=quota * reservoir_multiplier,
            )

    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for _, record in base_winners.values():
        patch_id = record["patch_id"]
        selected[patch_id] = record
        reasons[patch_id].add("base:systematic_10x10")

    supplemental_summary: dict[str, dict[str, int]] = {}
    for stratum, quota in quotas.items():
        ranked = sorted(reservoirs[stratum], key=lambda item: item[0], reverse=True)
        matched = 0
        newly_added = 0
        for _, _, record in ranked:
            if matched >= quota:
                break
            patch_id = record["patch_id"]
            matched += 1
            if patch_id not in selected:
                selected[patch_id] = record
                newly_added += 1
            reasons[patch_id].add(f"supplement:{stratum}")
        supplemental_summary[stratum] = {
            "requested": quota,
            "matched": matched,
            "newly_added": newly_added,
            "reservoir_size": len(reservoirs[stratum]),
        }

    registry: list[dict[str, Any]] = []
    by_admin = Counter()
    by_ecoregion = Counter()
    by_reason = Counter()
    for patch_id in sorted(selected):
        record = dict(selected[patch_id])
        record["sampling_reasons"] = sorted(reasons[patch_id])
        record["sampling_seed"] = seed
        record["macro_side_patches"] = macro_side
        registry.append(record)
        by_admin[str(record.get("admin1", "unknown"))] += 1
        by_ecoregion[str(record.get("ecoregion", "unknown"))] += 1
        for reason in reasons[patch_id]:
            by_reason[reason] += 1

    report = {
        "atlas_path": str(atlas_path),
        "atlas_sha256": _sha256(atlas_path),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "total_atlas_records": total_records,
        "eligible_records": eligible_records,
        "base_selected": len(base_winners),
        "total_selected": len(registry),
        "supplemental": supplemental_summary,
        "selected_by_admin1": dict(sorted(by_admin.items())),
        "selected_by_ecoregion": dict(sorted(by_ecoregion.items())),
        "selected_by_reason": dict(sorted(by_reason.items())),
        "sampling_seed": seed,
        "macro_side_patches": macro_side,
    }
    return registry, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path, help="Quality-qualified candidate atlas JSONL.")
    parser.add_argument("--policy", required=True, type=Path, help="Self-contained sampling policy JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Selected registry JSONL.")
    parser.add_argument("--report", required=True, type=Path, help="Sampling report JSON.")
    parser.add_argument("--macro-side", type=int, default=None, help="Override policy macrocell side length.")
    parser.add_argument("--seed", type=int, default=None, help="Override policy sampling seed.")
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Override or add one supplemental quota, e.g. worldcover:wetland=3000.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the report without writing.")
    args = parser.parse_args()

    registry, report = build_registry(
        atlas_path=args.atlas,
        policy_path=args.policy,
        macro_side=args.macro_side,
        seed=args.seed,
        quota_overrides=_parse_quota_overrides(args.quota),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in registry:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
