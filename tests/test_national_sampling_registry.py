from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/data/build_national_sampling_registry.py"
SPEC = importlib.util.spec_from_file_location("national_sampling_registry", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_one_chip_per_macrocell_and_supplement_reasons(tmp_path: Path) -> None:
    policy = {
        "sampling": {
            "macro_side_patches": 10,
            "sampling_seed": 7,
            "supplement_reservoir_multiplier": 3,
            "supplement_quotas": {"worldcover:wetland": 2},
        }
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    records = []
    for row in range(10):
        for col in range(10):
            records.append(
                {
                    "patch_id": f"A_{row}_{col}",
                    "grid_id": "A",
                    "grid_row": row,
                    "grid_col": col,
                    "eligible": True,
                    "admin1": "province-a",
                    "strata": ["worldcover:wetland"] if col < 2 else [],
                    "stratum_scores": {"worldcover:wetland": float(col + 1)},
                }
            )
    records.extend(
        [
            {
                "patch_id": "A_10_0",
                "grid_id": "A",
                "grid_row": 10,
                "grid_col": 0,
                "eligible": True,
                "admin1": "province-a",
                "strata": [],
            },
            {
                "patch_id": "excluded",
                "grid_id": "A",
                "grid_row": 10,
                "grid_col": 1,
                "eligible": False,
                "strata": ["worldcover:wetland"],
            },
        ]
    )
    atlas_path = tmp_path / "atlas.jsonl"
    atlas_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    registry, report = MODULE.build_registry(
        atlas_path=atlas_path,
        policy_path=policy_path,
        macro_side=None,
        seed=None,
        quota_overrides={},
    )

    selected = {record["patch_id"]: record for record in registry}
    assert report["base_selected"] == 2
    assert report["supplemental"]["worldcover:wetland"]["matched"] == 2
    assert "excluded" not in selected
    assert any(
        "supplement:worldcover:wetland" in record["sampling_reasons"]
        for record in registry
    )
