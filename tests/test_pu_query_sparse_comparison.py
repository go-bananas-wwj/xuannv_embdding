from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/eval/run_pu_query_sparse_eval.py"
SPEC = importlib.util.spec_from_file_location("pu_query_sparse_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def comparison_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_root=tmp_path / "xuannv",
        aef_embedding_root=tmp_path / "aef",
        data_root=tmp_path / "processed" / "haidian",
        manifest=tmp_path / "processed" / "haidian" / "manifest.json",
        tasks=["building"],
        fold=0,
        seed=17,
        output_root=tmp_path / "output",
    )


def test_feature_specs_describe_the_three_protocol_feature_sources(tmp_path: Path) -> None:
    args = comparison_args(tmp_path)

    specs = MODULE.feature_specs(args)

    assert specs["xuannv"].channels == 64
    assert specs["aef"].month == "202512"
    assert specs["traditional"].channels == 42
    assert specs["traditional"].kind == "fixed_highres_feature_map"


def test_pick_supports_is_deterministic_for_polygon_candidates() -> None:
    candidates = [
        MODULE.PolygonSupport(f"patch-{index}", np.eye(4, dtype=bool))
        for index in range(5)
    ]

    assert MODULE.pick_supports(candidates, 3, 91) == MODULE.pick_supports(candidates, 3, 91)


def test_comparison_reuses_each_task_supports_and_test_ids_for_all_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = comparison_args(tmp_path)
    support_mask = np.zeros((16, 16), dtype=bool)
    support_mask[7:9, 7:9] = True
    candidates = [
        MODULE.PolygonSupport(f"support-{index}", support_mask.copy())
        for index in range(4)
    ]
    test_ids = ["test-1", "test-2"]
    loaded: dict[str, set[str]] = {}

    monkeypatch.setattr(
        MODULE,
        "split_for",
        lambda _root, _fold: {
            "train": [candidate.patch_id for candidate in candidates],
            "val": ["val-1"],
            "test": test_ids,
        },
    )
    monkeypatch.setattr(MODULE, "collect_components", lambda _root, _train: candidates)

    def synthetic_feature(spec, patch_id: str) -> np.ndarray:
        loaded.setdefault(spec.name, set()).add(patch_id)
        seed = sum(map(ord, f"{spec.name}:{patch_id}"))
        return np.random.default_rng(seed).normal(size=(spec.channels, 16, 16)).astype(np.float32)

    def synthetic_mask(_root: Path, patch_id: str) -> np.ndarray:
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        if patch_id == "test-2":
            mask[15, 15] = True
        return mask

    monkeypatch.setattr(MODULE, "load_feature", synthetic_feature)
    monkeypatch.setattr(MODULE, "load_mask", synthetic_mask)

    payload = MODULE.run_comparison(args)

    output_payload = json.loads((args.output_root / "results.json").read_text(encoding="utf-8"))
    assert output_payload == payload
    assert payload["protocol"] == {
        "polygon_count": 3,
        "fold": 0,
        "shared_supports": True,
        "test_patch_count": 2,
    }
    assert {row["feature"] for row in payload["rows"]} == {"xuannv", "aef", "traditional"}
    xuannv_support_ids = next(
        tuple(row["support_patch_ids"])
        for row in payload["rows"]
        if row["feature"] == "xuannv"
    )
    assert {tuple(row["support_patch_ids"]) for row in payload["rows"]} == {
        xuannv_support_ids
    }
    assert {tuple(row["test_patch_ids"]) for row in payload["rows"]} == {tuple(test_ids)}
    selected_support_ids = set(payload["rows"][0]["support_patch_ids"])
    assert loaded == {
        "xuannv": selected_support_ids | set(test_ids),
        "aef": selected_support_ids | set(test_ids),
        "traditional": selected_support_ids | set(test_ids),
    }
