from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


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
        data_root=tmp_path / "data-root",
        manifest=tmp_path / "manifest-location" / "manifest.json",
        tasks=["building"],
        fold=0,
        seed=17,
        output_root=tmp_path / "output",
    )


def test_default_paths_target_p10c_epoch800_and_new_comparison_output(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    args = MODULE.parse_args()

    assert args.embedding_root == Path(
        "/data/xuannv_embedding/embeddings/production/p10c_epoch800_202604/"
        "artifacts/haidian-embedding-v1/embeddings/"
        "haidian_202512_202605_p10c_epoch800"
    )
    assert args.output_root == Path(
        "/data/xuannv_embedding/experiments/production/"
        "haidian_pu_query_3polygon_compare_20260726"
    )


def write_embedding(root: Path, patch_id: str, month: str, channels: int) -> np.ndarray:
    feature = np.arange(channels * 16 * 16, dtype=np.float32).reshape(channels, 16, 16)
    feature += sum(map(ord, patch_id))
    path = root / "haidian" / patch_id / f"{month}_embedding_map.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(feature), path)
    return feature


def write_manifest(args: SimpleNamespace, patch_ids: list[str]) -> None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            [
                {
                    "patch_id": patch_id,
                    "s2": [f"sources/{patch_id}_202604_scene.tif"],
                }
                for patch_id in patch_ids
            ]
        ),
        encoding="utf-8",
    )


def test_feature_specs_describe_the_three_protocol_feature_sources(tmp_path: Path) -> None:
    args = comparison_args(tmp_path)

    specs = MODULE.feature_specs(args)

    assert specs["xuannv"].channels == 64
    assert specs["aef"].month == "202512"
    assert specs["traditional"].channels == 42
    assert specs["traditional"].kind == "fixed_highres_feature_map"
    assert specs["traditional"].data_root == args.data_root


def test_pick_supports_is_deterministic_for_polygon_candidates() -> None:
    candidates = [
        MODULE.PolygonSupport(f"patch-{index}", np.eye(4, dtype=bool))
        for index in range(5)
    ]

    assert MODULE.pick_supports(candidates, 3, 91) == MODULE.pick_supports(candidates, 3, 91)


def test_load_feature_reads_embedding_and_manifest_relative_to_configured_data_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = comparison_args(tmp_path)
    patch_id = "patch-1"
    expected_embedding = write_embedding(args.embedding_root, patch_id, "202604", 64)
    write_manifest(args, [patch_id])
    MODULE.MANIFEST_CACHE.clear()
    observed_source_paths: list[Path] = []

    def fake_fixed_highres(record, month: str) -> np.ndarray:
        assert month == "202604"
        observed_source_paths.extend(record.sources["s2"])
        return np.zeros((42, 16, 16), dtype=np.float32)

    monkeypatch.setattr(MODULE, "fixed_highres_feature_map", fake_fixed_highres)
    specs = MODULE.feature_specs(args)

    np.testing.assert_array_equal(MODULE.load_feature(specs["xuannv"], patch_id), expected_embedding)
    np.testing.assert_array_equal(
        MODULE.load_feature(specs["traditional"], patch_id),
        np.zeros((42, 16, 16), dtype=np.float32),
    )
    assert observed_source_paths == [args.data_root / "sources" / f"{patch_id}_202604_scene.tif"]


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
    all_patch_ids = [candidate.patch_id for candidate in candidates] + test_ids
    write_manifest(args, all_patch_ids)
    MODULE.MANIFEST_CACHE.clear()
    for patch_id in all_patch_ids:
        write_embedding(args.embedding_root, patch_id, "202604", 64)
        write_embedding(args.aef_embedding_root, patch_id, "202512", 64)
    traditional_loaded: set[str] = set()

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

    def fake_fixed_highres(record, _month: str) -> np.ndarray:
        traditional_loaded.add(record.patch_id)
        assert record.sources["s2"] == [
            args.data_root / "sources" / f"{record.patch_id}_202604_scene.tif"
        ]
        seed = sum(map(ord, record.patch_id))
        return np.random.default_rng(seed).normal(size=(42, 16, 16)).astype(np.float32)

    def synthetic_mask(_root: Path, patch_id: str) -> np.ndarray:
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        if patch_id == "test-2":
            mask[15, 15] = True
        return mask

    monkeypatch.setattr(MODULE, "fixed_highres_feature_map", fake_fixed_highres)
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
    assert traditional_loaded == selected_support_ids | set(test_ids)
