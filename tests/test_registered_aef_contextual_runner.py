from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    path = Path(__file__).parents[1] / "scripts/eval/run_registered_aef_contextual_downstream.py"
    spec = importlib.util.spec_from_file_location("run_registered_aef_contextual", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_runs_all_contextual_gates_before_probe(monkeypatch) -> None:
    module = _load_runner()
    calls: list[str] = []
    matrix = {
        "tasks": ["building"],
        "shots": ["5"],
        "folds": [0],
        "seeds": [42],
        "spatial_split": "split.json",
        "spatial_split_sha256": "split",
        "manifest_sha256": "manifest",
        "statistics_registry_path": "tests/test_registered_aef_contextual_runner.py",
        "statistics_registry_sha256": "manifest",
        "label_root": "/labels",
        "labels": {"building": {"tree_sha256": "labels"}},
    }
    args = SimpleNamespace(
        matrix=Path("matrix.json"),
        embedding_root=Path("embeddings"),
        manifest=Path("manifest.json"),
        spatial_split=Path("split.json"),
        label_root=Path("/labels"),
        task="building",
        shot="5",
        fold=0,
        shot_seed=42,
    )
    monkeypatch.setattr(
        module, "load_contextual_matrix", lambda _: calls.append("matrix") or matrix
    )
    monkeypatch.setattr(module, "verify_frozen_shot_schedules", lambda _: calls.append("schedules"))
    monkeypatch.setattr(module, "verify_runtime_source_hashes", lambda *_: calls.append("sources"))
    monkeypatch.setattr(module, "verify_aef_embedding_export", lambda *_: calls.append("export"))
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: "split" if path.name == "split.json" else "manifest",
    )

    result = module.preflight(args)

    assert result is matrix
    assert calls == ["matrix", "schedules", "sources", "export"]
