from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/train/watch_training_jobs.py"
SPEC = importlib.util.spec_from_file_location("training_watchdog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_torchrun_jobs_discovers_config_from_proc(monkeypatch, tmp_path: Path) -> None:
    proc = tmp_path / "123"
    proc.mkdir()
    config = tmp_path / "experiment.yaml"
    proc.joinpath("cmdline").write_bytes(
        b"python\0torchrun\0scripts/train/train.py\0--config\0"
        + str(config).encode()
        + b"\0"
    )
    monkeypatch.setattr(MODULE, "Path", lambda value: tmp_path if value == "/proc" else Path(value))

    assert MODULE._torchrun_jobs() == {123: config}


def test_write_json_replaces_atomically(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    MODULE._write_json(path, {"jobs": [1]})
    MODULE._write_json(path, {"jobs": [2]})

    assert path.read_text(encoding="utf-8").strip() == '{\n  "jobs": [\n    2\n  ]\n}'
    assert not path.with_suffix(".json.tmp").exists()
