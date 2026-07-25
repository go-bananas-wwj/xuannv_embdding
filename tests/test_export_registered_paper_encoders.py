from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/eval/export_registered_paper_encoders.sh"


def test_export_launcher_dry_run_enumerates_five_folds_and_six_shards() -> None:
    """One ablation family must export every fold without shard/device collisions."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--family", "no_osm_150", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("EXPORT\t")]
    assert len(jobs) == 30
    for fold in range(5):
        fold_jobs = [line for line in jobs if f"fold={fold}" in line]
        assert len(fold_jobs) == 6
        assert {f"shard={shard}" for shard in range(6)} <= set(" ".join(fold_jobs).split())
        assert {f"device=npu:{device}" for device in range(6)} <= set(
            " ".join(fold_jobs).split()
        )


def test_export_launcher_refuses_ambiguous_roots_and_executes_recorded_commands() -> None:
    """A sealed export must have one root, exact commands, and one metadata writer."""
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'if (( ${#existing[@]} != 0 )); then' in source
    assert 'existing=()' not in source
    assert 'eval "${commands[$shard]}"' in source
    assert 'commands+=("ASCEND_RT_VISIBLE_DEVICES=${shard} python ' not in source
    assert '--skip-meta' in source
    assert 'write_export_meta "$export_root" "$config" "$checkpoint"' in source


def test_precompute_rejects_skip_meta_without_valid_shard_arguments() -> None:
    """A root metadata skip is valid only while a sharded coordinator will replace it."""
    script = ROOT / "downstreams/scripts/precompute_embeddings.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "missing.yaml",
            "--regions",
            "haidian",
            "--output-root",
            "unused-output",
            "--checkpoint",
            "missing.pt",
            "--skip-meta",
        ],
        cwd=ROOT,
        env={
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}:{ROOT / 'downstreams'}",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--skip-meta requires" in result.stderr
