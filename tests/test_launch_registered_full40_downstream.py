from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/eval/launch_registered_full40_downstream.sh"


def test_full40_launcher_dry_run_covers_registered_probe_matrix() -> None:
    """The launcher must enumerate the exact 5-fold, 3-task, 2-shot, 3-seed matrix."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("JOB\t")]
    assert len(jobs) == 90
    assert {f"fold={fold}" for fold in range(5)} <= set(" ".join(jobs).split())
    assert {f"task={task}" for task in ("building", "road", "water")} <= set(
        " ".join(jobs).split()
    )
    assert {"shot=5", "shot=10"} <= set(" ".join(jobs).split())
    assert {f"seed={seed}" for seed in (42, 43, 44)} <= set(" ".join(jobs).split())
    assert {f"device=npu:{device}" for device in range(6)} <= set(" ".join(jobs).split())


def test_full40_launcher_leaves_result_directory_creation_to_probe() -> None:
    """Registered probes require a fresh output directory for atomic result declaration."""
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'mkdir -p "$LOG_ROOT" "$output_root"' not in source
