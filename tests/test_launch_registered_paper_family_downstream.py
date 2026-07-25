from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/eval/launch_registered_paper_family_downstream.sh"


def test_ablation_launcher_dry_run_preserves_the_registered_probe_matrix() -> None:
    """Every ablation must use the same 90-cell downstream protocol as the full recipe."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--family", "no_osm_150", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("JOB\t")]
    assert len(jobs) == 90
    assert all("family=no_osm_150" in line for line in jobs)
    assert {f"fold={fold}" for fold in range(5)} <= set(" ".join(jobs).split())
    assert {f"task={task}" for task in ("building", "road", "water")} <= set(
        " ".join(jobs).split()
    )
    assert {"shot=5", "shot=10"} <= set(" ".join(jobs).split())
    assert {f"seed={seed}" for seed in (42, 43, 44)} <= set(" ".join(jobs).split())
