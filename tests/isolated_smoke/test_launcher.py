from __future__ import annotations

import re
from pathlib import Path

LAUNCHER = Path("scripts/smoke/run_china_v1_isolated_fusion_smoke.sh")
EXPECTED_WORKTREE = "/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke"
EXPECTED_SANDBOX = "/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815"


def _launcher_source() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_is_fixed_to_the_isolated_worktree_and_sandbox() -> None:
    """启动器不得从主 worktree 或生产输出目录运行。"""
    source = _launcher_source()

    assert f"WORKTREE={EXPECTED_WORKTREE}" in source
    assert f"SANDBOX={EXPECTED_SANDBOX}" in source
    assert 'cd "${WORKTREE}"' in source
    assert '"${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner' in source
    assert "/data/xuannv_embedding/outputs" not in source
    assert "/data/xuannv_embedding/processed" not in source
    assert "/data/xuannv_embedding/embeddings" not in source


def test_launcher_exposes_only_physical_npu2_as_logical_npu0() -> None:
    """物理卡 2 必须是唯一可见设备并映射为逻辑 npu:0。"""
    source = _launcher_source()

    assert "[[ ! -e /dev/davinci2 ]]" in source
    assert "fuser /dev/davinci2" not in source
    assert "_physical_npu2_is_idle" in source
    assert '"${SANDBOX}/env/bin/python" -' in source
    assert "export ASCEND_RT_VISIBLE_DEVICES=2" in source
    assert "export XUANNV_SMOKE_DEVICE=npu:0" in source


def test_launcher_uses_cann_sandbox_imports_and_disabled_wandb() -> None:
    """启动器必须固定 CANN 9、隔离导入路径和禁用的外部记录。"""
    source = _launcher_source()

    assert "source /usr/local/Ascend/cann-9.0.0/set_env.sh" in source
    assert "export PYTHONNOUSERSITE=1" in source
    assert 'export PYTHONPATH="${WORKTREE}/src:${WORKTREE}/downstreams:${WORKTREE}"' in source
    assert "export WANDB_MODE=disabled" in source


def test_launcher_keeps_cann_python_modules_after_prioritizing_the_worktree() -> None:
    """worktree 导入优先级不得移除 CANN 编译器所需的 tbe/te 路径。"""
    source = _launcher_source()

    cann_source = "source /usr/local/Ascend/cann-9.0.0/set_env.sh"
    cann_capture = 'CANN_PYTHONPATH="${PYTHONPATH:-}"'
    isolated_export = 'export PYTHONPATH="${WORKTREE}/src:${WORKTREE}/downstreams:${WORKTREE}"'
    cann_append = 'export PYTHONPATH="${PYTHONPATH}:${CANN_PYTHONPATH}"'
    assert cann_capture in source
    assert isolated_export in source
    assert cann_append in source
    assert (
        source.index(cann_source)
        < source.index(cann_capture)
        < source.index(isolated_export)
        < source.index(cann_append)
    )


def test_launcher_runs_the_only_supported_smoke_in_the_foreground() -> None:
    """唯一入口必须直接以前台 bash 脚本祖先调用 npu-smoke。"""
    source = _launcher_source()

    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "--config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml" in source
    assert "--stage npu-smoke" in source
    assert "experiments.china_v1_fusion_smoke.safe_tee" in source
    assert "--reserve" in source
    assert "--stream-reserved" in source
    assert "| tee " not in source
    assert "READY_TO_SEAL" in source
    assert "TEE_COMPLETE" in source
    assert "READY_TO_SEAL exists without TEE_COMPLETE" in source
    assert "TEE_COMPLETE exists without READY_TO_SEAL" in source
    assert "--mark-tee-complete" in source
    assert "--finalize-seal" in source
    assert (
        source.index("--stage npu-smoke")
        < source.index("--mark-tee-complete")
        < source.index("--finalize-seal")
    )
    assert source.rstrip().endswith("--finalize-seal")
    assert "torchrun" not in source
    assert "nohup" not in source
    assert not re.search(r"(?m)&\s*(?:#.*)?$", source)
    assert not re.search(r"(?m)^\s*exec\s", source)
