#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke
SANDBOX=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815

if [[ ! -e /dev/davinci2 ]]; then
  echo "NPU 2 device is absent" >&2
  exit 19
fi
if [[ -n "$(fuser /dev/davinci2 2>/dev/null || true)" ]]; then
  echo "NPU 2 is busy; refusing smoke run" >&2
  exit 20
fi

source /usr/local/Ascend/cann-9.0.0/set_env.sh
CANN_PYTHONPATH="${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${WORKTREE}/src:${WORKTREE}/downstreams:${WORKTREE}"
export PYTHONPATH="${PYTHONPATH}:${CANN_PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=2
export XUANNV_SMOKE_DEVICE=npu:0
export WANDB_MODE=disabled

cd "${WORKTREE}"
"${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
  --stage npu-smoke 2>&1 | tee "${SANDBOX}/logs/npu_smoke.log"
