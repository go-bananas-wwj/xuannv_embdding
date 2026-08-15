#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke
SANDBOX=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
READY_TO_SEAL=${SANDBOX}/READY_TO_SEAL
TEE_COMPLETE=${SANDBOX}/TEE_COMPLETE
LOG=${SANDBOX}/logs/npu_smoke.log

if [[ ! -e /dev/davinci2 ]]; then
  echo "NPU 2 device is absent" >&2
  exit 19
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
if ! "${SANDBOX}/env/bin/python" - <<'PY'
from experiments.china_v1_fusion_smoke.runner import (
    _physical_npu2_exists,
    _physical_npu2_is_idle,
)

if not _physical_npu2_exists():
    raise SystemExit("NPU 2 device is absent")
if not _physical_npu2_is_idle():
    raise SystemExit("NPU 2 is busy; refusing smoke run")
PY
then
  echo "NPU 2 is busy or occupancy is unknown; refusing smoke run" >&2
  exit 20
fi

if [[ -f "${READY_TO_SEAL}" && ! -f "${TEE_COMPLETE}" ]]; then
  echo "READY_TO_SEAL exists without TEE_COMPLETE; refusing unsafe recovery" >&2
  exit 21
fi
if [[ -f "${TEE_COMPLETE}" && ! -f "${READY_TO_SEAL}" ]]; then
  echo "TEE_COMPLETE exists without READY_TO_SEAL; refusing inconsistent recovery" >&2
  exit 22
fi

if [[ ! -f "${READY_TO_SEAL}" && ! -f "${TEE_COMPLETE}" ]]; then
  "${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.safe_tee \
    --sandbox-root "${SANDBOX}" --log "${LOG}" --reserve
  "${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner \
    --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
    --stage npu-smoke 2>&1 | "${SANDBOX}/env/bin/python" \
      -m experiments.china_v1_fusion_smoke.safe_tee \
      --sandbox-root "${SANDBOX}" --log "${LOG}" --stream-reserved
  "${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner \
    --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
    --mark-tee-complete
fi

"${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
  --finalize-seal
