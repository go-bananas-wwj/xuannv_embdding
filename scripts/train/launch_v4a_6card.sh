#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
source "${CANN_ENV}"

CONFIG="${CONFIG:-configs/v4a_distill_scratch_haidian_harbin_202512_202605_20260711.yaml}"
LOG_ROOT="${LOG_ROOT:-/data/xuannv_embedding/logs/v4a_20260711}"
SESSION="${SESSION:-xuannv_v4a_6card}"
DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
HCCL_PORT="${HCCL_IF_BASE_PORT:-35000}"
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICE_ARRAY[@]}}"
mkdir -p "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists" >&2
  exit 1
fi

# 从零训练：无 --init-from
tmux new-session -d -s "${SESSION}" \
  "bash -lc \"set -euo pipefail; cd '$PWD'; source '${CANN_ENV}'; export PYTHONPATH='$PWD/src:'\\\${PYTHONPATH:-}; export ASCEND_RT_VISIBLE_DEVICES='${DEVICES}'; export HCCL_IF_BASE_PORT='${HCCL_PORT}'; python -c 'import tbe; print(\\\"tbe ok\\\")'; torchrun --standalone --nproc_per_node='${NPROC_PER_NODE}' scripts/train/train.py --config '${CONFIG}' 2>&1 | tee '${LOG_ROOT}/v4a_6card.log'\""

echo "Started ${SESSION} devices=${DEVICES} log=${LOG_ROOT}/v4a_6card.log"
