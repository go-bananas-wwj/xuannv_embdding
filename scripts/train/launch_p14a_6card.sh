#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/v3_p14a_proto_boundary_haidian_202512_202605_20260710.yaml}"
INIT_FROM="${INIT_FROM:-/data/xuannv_embedding/outputs/v2_p13a_c_latent_dim_haidian_202512_202605_20260708/epoch_800.pt}"
LOG_ROOT="${LOG_ROOT:-/data/xuannv_embedding/logs/p14a_20260710}"
SESSION="${SESSION:-xuannv_p14a_6card}"
DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
HCCL_PORT="${HCCL_IF_BASE_PORT:-35000}"
IFS=',' read -r -a DEVICE_ARRAY <<< "${DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#DEVICE_ARRAY[@]}}"
mkdir -p "${LOG_ROOT}"

if [[ "${NPROC_PER_NODE}" -ne "${#DEVICE_ARRAY[@]}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} does not match device count ${#DEVICE_ARRAY[@]} (${DEVICES})" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists; attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "bash -lc \"set -euo pipefail; cd '$PWD'; source '${CANN_ENV}'; export PYTHONPATH='$PWD/src:'\\\${PYTHONPATH:-}; export ASCEND_RT_VISIBLE_DEVICES='${DEVICES}'; export HCCL_IF_BASE_PORT='${HCCL_PORT}'; python -c 'import tbe; print(\\\"tbe import ok:\\\", tbe.__file__)'; torchrun --standalone --nproc_per_node='${NPROC_PER_NODE}' scripts/train/train.py --config '${CONFIG}' --init-from '${INIT_FROM}' 2>&1 | tee '${LOG_ROOT}/p14a_6card.log'\""

echo "Started ${SESSION}"
echo "  devices=${DEVICES}"
echo "  nproc_per_node=${NPROC_PER_NODE}"
echo "  hccl_port=${HCCL_PORT}"
echo "  config=${CONFIG}"
echo "  init_from=${INIT_FROM}"
echo "  log=${LOG_ROOT}/p14a_6card.log"
