#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

INIT_FROM="${INIT_FROM:-/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt}"
LOG_ROOT="${LOG_ROOT:-/data/xuannv_embedding/logs/p12_context_20260707}"
mkdir -p "${LOG_ROOT}"

declare -A CONFIGS=(
  [p12b]="configs/v2_p12b_context160_center128_haidian_202512_202605_20260707.yaml"
  [p12c]="configs/v2_p12c_context160_semantic_rich_haidian_202512_202605_20260707.yaml"
)

declare -A DEVICES=(
  [p12b]="0,1,2"
  [p12c]="3,4,5"
)

declare -A HCCL_PORTS=(
  [p12b]="33000"
  [p12c]="34000"
)

for name in p12b p12c; do
  session="xuannv_${name}_context"
  config="${CONFIGS[$name]}"
  devices="${DEVICES[$name]}"
  hccl_port="${HCCL_PORTS[$name]}"
  log_file="${LOG_ROOT}/${name}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session ${session} already exists; skip."
    continue
  fi
  tmux new-session -d -s "${session}" \
    "bash -lc \"set -euo pipefail; cd '$PWD'; source '${CANN_ENV}'; export PYTHONPATH='$PWD/src:'\\\${PYTHONPATH:-}; export ASCEND_RT_VISIBLE_DEVICES='${devices}'; export HCCL_IF_BASE_PORT='${hccl_port}'; python -c 'import tbe; print(\\\"tbe import ok:\\\", tbe.__file__)'; torchrun --standalone --nproc_per_node=3 scripts/train/train.py --config '${config}' --init-from '${INIT_FROM}' 2>&1 | tee '${log_file}'\""
  echo "Started ${session}: devices=${devices}, hccl_port=${hccl_port}, config=${config}, log=${log_file}"
done
