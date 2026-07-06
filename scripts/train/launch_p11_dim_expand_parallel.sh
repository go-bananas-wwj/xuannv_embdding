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
LOG_ROOT="${LOG_ROOT:-/data/xuannv_embedding/logs/p11_dim_expand_20260706}"
mkdir -p "${LOG_ROOT}"

declare -A CONFIGS=(
  [p11a]="configs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11b]="configs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11c]="configs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706.yaml"
)

declare -A DEVICES=(
  [p11a]="0,1"
  [p11b]="2,3"
  [p11c]="4,5"
)

for name in p11a p11b p11c; do
  session="xuannv_${name}_dim_expand"
  config="${CONFIGS[$name]}"
  devices="${DEVICES[$name]}"
  log_file="${LOG_ROOT}/${name}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session ${session} already exists; skip."
    continue
  fi
  tmux new-session -d -s "${session}" \
    "cd '$PWD' && source '${CANN_ENV}' && export PYTHONPATH='$PWD/src:\${PYTHONPATH:-}' && export ASCEND_RT_VISIBLE_DEVICES='${devices}' && torchrun --standalone --nproc_per_node=2 scripts/train/train.py --config '${config}' --init-from '${INIT_FROM}' 2>&1 | tee '${log_file}'"
  echo "Started ${session}: devices=${devices}, config=${config}, log=${log_file}"
done
