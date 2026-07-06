#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

LOG_ROOT="${LOG_ROOT:-/data/xuannv_embedding/logs/p11_dim_expand_20260706}"
mkdir -p "${LOG_ROOT}"

declare -A CONFIGS=(
  [p11a]="configs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11b]="configs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11c]="configs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706.yaml"
)

declare -A OUTPUT_DIRS=(
  [p11a]="/data/xuannv_embedding/outputs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706"
  [p11b]="/data/xuannv_embedding/outputs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706"
  [p11c]="/data/xuannv_embedding/outputs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706"
)

declare -A DEVICES=(
  [p11a]="0,1"
  [p11b]="2,3"
  [p11c]="4,5"
)

declare -A HCCL_PORTS=(
  [p11a]="30000"
  [p11b]="31000"
  [p11c]="32000"
)

is_original_session_running() {
  local name="$1"
  tmux has-session -t "xuannv_${name}_dim_expand" 2>/dev/null
}

launch_resume() {
  local name="$1"
  local session="xuannv_${name}_dim_expand_600"
  local config="${CONFIGS[$name]}"
  local output_dir="${OUTPUT_DIRS[$name]}"
  local resume_ckpt="${output_dir}/epoch_400.pt"
  local devices="${DEVICES[$name]}"
  local hccl_port="${HCCL_PORTS[$name]}"
  local log_file="${LOG_ROOT}/${name}_resume_400_to_600.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "$(date '+%F %T') ${session} already running; skip."
    return
  fi

  tmux new-session -d -s "${session}" \
    "bash -lc \"set -euo pipefail; cd '$PWD'; source '${CANN_ENV}'; export PYTHONPATH='$PWD/src:'\\\${PYTHONPATH:-}; export ASCEND_RT_VISIBLE_DEVICES='${devices}'; export HCCL_IF_BASE_PORT='${hccl_port}'; python -c 'import tbe; print(\\\"tbe import ok:\\\", tbe.__file__)'; torchrun --standalone --nproc_per_node=2 scripts/train/train.py --config '${config}' --resume '${resume_ckpt}' 2>&1 | tee '${log_file}'\""
  echo "$(date '+%F %T') launched ${session}: resume=${resume_ckpt}, devices=${devices}, log=${log_file}"
}

while true; do
  remaining=0
  for name in p11a p11b p11c; do
    resume_ckpt="${OUTPUT_DIRS[$name]}/epoch_400.pt"
    resume_session="xuannv_${name}_dim_expand_600"
    if tmux has-session -t "${resume_session}" 2>/dev/null; then
      continue
    fi
    if [[ -f "${resume_ckpt}" ]] && ! is_original_session_running "${name}"; then
      launch_resume "${name}"
      continue
    fi
    remaining=$((remaining + 1))
    ckpt_status="no"
    original_status="no"
    if [[ -f "${resume_ckpt}" ]]; then
      ckpt_status="yes"
    fi
    if is_original_session_running "${name}"; then
      original_status="yes"
    fi
    echo "$(date '+%F %T') waiting ${name}: ckpt=${ckpt_status}, original_session=${original_status}"
  done

  if [[ "${remaining}" -eq 0 ]]; then
    echo "$(date '+%F %T') all P11 resume jobs launched."
    break
  fi
  sleep "${POLL_SECONDS:-300}"
done
