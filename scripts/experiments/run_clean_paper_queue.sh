#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/workspace/xuannv
CONFIG_ROOT="$ROOT/configs/paper_20260715"
LOG_ROOT=/data/xuannv_embedding/logs/paper_clean_20260715
STATUS_FILE="$LOG_ROOT/status.tsv"

mkdir -p "$LOG_ROOT"
cd "$ROOT"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

wait_for_e5() {
  while pgrep -f '[p]aper_e5_p10c_recipe_harbin_scratch_20260714.yaml' >/dev/null; do
    printf '%s\twaiting_for_e5\n' "$(date -Iseconds)" >> "$STATUS_FILE"
    sleep 300
  done
}

run_one() {
  local devices=$1
  local port=$2
  local name=$3
  local log="$LOG_ROOT/$name.log"
  printf '%s\tstart\t%s\t%s\n' "$(date -Iseconds)" "$name" "$devices" >> "$STATUS_FILE"
  ASCEND_RT_VISIBLE_DEVICES="$devices" HCCL_IF_BASE_PORT="$port" \
    torchrun --standalone --master_port "$port" --nproc_per_node=2 \
      scripts/train/train.py --config "$CONFIG_ROOT/$name.yaml" > "$log" 2>&1
  local status=$?
  printf '%s\tend\t%s\texit=%s\n' "$(date -Iseconds)" "$name" "$status" >> "$STATUS_FILE"
  return "$status"
}

run_group() {
  local group=$1
  shift
  printf '%s\tgroup_start\t%s\n' "$(date -Iseconds)" "$group" >> "$STATUS_FILE"
  local pids=()
  while (( $# )); do
    run_one "$1" "$2" "$3" &
    pids+=("$!")
    shift 3
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  printf '%s\tgroup_end\t%s\tfailed=%s\n' "$(date -Iseconds)" "$group" "$failed" >> "$STATUS_FILE"
  return "$failed"
}

wait_for_e5

run_group group1 \
  0,1 35201 paper_clean_full_160_fold0_20260715 \
  2,3 35202 paper_clean_no_osm_160_fold0_20260715 \
  4,5 35203 paper_clean_coarse_osm_only_160_fold0_20260715 || exit 1

run_group group2 \
  0,1 35211 paper_clean_probe_nohardneg_160_fold0_20260715 \
  2,3 35212 paper_clean_no_highres_160_fold0_20260715 \
  4,5 35213 paper_clean_no_masking_160_fold0_20260715 || exit 1

run_group group3 \
  0,1 35221 paper_clean_full_40_fold0_20260715 \
  2,3 35222 paper_clean_full_80_fold0_20260715 || exit 1

printf '%s\tall_training_complete\n' "$(date -Iseconds)" >> "$STATUS_FILE"
