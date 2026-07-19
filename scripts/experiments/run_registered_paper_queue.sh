#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/workspace/xuannv
CONFIG_ROOT="$ROOT/configs/paper_registered_20260716"
OUTPUT_ROOT=/data/xuannv_embedding/outputs/paper_registered_20260716
LOG_ROOT=/data/xuannv_embedding/logs/paper_registered_20260716
STATUS_FILE="$LOG_ROOT/status.tsv"
EXPECTED_CONFIGS=40
MAX_JOB_RETRIES=${MAX_JOB_RETRIES:-3}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
ACTIVE_LANES=${ACTIVE_LANES:-0,1,2}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

mapfile -t CONFIGS < <(
  python - "$CONFIG_ROOT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
paths = sorted(root.glob("*.yaml"))

def size(path: Path) -> int:
    match = re.search(r"_(40|80|150)_fold", path.stem)
    if match is None:
        raise SystemExit(f"cannot infer training size from {path.name}")
    return int(match.group(1))

# Longest-processing-time order keeps the three fixed NPU lanes balanced.
for path in sorted(paths, key=lambda value: (-size(value), value.name)):
    print(path.stem)
PY
)

if (( ${#CONFIGS[@]} != EXPECTED_CONFIGS )); then
  printf 'expected %d configs, found %d\n' "$EXPECTED_CONFIGS" "${#CONFIGS[@]}" >&2
  exit 2
fi

declare -a LANE0=() LANE1=() LANE2=()
declare -a LOADS=(0 0 0)

for name in "${CONFIGS[@]}"; do
  size=$(sed -nE 's/.*_(40|80|150)_fold.*/\1/p' <<< "$name")
  lane=0
  if (( LOADS[1] < LOADS[lane] )); then lane=1; fi
  if (( LOADS[2] < LOADS[lane] )); then lane=2; fi
  case "$lane" in
    0) LANE0+=("$name") ;;
    1) LANE1+=("$name") ;;
    2) LANE2+=("$name") ;;
  esac
  LOADS[$lane]=$(( LOADS[$lane] + size ))
done

printf '%s\tqueue_start\tconfigs=%d\tgit=%s\tlane_loads=%s,%s,%s\n' \
  "$(date -Iseconds)" "${#CONFIGS[@]}" "$(git rev-parse HEAD)" \
  "${LOADS[0]}" "${LOADS[1]}" "${LOADS[2]}" >> "$STATUS_FILE"

run_one() {
  local lane=$1
  local devices=$2
  local port=$3
  local name=$4
  local config="$CONFIG_ROOT/$name.yaml"
  local output="$OUTPUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  local resume_checkpoint=""
  local -a resume_args=()

  if [[ -f "$output/epoch_800.pt" ]]; then
    printf '%s\tskip_complete\t%s\tlane=%s\n' "$(date -Iseconds)" "$name" "$lane" >> "$STATUS_FILE"
    return 0
  fi
  if [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
    resume_checkpoint=$(find "$output" -maxdepth 1 -type f -name 'epoch_*.pt' -printf '%f\n' \
      | sort -V | tail -1)
    if [[ -z "$resume_checkpoint" ]]; then
      printf '%s\trefuse_partial_without_epoch_checkpoint\t%s\tlane=%s\toutput=%s\n' \
        "$(date -Iseconds)" "$name" "$lane" "$output" >> "$STATUS_FILE"
      return 3
    fi
    resume_checkpoint="$output/$resume_checkpoint"
    resume_args=(--resume "$resume_checkpoint")
    printf '%s\tresume\t%s\tlane=%s\tcheckpoint=%s\n' \
      "$(date -Iseconds)" "$name" "$lane" "$resume_checkpoint" >> "$STATUS_FILE"
  fi

  printf '%s\tstart\t%s\tlane=%s\tdevices=%s\tconfig_sha256=%s\n' \
    "$(date -Iseconds)" "$name" "$lane" "$devices" "$(sha256sum "$config" | cut -d' ' -f1)" \
    >> "$STATUS_FILE"

  ASCEND_RT_VISIBLE_DEVICES="$devices" HCCL_IF_BASE_PORT="$port" \
    torchrun --standalone --master_port "$port" --nproc_per_node=2 \
      scripts/train/train.py --config "$config" "${resume_args[@]}" >> "$log" 2>&1
  local status=$?

  if (( status == 0 )) && [[ ! -f "$output/epoch_800.pt" ]]; then
    status=4
    printf '%s\tmissing_final_checkpoint\t%s\tlane=%s\n' \
      "$(date -Iseconds)" "$name" "$lane" >> "$STATUS_FILE"
  fi
  printf '%s\tend\t%s\tlane=%s\texit=%s\n' \
    "$(date -Iseconds)" "$name" "$lane" "$status" >> "$STATUS_FILE"
  return "$status"
}

run_lane() {
  local lane=$1
  local devices=$2
  local port=$3
  shift 3
  local name
  local attempt
  local status
  local lane_failed=0

  printf '%s\tlane_start\tlane=%s\tdevices=%s\tjobs=%d\n' \
    "$(date -Iseconds)" "$lane" "$devices" "$#" >> "$STATUS_FILE"
  for name in "$@"; do
    attempt=0
    while true; do
      run_one "$lane" "$devices" "$port" "$name"
      status=$?
      if (( status == 0 )); then
        break
      fi
      attempt=$((attempt + 1))
      if (( attempt > MAX_JOB_RETRIES )); then
        printf '%s\tlane_failed\tlane=%s\tname=%s\texit=%s\tattempts=%s\n' \
          "$(date -Iseconds)" "$lane" "$name" "$status" "$attempt" >> "$STATUS_FILE"
        lane_failed=1
        break
      fi
      printf '%s\tretry\t%s\tlane=%s\texit=%s\tattempt=%s/%s\tdelay=%s\n' \
        "$(date -Iseconds)" "$name" "$lane" "$status" "$attempt" "$MAX_JOB_RETRIES" \
        "$RETRY_DELAY_SECONDS" >> "$STATUS_FILE"
      sleep "$RETRY_DELAY_SECONDS"
    done
  done
  printf '%s\tlane_complete\tlane=%s\tfailed=%s\n' \
    "$(date -Iseconds)" "$lane" "$lane_failed" >> "$STATUS_FILE"
  return "$lane_failed"
}

lane_enabled() {
  [[ ",$ACTIVE_LANES," == *",$1,"* ]]
}

declare -a lane_pids=()
if lane_enabled 0; then run_lane 0 0,1 35401 "${LANE0[@]}" & lane_pids+=("$!"); fi
if lane_enabled 1; then run_lane 1 2,3 35402 "${LANE1[@]}" & lane_pids+=("$!"); fi
if lane_enabled 2; then run_lane 2 4,5 35403 "${LANE2[@]}" & lane_pids+=("$!"); fi
if (( ${#lane_pids[@]} == 0 )); then
  printf 'ACTIVE_LANES must select at least one of 0,1,2: %s\n' "$ACTIVE_LANES" >&2
  exit 2
fi

failed=0
for pid in "${lane_pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done

printf '%s\tqueue_end\tfailed=%s\n' "$(date -Iseconds)" "$failed" >> "$STATUS_FILE"
exit "$failed"
