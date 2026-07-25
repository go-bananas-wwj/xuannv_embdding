#!/usr/bin/env bash
# Run the frozen full_40 probe matrix in deterministic six-job waves.
set -euo pipefail

ROOT=/root/workspace/xuannv
EVAL_ROOT=/data/xuannv_embedding/experiments/paper_registered_eval_20260725
EMBEDDING_ROOT="$EVAL_ROOT/embeddings"
REGISTRY="$EVAL_ROOT/registry/results_v3.jsonl"
LOG_ROOT="$EVAL_ROOT/logs/full40_downstream"
MANIFEST=/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
LABEL_ROOT=/data/xuannv_embedding/processed/haidian/labels
SPLIT="$ROOT/configs/eval/haidian_spatial_5fold_buffer1_seed42.json"
EMBEDDING_REGISTRY="$ROOT/configs/eval/registered_embedding_exports_20260725.json"
SCHEDULE_ROOT="$EVAL_ROOT/shared_shot_schedules_v2"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

cd "$ROOT"
if [[ "$DRY_RUN" == false ]]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/downstreams:${PYTHONPATH:-}"

declare -a JOBS=()
for fold in 0 1 2 3 4; do
  for task in building road water; do
    for shot in 5 10; do
      for seed in 42 43 44; do
        JOBS+=("$fold|$task|$shot|$seed")
      done
    done
  done
done

if (( ${#JOBS[@]} != 90 )); then
  echo "internal error: expected 90 jobs, got ${#JOBS[@]}" >&2
  exit 3
fi

run_job() {
  local spec=$1
  local device=$2
  IFS='|' read -r fold task shot seed <<< "$spec"
  local stem="paper_registered_full_40_fold${fold}_20260716"
  local embedding_root="$EMBEDDING_ROOT/20260725_${stem}_best_full40_fold${fold}"
  local output_root="$EVAL_ROOT/probes_v3/full40_fold${fold}/${task}/shot_${shot}/seed_${seed}"
  local schedule="$SCHEDULE_ROOT/${task}_fold${fold}_seed${seed}.json"
  local log="$LOG_ROOT/fold${fold}_${task}_shot${shot}_seed${seed}.log"

  printf 'JOB\tfold=%s task=%s shot=%s seed=%s device=npu:%s\n' \
    "$fold" "$task" "$shot" "$seed" "$device"
  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  mkdir -p "$LOG_ROOT"
  python scripts/eval/run_registered_paper_downstream.py \
    --encoder-config "$ROOT/configs/paper_registered_20260716/${stem}.yaml" \
    --encoder-checkpoint "/data/xuannv_embedding/outputs/paper_registered_20260716/${stem}/best.pt" \
    --embedding-root "$embedding_root" \
    --embedding-registry "$EMBEDDING_REGISTRY" \
    --manifest "$MANIFEST" \
    --label-root "$LABEL_ROOT" \
    --spatial-split "$SPLIT" \
    --registry "$REGISTRY" \
    --fold "$fold" --month 202604 --task "$task" --shot "$shot" --shot-seed "$seed" \
    --shot-manifest "$schedule" --output-root "$output_root" --device "npu:${device}" \
    >"$log" 2>&1
}

if [[ "$DRY_RUN" == true ]]; then
  for index in "${!JOBS[@]}"; do
    run_job "${JOBS[$index]}" "$((index % 6))"
  done
  exit 0
fi

mkdir -p "$LOG_ROOT"
printf '%s\tstart\tjobs=%s\tgit=%s\n' "$(date -Iseconds)" "${#JOBS[@]}" \
  "$(git rev-parse HEAD)" > "$LOG_ROOT/status.tsv"

failed=0
for ((start = 0; start < ${#JOBS[@]}; start += 6)); do
  declare -a pids=()
  for ((device = 0; device < 6 && start + device < ${#JOBS[@]}; device++)); do
    run_job "${JOBS[$((start + device))]}" "$device" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  printf '%s\twave_complete\tstart=%s\tfailed=%s\n' "$(date -Iseconds)" "$start" "$failed" \
    >> "$LOG_ROOT/status.tsv"
  if (( failed != 0 )); then
    break
  fi
done

printf '%s\tend\tfailed=%s\n' "$(date -Iseconds)" "$failed" >> "$LOG_ROOT/status.tsv"
exit "$failed"
