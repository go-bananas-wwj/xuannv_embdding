#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi
source "${CANN_ENV}"

export PYTHONPATH="$PWD/src:$PWD:$PWD/downstreams:${PYTHONPATH:-}"

RUN_ROOT="${RUN_ROOT:-/data/xuannv_embedding/experiments/p12_context_eval_20260707}"
EMBED_ROOT="${RUN_ROOT}/embeddings"
BENCH_ROOT="${RUN_ROOT}/benchmarks"
LOG_ROOT="${RUN_ROOT}/logs"
MONTH="${MONTH:-202604}"

mkdir -p "${EMBED_ROOT}" "${BENCH_ROOT}" "${LOG_ROOT}"

declare -A CONFIGS=(
  [p12a_p10c_context]="configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml"
  [p12b_best]="configs/v2_p12b_context160_center128_haidian_202512_202605_20260707.yaml"
  [p12b_e400]="configs/v2_p12b_context160_center128_haidian_202512_202605_20260707.yaml"
  [p12c_best]="configs/v2_p12c_context160_semantic_rich_haidian_202512_202605_20260707.yaml"
  [p12c_e400]="configs/v2_p12c_context160_semantic_rich_haidian_202512_202605_20260707.yaml"
)

declare -A CHECKPOINTS=(
  [p12a_p10c_context]="/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt"
  [p12b_best]="/data/xuannv_embedding/outputs/v2_p12b_context160_center128_haidian_202512_202605_20260707/best.pt"
  [p12b_e400]="/data/xuannv_embedding/outputs/v2_p12b_context160_center128_haidian_202512_202605_20260707/epoch_400.pt"
  [p12c_best]="/data/xuannv_embedding/outputs/v2_p12c_context160_semantic_rich_haidian_202512_202605_20260707/best.pt"
  [p12c_e400]="/data/xuannv_embedding/outputs/v2_p12c_context160_semantic_rich_haidian_202512_202605_20260707/epoch_400.pt"
)

TAGS=(p12a_p10c_context p12b_best p12b_e400 p12c_best p12c_e400)

run_export() {
  local tag="$1"
  local device="$2"
  local marker="${RUN_ROOT}/${tag}_embedding_root.txt"
  if [[ -s "${marker}" ]]; then
    local existing
    existing="$(cat "${marker}")"
    if [[ -d "${existing}" ]]; then
      echo "$(date '+%F %T') export exists ${tag}: ${existing}"
      return
    fi
  fi
  if [[ ! -f "${CHECKPOINTS[$tag]}" ]]; then
    echo "$(date '+%F %T') checkpoint missing; skip export ${tag}: ${CHECKPOINTS[$tag]}"
    return
  fi
  echo "$(date '+%F %T') export start ${tag} on ${device}"
  ASCEND_RT_VISIBLE_DEVICES="${device}" python downstreams/scripts/precompute_embeddings.py \
    --config "${CONFIGS[$tag]}" \
    --regions haidian \
    --output-root "${EMBED_ROOT}" \
    --checkpoint "${CHECKPOINTS[$tag]}" \
    --suffix "${tag}" \
    --months "${MONTH}" \
    --context-margin 16 \
    --center-crop-size 128 \
    --device npu:0 \
    > "${LOG_ROOT}/${tag}_export.log" 2>&1
  local out_dir
  out_dir="$(find "${EMBED_ROOT}" -maxdepth 1 -type d -name "*_${tag}" | sort | tail -n 1)"
  if [[ -z "${out_dir}" || ! -d "${out_dir}" ]]; then
    echo "Failed to locate exported embedding root for ${tag}" >&2
    exit 1
  fi
  echo "${out_dir}" > "${marker}"
  echo "$(date '+%F %T') export done ${tag}: ${out_dir}"
}

run_downstream() {
  local tag="$1"
  local marker="${RUN_ROOT}/${tag}_embedding_root.txt"
  if [[ ! -s "${marker}" ]]; then
    echo "$(date '+%F %T') no embedding for ${tag}; skip downstream"
    return
  fi
  local embedding_root
  embedding_root="$(cat "${marker}")"
  local out_dir="${BENCH_ROOT}/${tag}"
  if [[ -f "${out_dir}/POST_TRAINING_REPORT.md" ]]; then
    echo "$(date '+%F %T') downstream exists ${tag}"
    return
  fi
  echo "$(date '+%F %T') downstream start ${tag}"
  python scripts/scale/post_training_eval.py \
    --embedding-root "${embedding_root}" \
    --run-name "${tag}" \
    --benchmark-root "${out_dir}" \
    --config downstreams/configs/v2_probe_mlp_single_202604.yaml \
    --tasks construction haidian_building_osm haidian_road_osm haidian_water_osm \
    --npu 0,1,2,3 \
    --parallel-tasks \
    --samples-per-task 8 \
    --visualization-months "${MONTH}" "${MONTH}" \
    --skip-v1-comparison \
    > "${LOG_ROOT}/${tag}_downstream.log" 2>&1
  echo "$(date '+%F %T') downstream done ${tag}: ${out_dir}"
}

export_pids=()
for idx in "${!TAGS[@]}"; do
  run_export "${TAGS[$idx]}" "$((idx % 6))" &
  export_pids+=("$!")
done
for pid in "${export_pids[@]}"; do
  wait "${pid}"
done

for tag in "${TAGS[@]}"; do
  run_downstream "${tag}"
done

echo "$(date '+%F %T') P12 context eval suite done: ${RUN_ROOT}"
