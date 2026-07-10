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

RUN_ROOT="${RUN_ROOT:-/data/xuannv_embedding/experiments/p14_proto_boundary_eval_20260710}"
EMBED_ROOT="${RUN_ROOT}/embeddings"
BENCH_ROOT="${RUN_ROOT}/benchmarks"
DIAG_ROOT="${RUN_ROOT}/embedding_space"
LOG_ROOT="${RUN_ROOT}/logs"
MONTH="${MONTH:-202604}"
CONFIG="${CONFIG:-configs/v3_p14a_proto_boundary_haidian_202512_202605_20260710.yaml}"
DOWNSTREAM_CONFIG="${DOWNSTREAM_CONFIG:-downstreams/configs/v2_probe_binary_conv3x3_single_202604.yaml}"
TASKS=(construction haidian_building_osm haidian_road_osm haidian_water_osm)
EXPORT_SHARDS_PER_TAG="${EXPORT_SHARDS_PER_TAG:-2}"
EXPORT_DEVICES=(${EXPORT_DEVICES:-0 1 2 3 4 5})
AEF_ROOT="${AEF_ROOT:-/data/xuannv_embedding/embeddings/aef_official_2025_annual}"
AEF_MONTH="${AEF_MONTH:-202512}"

mkdir -p "${EMBED_ROOT}" "${BENCH_ROOT}" "${DIAG_ROOT}" "${LOG_ROOT}"

OUTPUT_DIR="/data/xuannv_embedding/outputs/v3_p14a_proto_boundary_haidian_202512_202605_20260710"
declare -A CHECKPOINTS=(
  [p14a_best]="${OUTPUT_DIR}/best.pt"
  [p14a_e800]="${OUTPUT_DIR}/epoch_800.pt"
)

TAGS=(${TAGS:-p14a_best p14a_e800})

run_export() {
  local tag="$1"
  local tag_index="$2"
  local marker="${RUN_ROOT}/${tag}_embedding_root.txt"
  if [[ -s "${marker}" ]]; then
    local existing
    existing="$(cat "${marker}")"
    if [[ -d "${existing}" ]]; then
      local count
      count="$(find "${existing}/haidian" -name "${MONTH}_embedding_map.pt" 2>/dev/null | wc -l)"
      if [[ "${count}" -ge 320 ]]; then
        echo "$(date '+%F %T') export exists ${tag}: ${existing}"
        return
      fi
      echo "$(date '+%F %T') export marker incomplete ${tag}: ${existing} (${count}/320), resuming"
    fi
  fi
  if [[ ! -f "${CHECKPOINTS[$tag]}" ]]; then
    echo "$(date '+%F %T') checkpoint missing: ${CHECKPOINTS[$tag]}" >&2
    exit 1
  fi
  local pids=()
  for shard_id in $(seq 0 "$((EXPORT_SHARDS_PER_TAG - 1))"); do
    local device_index=$((tag_index * EXPORT_SHARDS_PER_TAG + shard_id))
    local device="${EXPORT_DEVICES[$device_index]:-}"
    if [[ -z "${device}" ]]; then
      echo "Not enough EXPORT_DEVICES for ${tag} shard ${shard_id}" >&2
      exit 1
    fi
    echo "$(date '+%F %T') export start ${tag} shard ${shard_id}/${EXPORT_SHARDS_PER_TAG} on NPU ${device}"
    ASCEND_RT_VISIBLE_DEVICES="${device}" python downstreams/scripts/precompute_embeddings.py \
      --config "${CONFIG}" \
      --regions haidian \
      --output-root "${EMBED_ROOT}" \
      --checkpoint "${CHECKPOINTS[$tag]}" \
      --suffix "${tag}" \
      --months "${MONTH}" \
      --context-margin 16 \
      --center-crop-size 128 \
      --num-shards "${EXPORT_SHARDS_PER_TAG}" \
      --shard-id "${shard_id}" \
      --device npu:0 \
      > "${LOG_ROOT}/${tag}_export_shard${shard_id}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  local out_dir
  out_dir="$(find "${EMBED_ROOT}" -maxdepth 1 -type d -name "*_${tag}" | sort | tail -n 1)"
  if [[ -z "${out_dir}" || ! -d "${out_dir}" ]]; then
    echo "Failed to locate exported embedding root for ${tag}" >&2
    exit 1
  fi
  local count
  count="$(find "${out_dir}/haidian" -name "${MONTH}_embedding_map.pt" 2>/dev/null | wc -l)"
  if [[ "${count}" -lt 320 ]]; then
    echo "Export incomplete for ${tag}: ${count}/320 maps in ${out_dir}" >&2
    exit 1
  fi
  echo "${out_dir}" > "${marker}"
  echo "$(date '+%F %T') export done ${tag}: ${out_dir} (${count}/320)"
}

run_downstream() {
  local tag="$1"
  local marker="${RUN_ROOT}/${tag}_embedding_root.txt"
  if [[ ! -s "${marker}" ]]; then
    echo "$(date '+%F %T') no embedding marker for ${tag}" >&2
    exit 1
  fi
  local embedding_root
  embedding_root="$(cat "${marker}")"
  local out_dir="${BENCH_ROOT}/${tag}"
  if [[ -f "${out_dir}/POST_TRAINING_REPORT.md" ]]; then
    echo "$(date '+%F %T') downstream exists ${tag}: ${out_dir}"
    return
  fi
  echo "$(date '+%F %T') downstream start ${tag}"
  python scripts/scale/post_training_eval.py \
    --embedding-root "${embedding_root}" \
    --run-name "${tag}" \
    --benchmark-root "${out_dir}" \
    --config "${DOWNSTREAM_CONFIG}" \
    --tasks "${TASKS[@]}" \
    --npu 0,1,2,3 \
    --parallel-tasks \
    --samples-per-task 8 \
    --visualization-months "${MONTH}" "${MONTH}" \
    --skip-v1-comparison \
    > "${LOG_ROOT}/${tag}_downstream.log" 2>&1
  echo "$(date '+%F %T') downstream done ${tag}: ${out_dir}"
}

run_diagnostics() {
  local tag="$1"
  local marker="${RUN_ROOT}/${tag}_embedding_root.txt"
  local embedding_root
  embedding_root="$(cat "${marker}")"
  local out_dir="${DIAG_ROOT}/${tag}"
  if [[ -f "${out_dir}/embedding_space_summary.csv" ]]; then
    echo "$(date '+%F %T') diagnostics exists ${tag}: ${out_dir}"
    return
  fi
  mkdir -p "${out_dir}"
  echo "$(date '+%F %T') diagnostics start ${tag}"
  python scripts/report/diagnose_embedding_space.py \
    --xuannv-root "${embedding_root}" \
    --xuannv-month "${MONTH}" \
    --aef-root "${AEF_ROOT}" \
    --aef-month "${AEF_MONTH}" \
    --samples-per-class 20000 \
    --background-samples 20000 \
    --output-root "${out_dir}" \
    > "${LOG_ROOT}/${tag}_diagnostics.log" 2>&1
  echo "$(date '+%F %T') diagnostics done ${tag}: ${out_dir}"
}

tag_index=0
for tag in "${TAGS[@]}"; do
  run_export "${tag}" "${tag_index}"
  tag_index=$((tag_index + 1))
done

for tag in "${TAGS[@]}"; do
  run_downstream "${tag}"
  run_diagnostics "${tag}"
done

echo "$(date '+%F %T') all done"
