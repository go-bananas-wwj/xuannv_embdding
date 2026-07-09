#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ -f "${CANN_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${CANN_ENV}"
fi

export PYTHONPATH="$PWD/src:$PWD:$PWD/downstreams:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/production/haidian_embedding_v1.yaml}"
CHECKPOINT="${CHECKPOINT:-/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/xuannv_embedding/embeddings/production}"
RUN_ROOT="${RUN_ROOT:-/data/xuannv_embedding/production/haidian_embedding_v1_p10c_epoch800}"
SUFFIX="${SUFFIX:-haidian_embedding_v1_p10c_epoch800}"
MONTHS=(${MONTHS:-202512 202601 202602 202603 202604 202605})
DEVICES=(${DEVICES:-0 1 2 3 4 5})
NUM_SHARDS="${NUM_SHARDS:-${#DEVICES[@]}}"
CONTEXT_MARGIN="${CONTEXT_MARGIN:-16}"
CENTER_CROP_SIZE="${CENTER_CROP_SIZE:-128}"

mkdir -p "${RUN_ROOT}/logs" "${OUTPUT_ROOT}"

if [[ "${NUM_SHARDS}" -ne "${#DEVICES[@]}" ]]; then
  echo "NUM_SHARDS (${NUM_SHARDS}) must match DEVICES count (${#DEVICES[@]}) for this launcher." >&2
  exit 1
fi

pids=()
for shard_id in $(seq 0 "$((NUM_SHARDS - 1))"); do
  device="${DEVICES[$shard_id]}"
  echo "$(date '+%F %T') export shard ${shard_id}/${NUM_SHARDS} on NPU ${device}"
  ASCEND_RT_VISIBLE_DEVICES="${device}" python scripts/production/export_haidian_embedding.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --suffix "${SUFFIX}" \
    --months "${MONTHS[@]}" \
    --device npu:0 \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --context-margin "${CONTEXT_MARGIN}" \
    --center-crop-size "${CENTER_CROP_SIZE}" \
    > "${RUN_ROOT}/logs/export_shard${shard_id}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

out_dir="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "*_${SUFFIX}" | sort | tail -n 1)"
if [[ -z "${out_dir}" || ! -d "${out_dir}" ]]; then
  echo "Failed to locate exported embedding root under ${OUTPUT_ROOT}" >&2
  exit 1
fi

for month in "${MONTHS[@]}"; do
  count="$(find "${out_dir}/haidian" -name "${month}_embedding_map.pt" 2>/dev/null | wc -l)"
  if [[ "${count}" -lt 320 ]]; then
    echo "Export incomplete for ${month}: ${count}/320 maps in ${out_dir}" >&2
    exit 1
  fi
done

echo "${out_dir}" > "${RUN_ROOT}/embedding_root.txt"
echo "$(date '+%F %T') Haidian V1 embedding export done: ${out_dir}"
