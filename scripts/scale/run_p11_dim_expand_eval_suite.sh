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

RUN_ROOT="${RUN_ROOT:-/data/xuannv_embedding/experiments/p11_dim_expand_eval_20260707}"
EMBED_ROOT="${RUN_ROOT}/embeddings"
DIAG_ROOT="${RUN_ROOT}/embedding_space"
BENCH_ROOT="${RUN_ROOT}/benchmarks"
LOG_ROOT="${RUN_ROOT}/logs"
MONTH="${MONTH:-202604}"
AEF_ROOT="${AEF_ROOT:-/data/xuannv_embedding/embeddings/aef_official_2025_annual}"
AEF_MONTH="${AEF_MONTH:-202512}"

mkdir -p "${EMBED_ROOT}" "${DIAG_ROOT}" "${BENCH_ROOT}" "${LOG_ROOT}"

declare -A CONFIGS=(
  [p11a_best]="configs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11a_e600]="configs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11b_best]="configs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11b_e600]="configs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11c_best]="configs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706.yaml"
  [p11c_e600]="configs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706.yaml"
)

declare -A CHECKPOINTS=(
  [p11a_best]="/data/xuannv_embedding/outputs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706/best.pt"
  [p11a_e600]="/data/xuannv_embedding/outputs/v2_p11a_uniformity400_haidian_202512_202605_dim_expand_20260706/epoch_600.pt"
  [p11b_best]="/data/xuannv_embedding/outputs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706/best.pt"
  [p11b_e600]="/data/xuannv_embedding/outputs/v2_p11b_uniformity_vcreg400_haidian_202512_202605_dim_expand_20260706/epoch_600.pt"
  [p11c_best]="/data/xuannv_embedding/outputs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706/best.pt"
  [p11c_e600]="/data/xuannv_embedding/outputs/v2_p11c_uniformity_vcreg_patchdisc400_haidian_202512_202605_dim_expand_20260706/epoch_600.pt"
)

TAGS=(p11a_best p11a_e600 p11b_best p11b_e600 p11c_best p11c_e600)

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
  echo "$(date '+%F %T') export start ${tag} on ${device}"
  ASCEND_RT_VISIBLE_DEVICES="${device}" python downstreams/scripts/precompute_embeddings.py \
    --config "${CONFIGS[$tag]}" \
    --regions haidian \
    --output-root "${EMBED_ROOT}" \
    --checkpoint "${CHECKPOINTS[$tag]}" \
    --suffix "${tag}" \
    --months "${MONTH}" \
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

run_diagnostics() {
  local tag="$1"
  local embedding_root
  embedding_root="$(cat "${RUN_ROOT}/${tag}_embedding_root.txt")"
  local out_dir="${DIAG_ROOT}/${tag}"
  if [[ -f "${out_dir}/embedding_space_summary.csv" ]]; then
    echo "$(date '+%F %T') diagnostics exists ${tag}"
    return
  fi
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

run_downstream() {
  local tag="$1"
  local embedding_root
  embedding_root="$(cat "${RUN_ROOT}/${tag}_embedding_root.txt")"
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
    --tasks construction haidian_building_osm haidian_road_osm haidian_water_osm \
    --npu 0,1,2,3 \
    --parallel-tasks \
    --samples-per-task 8 \
    --visualization-months "${MONTH}" "${MONTH}" \
    --skip-v1-comparison \
    > "${LOG_ROOT}/${tag}_downstream.log" 2>&1
  echo "$(date '+%F %T') downstream done ${tag}: ${out_dir}"
}

echo "$(date '+%F %T') P11 eval suite start: ${RUN_ROOT}"

for idx in "${!TAGS[@]}"; do
  tag="${TAGS[$idx]}"
  device="$((idx % 6))"
  run_export "${tag}" "${device}"
done

for tag in "${TAGS[@]}"; do
  run_diagnostics "${tag}"
done

for tag in "${TAGS[@]}"; do
  run_downstream "${tag}"
done

echo "$(date '+%F %T') P11 eval suite done: ${RUN_ROOT}"
