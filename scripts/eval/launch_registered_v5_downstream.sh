#!/usr/bin/env bash
# Launch only committed v5 OSM-assisted Conv3x3 cells in attempt-specific paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAMILY=""
PROTOCOL=""
DRY_RUN=false
MATRIX="$ROOT/configs/eval/rse_v5_osm_assisted_matrix.json"
RESULT_ROOT=/data/xuannv_embedding/experiments/paper_registered_v5_20260726/downstream
ENCODER_ROOT=/data/xuannv_embedding/outputs/paper_registered_v5_20260726
LABEL_ROOT=/data/xuannv_embedding/processed/haidian/labels
MANIFEST=/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
SPLIT="$ROOT/configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json"
EMBEDDING_REGISTRY="$ROOT/configs/eval/registered_embedding_exports_v5_20260726.json"
RESULT_REGISTRY="$RESULT_ROOT/registry/results_v5.jsonl"
SCHEDULE_ROOT=/data/xuannv_embedding/experiments/paper_registered_v5_20260726/shared_shot_schedules

usage() {
  echo "usage: $0 --protocol v5_osm_assisted --family <declared-v5-family> [--matrix path] [--dry-run]" >&2
}

while (( $# > 0 )); do
  case "$1" in
    --protocol) PROTOCOL=${2:-}; shift 2 ;;
    --family) FAMILY=${2:-}; shift 2 ;;
    --matrix) MATRIX=${2:-}; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$PROTOCOL" == "v5_osm_assisted" && -n "$FAMILY" ]] || { usage; exit 2; }
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/downstreams:${PYTHONPATH:-}"

python - "$FAMILY" "$MATRIX" <<'PY'
import sys
from pathlib import Path
from scripts.eval.run_registered_paper_downstream import load_registered_v5_matrix

family = sys.argv[1]
matrix = load_registered_v5_matrix(Path(sys.argv[2]))
if family not in matrix["families"]:
    raise SystemExit(2)
probe = matrix["probe"]
if probe["expected_job_count"] != 90:
    raise SystemExit("registered v5 matrix must contain exactly 90 probe jobs per family")
PY

declare -a JOBS=()
for task in building road water; do
  for shot in 5 10; do
    for fold in 0 1 2 3 4; do
      for seed in 42 43 44; do
        JOBS+=("$fold|$task|$shot|$seed")
      done
    done
  done
done
(( ${#JOBS[@]} == 90 )) || { echo "expected 90 v5 jobs" >&2; exit 3; }

print_job() {
  local spec=$1
  local device=$2
  IFS='|' read -r fold task shot seed <<< "$spec"
  printf 'PROBE protocol=v5_osm_assisted family=%s task=%s shot=%s fold=%s seed=%s device=npu:%s attempt=next\n' \
    "$FAMILY" "$task" "$shot" "$fold" "$seed" "$device"
}

if [[ "$DRY_RUN" == true ]]; then
  for index in "${!JOBS[@]}"; do
    print_job "${JOBS[$index]}" "$((index % 6))"
  done
  exit 0
fi

source /usr/local/Ascend/cann-9.0.0/set_env.sh

embedding_root_for_fold() {
  local fold=$1
  python - "$EMBEDDING_REGISTRY" "$FAMILY" "$fold" <<'PY'
import json
import sys
from pathlib import Path

from scripts.eval.run_registered_paper_downstream import validate_v5_embedding_registry

registry = Path(sys.argv[1])
family = sys.argv[2]
fold = int(sys.argv[3])
raw = json.loads(registry.read_text(encoding="utf-8"))
validate_v5_embedding_registry(registry, raw)
matches = [
    entry for entry in raw["exports"]
    if entry.get("family") == family and entry.get("encoder_fold") == fold
]
if len(matches) != 1 or not isinstance(matches[0].get("embedding_root"), str):
    raise SystemExit("registered v5 embedding root is missing or ambiguous")
print(matches[0]["embedding_root"])
PY
}

write_canonical_pointer() {
  local job_root=$1
  local attempt_root=$2
  python - "$job_root" "$attempt_root" "$RESULT_REGISTRY" <<'PY'
import json
import os
import sys
from pathlib import Path

from scripts.eval.run_registered_paper_downstream import (
    _canonical_sha256,
    sha256_file,
    verify_artifact_registry_binding,
)

job_root = Path(sys.argv[1])
attempt_root = Path(sys.argv[2])
registry = Path(sys.argv[3])
artifact_path = attempt_root / "artifact_manifest.json"
verify_artifact_registry_binding(artifact_path, registry)
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
pointer = job_root / "canonical_result.json"
payload = {
    "schema_version": 1,
    "attempt_dir": str(attempt_root.resolve()),
    "result_id": artifact["result_id"],
    "artifact_sha256": sha256_file(artifact_path),
    "registry_path": str(registry.resolve()),
}
payload["sha256"] = _canonical_sha256(payload)
try:
    descriptor = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
except FileExistsError as exc:
    raise SystemExit("canonical result already exists for this job") from exc
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

run_job() {
  local spec=$1
  local device=$2
  IFS='|' read -r fold task shot seed <<< "$spec"
  local job_root="$RESULT_ROOT/probes/$FAMILY/fold${fold}/${task}/shot_${shot}/seed_${seed}"
  local attempt=1
  while [[ -e "$job_root/attempt_${attempt}" ]]; do ((attempt += 1)); done
  local attempt_root="$job_root/attempt_${attempt}"
  local log="$job_root/logs/attempt_${attempt}.log"
  local config="$ROOT/configs/paper_registered_v5_20260726/paper_registered_v5_${FAMILY}_fold${fold}_20260726.yaml"
  local checkpoint="$ENCODER_ROOT/paper_registered_v5_${FAMILY}_fold${fold}_20260726/best.pt"
  local schedule="$SCHEDULE_ROOT/${task}_fold${fold}_seed${seed}.json"
  local embedding_root
  embedding_root="$(embedding_root_for_fold "$fold")"
  mkdir -p "$job_root/logs"
  print_job "$spec" "$device"
  python "$ROOT/scripts/eval/run_registered_paper_downstream.py" \
    --protocol v5_osm_assisted --encoder-config "$config" --encoder-checkpoint "$checkpoint" \
    --embedding-root "$embedding_root" --embedding-registry "$EMBEDDING_REGISTRY" \
    --manifest "$MANIFEST" --label-root "$LABEL_ROOT" --spatial-split "$SPLIT" \
    --output-root "$attempt_root" --registry "$RESULT_REGISTRY" --task "$task" --fold "$fold" \
    --month 202604 --shot "$shot" --shot-seed "$seed" --shot-manifest "$schedule" \
    --device "npu:${device}" >"$log" 2>&1
  write_canonical_pointer "$job_root" "$attempt_root"
}

mkdir -p "$RESULT_ROOT/logs"
printf '%s\tstart\tjobs=90\n' "$(date -Iseconds)" > "$RESULT_ROOT/logs/${FAMILY}_status.tsv"
failed=0
for ((start = 0; start < ${#JOBS[@]}; start += 6)); do
  declare -a pids=()
  for ((device = 0; device < 6 && start + device < ${#JOBS[@]}; device++)); do
    run_job "${JOBS[$((start + device))]}" "$device" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  printf '%s\twave_complete\tstart=%s\tfailed=%s\n' "$(date -Iseconds)" "$start" "$failed" \
    >> "$RESULT_ROOT/logs/${FAMILY}_status.tsv"
  (( failed == 0 )) || break
done
printf '%s\tend\tfailed=%s\n' "$(date -Iseconds)" "$failed" >> "$RESULT_ROOT/logs/${FAMILY}_status.tsv"
exit "$failed"
