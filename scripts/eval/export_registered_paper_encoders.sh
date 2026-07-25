#!/usr/bin/env bash
# Export one registered five-fold encoder family with six sealed NPU shards per fold.
set -euo pipefail

ROOT=/root/workspace/xuannv
EVAL_ROOT=/data/xuannv_embedding/experiments/paper_registered_eval_20260725
CONFIG_ROOT="$ROOT/configs/paper_registered_20260716"
OUTPUT_ROOT=/data/xuannv_embedding/outputs/paper_registered_20260716
EMBED_ROOT="$EVAL_ROOT/embeddings"
LOG_ROOT="$EVAL_ROOT/logs/registered_encoder_exports"
MANIFEST=/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
MONTH=202604
FAMILY=""
DRY_RUN=false

usage() {
  echo "usage: $0 --family <family> [--dry-run]" >&2
}

while (( $# > 0 )); do
  case "$1" in
    --family)
      FAMILY=${2:-}
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "$FAMILY" in
  no_osm_150|coarse_osm_only_150|probe_nohardneg_150|no_highres_path_150|no_masking_150)
    ;;
  *)
    usage
    exit 2
    ;;
esac

cd "$ROOT"
if [[ "$DRY_RUN" == false ]]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/downstreams:${PYTHONPATH:-}"

finalize_export() {
  local export_root=$1
  local commands_json=$2
  /data/wwj_torch21/conda/envs/torch26/bin/python - "$export_root" "$commands_json" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

from scripts.eval.run_registered_paper_downstream import (
    canonicalize_embedding_export,
    seal_embedding_file_index,
)

export_root = Path(sys.argv[1])
commands = {int(key): value for key, value in json.loads(sys.argv[2]).items()}
manifest = Path(sys.argv[3])
patch_ids = {entry["patch_id"] for entry in json.loads(manifest.read_text(encoding="utf-8"))}
canonicalize_embedding_export(
    export_root,
    "haidian",
    "202604",
    expected_patch_ids=patch_ids,
    shard_commands=commands,
)
index = seal_embedding_file_index(export_root, "haidian", "202604")
meta = json.loads((export_root / "meta.json").read_text(encoding="utf-8"))
print(json.dumps({
    "embedding_root": str(export_root),
    "index_sha256": index["index_sha256"],
    "canonical_provenance_sha256": meta["canonical_export_provenance_sha256"],
}, sort_keys=True))
PY
}

write_export_meta() {
  local export_root=$1
  local config=$2
  local checkpoint=$3
  /data/wwj_torch21/conda/envs/torch26/bin/python - "$export_root" "$config" "$checkpoint" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

from downstreams.inference import write_meta_json

output_root = Path(sys.argv[1])
config = Path(sys.argv[2])
checkpoint = Path(sys.argv[3])
manifest = Path(sys.argv[4])
write_meta_json(
    output_root,
    checkpoint,
    config,
    "registered_sharded_export_coordinator",
    manifest_path=manifest,
)
PY
}

for fold in 0 1 2 3 4; do
  stem="paper_registered_${FAMILY}_fold${fold}_20260716"
  config="$CONFIG_ROOT/${stem}.yaml"
  checkpoint="$OUTPUT_ROOT/${stem}/best.pt"
  suffix="${FAMILY}_fold${fold}"

  if [[ ! -f "$config" || ! -f "$checkpoint" ]]; then
    echo "missing config or best checkpoint for ${stem}" >&2
    exit 3
  fi

  declare -a commands=()
  for shard in 0 1 2 3 4 5; do
    printf 'EXPORT\tfamily=%s fold=%s shard=%s device=npu:%s\n' \
      "$FAMILY" "$fold" "$shard" "$shard"
    printf -v command 'ASCEND_RT_VISIBLE_DEVICES=%q ' "$shard"
    printf -v escaped '%q ' \
      /data/wwj_torch21/conda/envs/torch26/bin/python \
      downstreams/scripts/precompute_embeddings.py \
      --config "$config" --checkpoint "$checkpoint" --regions haidian \
      --manifest-path "$MANIFEST" --output-root "$EMBED_ROOT" --suffix "$suffix" \
      --months "$MONTH" --center-crop-size 128 --num-shards 6 --shard-id "$shard" --device npu:0 \
      --skip-meta
    commands+=("${command}${escaped% }")
  done
  if [[ "$DRY_RUN" == true ]]; then
    continue
  fi

  shopt -s nullglob
  existing=("$EMBED_ROOT"/*_"$suffix")
  shopt -u nullglob
  if (( ${#existing[@]} != 0 )); then
    printf 'refusing to mix with existing export(s): %s\n' "${existing[*]}" >&2
    exit 4
  fi

  mkdir -p "$LOG_ROOT"
  declare -a pids=()
  for shard in 0 1 2 3 4 5; do
    eval "${commands[$shard]}" >"$LOG_ROOT/${FAMILY}_fold${fold}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "one or more export shards failed for ${stem}" >&2
    exit 5
  fi

  mapfile -t produced < <(find "$EMBED_ROOT" -maxdepth 1 -type d -name "*_${suffix}" -printf '%p\n' | sort)
  if (( ${#produced[@]} != 1 )); then
    printf 'expected exactly one export root for %s, found: %s\n' "$stem" "${produced[*]:-none}" >&2
    exit 6
  fi
  export_root="${produced[0]}"
  write_export_meta "$export_root" "$config" "$checkpoint"
  commands_json="$(/data/wwj_torch21/conda/envs/torch26/bin/python - "${commands[@]}" <<'PY'
import json
import sys
print(json.dumps({str(i): item for i, item in enumerate(sys.argv[1:])}, sort_keys=True))
PY
  )"
  finalize_export "$export_root" "$commands_json" | tee "$LOG_ROOT/${FAMILY}_fold${fold}_finalize.json"
done
