#!/usr/bin/env bash
# Export one declared v5 encoder family through six independently sealed shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAMILY=""
PROTOCOL=""
DRY_RUN=false
MATRIX="$ROOT/configs/eval/rse_v5_osm_assisted_matrix.json"
MONTH=202604
MANIFEST=/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
OUTPUT_ROOT=/data/xuannv_embedding/outputs/paper_registered_v5_20260726
EMBED_ROOT=/data/xuannv_embedding/experiments/paper_registered_v5_20260726/embeddings
REGISTRY="$ROOT/configs/eval/registered_embedding_exports_v5_20260726.json"
LOG_ROOT=/data/xuannv_embedding/experiments/paper_registered_v5_20260726/logs/exports

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
PY

for fold in 0 1 2 3 4; do
  config="$ROOT/configs/paper_registered_v5_20260726/paper_registered_v5_${FAMILY}_fold${fold}_20260726.yaml"
  checkpoint="$OUTPUT_ROOT/paper_registered_v5_${FAMILY}_fold${fold}_20260726/best.pt"
  suffix="${FAMILY}_fold${fold}"
  declare -a commands=()
  for shard in 0 1 2 3 4 5; do
    printf 'EXPORT protocol=v5_osm_assisted family=%s fold=%s shard=%s device=npu:%s\n' \
      "$FAMILY" "$fold" "$shard" "$shard"
    commands+=("precompute_embeddings.py family=${FAMILY} fold=${fold} shard=${shard} device=npu:${shard}")
  done
  printf 'EXPORT protocol=v5_osm_assisted family=%s fold=%s phase=finalize shards=6 registry=%s\n' \
    "$FAMILY" "$fold" "$REGISTRY"
  [[ "$DRY_RUN" == true ]] && continue
  [[ -f "$config" && -f "$checkpoint" ]] || {
    echo "missing registered v5 config or best checkpoint for ${FAMILY}/fold${fold}" >&2
    exit 3
  }
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
  mkdir -p "$LOG_ROOT"
  declare -a pids=()
  for shard in 0 1 2 3 4 5; do
    ASCEND_RT_VISIBLE_DEVICES="$shard" python "$ROOT/downstreams/scripts/precompute_embeddings.py" \
      --config "$config" --checkpoint "$checkpoint" --regions haidian --manifest-path "$MANIFEST" \
      --output-root "$EMBED_ROOT" --suffix "$suffix" --months "$MONTH" --center-crop-size 128 \
      --num-shards 6 --shard-id "$shard" --device npu:0 --skip-meta \
      >"$LOG_ROOT/${suffix}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  (( failed == 0 )) || {
    echo "one or more v5 export shards failed for ${suffix}" >&2
    exit 5
  }
  mapfile -t produced < <(find "$EMBED_ROOT" -maxdepth 1 -type d -name "*_${suffix}" -printf '%p\n' | sort)
  (( ${#produced[@]} == 1 )) || {
    echo "expected one dated output root for ${suffix}, found ${#produced[@]}" >&2
    exit 6
  }
  export_root="${produced[0]}"
  commands_json="$(python - "${commands[@]}" <<'PY'
import hashlib
import json
import sys
print(json.dumps({str(index): command for index, command in enumerate(sys.argv[1:])}, sort_keys=True))
PY
  )"
  python - "$export_root" "$config" "$checkpoint" "$MANIFEST" "$FAMILY" "$fold" "$MONTH" \
    "$commands_json" "$REGISTRY" <<'PY'
import json
import sys
from pathlib import Path

from downstreams.inference import write_meta_json
from scripts.eval.run_registered_paper_downstream import (
    _canonical_sha256,
    _write_json,
    build_v5_embedding_registry_entry,
    canonicalize_embedding_export,
    resolve_registered_protocol,
    seal_embedding_file_index,
    validate_v5_embedding_registry,
)

export_root, config, checkpoint, manifest, family, fold, month, commands, registry = sys.argv[1:]
export_root = Path(export_root)
config = Path(config)
checkpoint = Path(checkpoint)
manifest = Path(manifest)
registry = Path(registry)
write_meta_json(export_root, checkpoint, config, "registered_v5_sharded_export", manifest_path=manifest)
meta_path = export_root / "meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
descriptor = resolve_registered_protocol("v5_osm_assisted")
meta.update(
    {
        "protocol_id": "v5_osm_assisted",
        "spatial_split": str(descriptor["split_path"]),
        "spatial_split_sha256": descriptor["split_sha256"],
        "manifest_sha256": descriptor["manifest_sha256"],
        "statistics_registry_sha256": descriptor["statistics_registry_sha256"],
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
)
_write_json(meta_path, meta)
patch_ids = {str(record["patch_id"]) for record in json.loads(manifest.read_text(encoding="utf-8"))}
canonicalize_embedding_export(
    export_root,
    "haidian",
    month,
    expected_patch_ids=patch_ids,
    shard_commands={int(key): value for key, value in json.loads(commands).items()},
)
seal_embedding_file_index(export_root, "haidian", month)
entry = build_v5_embedding_registry_entry(
    export_root, family=family, fold=int(fold), region="haidian", month=month
)
validate_v5_embedding_registry(registry, {
    "spatial_split": str(descriptor["split_path"]),
    "spatial_split_sha256": descriptor["split_sha256"],
    "manifest_sha256": descriptor["manifest_sha256"],
    "statistics_registry_sha256": descriptor["statistics_registry_sha256"],
})
pending = {
    "schema_version": 1,
    "registry_path": str(registry.resolve()),
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "entry": {**entry, "embedding_root": str(export_root.resolve())},
}
pending["entry_sha256"] = _canonical_sha256(pending["entry"])
_write_json(export_root / "pending_registry_entry.json", pending)
print(json.dumps({"pending_registry_entry": str(export_root / "pending_registry_entry.json")}, sort_keys=True))
PY
done
