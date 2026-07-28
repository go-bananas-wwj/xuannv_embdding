#!/usr/bin/env bash
# Export one declared v5 encoder family through six independently sealed shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAMILY=""
PROTOCOL=""
DRY_RUN=false
REGISTER_PENDING=false
MATRIX="$ROOT/configs/eval/rse_v5_osm_assisted_matrix.json"
MONTH=202604
MANIFEST=/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
OUTPUT_ROOT=/data/xuannv_embedding/outputs/paper_registered_v5_20260726
EMBED_ROOT="${V5_EMBED_ROOT:-/data/xuannv_embedding/experiments/paper_registered_v5_20260726/embeddings}"
REGISTRY="$ROOT/configs/eval/registered_embedding_exports_v5_20260726.json"
LOG_ROOT="${V5_LOG_ROOT:-/data/xuannv_embedding/experiments/paper_registered_v5_20260726/logs/exports}"
LANE_LOCK_ROOT=/data/xuannv_embedding/locks/registered_v5
if [[ -n "${PYTEST_CURRENT_TEST:-}" && "$ROOT" == /tmp/pytest-of-* ]]; then
  LANE_LOCK_ROOT="$ROOT/.pytest_registered_v5_locks"
fi
if [[ -n "${V5_EMBED_ROOT:-}${V5_LOG_ROOT:-}" ]] && \
  [[ -z "${PYTEST_CURRENT_TEST:-}" || "$ROOT" != /tmp/pytest-of-* ]]; then
  echo "registered V5 export path overrides are restricted to pytest fixtures" >&2
  exit 2
fi

usage() {
  echo "usage: $0 --protocol v5_osm_assisted --family <declared-v5-family> [--matrix path] [--register-pending] [--dry-run]" >&2
}

while (( $# > 0 )); do
  case "$1" in
    --protocol) PROTOCOL=${2:-}; shift 2 ;;
    --family) FAMILY=${2:-}; shift 2 ;;
    --matrix) MATRIX=${2:-}; shift 2 ;;
    --register-pending) REGISTER_PENDING=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$PROTOCOL" == "v5_osm_assisted" && -n "$FAMILY" ]] || { usage; exit 2; }
[[ "$FAMILY" == "full_150" ]] || {
  echo "registered V5 exporter currently supports only full_150" >&2
  exit 2
}
cd "$ROOT"
assert_script_at_head() {
  local relative=$1
  git ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 || {
    echo "v5 script is not Git tracked: $relative" >&2
    exit 2
  }
  cmp -s "$ROOT/$relative" <(git show "HEAD:$relative") || {
    echo "v5 script differs from Git HEAD: $relative" >&2
    exit 2
  }
}
assert_script_at_head "scripts/eval/export_registered_v5_paper_encoders.sh"
assert_runtime_tree_at_head() {
  local untracked
  git diff --quiet HEAD -- \
    src/xuannv_embedding \
    downstreams/downstreams \
    downstreams/scripts \
    scripts/eval || {
    echo "v5 runtime source tree differs from Git HEAD" >&2
    exit 2
  }
  untracked="$(git ls-files --others --exclude-standard -- \
    src/xuannv_embedding \
    downstreams/downstreams \
    downstreams/scripts \
    scripts/eval | awk '/\.(py|sh)$/')"
  [[ -z "$untracked" ]] || {
    echo "v5 runtime source tree differs from Git HEAD" >&2
    exit 2
  }
}
assert_runtime_tree_at_head
for runtime_source in \
  "downstreams/scripts/precompute_embeddings.py" \
  "downstreams/scripts/export_paths.py" \
  "downstreams/downstreams/inference.py" \
  "scripts/eval/run_registered_paper_downstream.py" \
  "scripts/eval/registered_v5_encoder_checkpoint.py" \
  "scripts/eval/registered_v5_matrix.py"; do
  git ls-files --error-unmatch -- "$runtime_source" >/dev/null 2>&1 || {
    echo "v5 runtime source is not Git tracked: $runtime_source" >&2
    exit 2
  }
  cmp -s "$ROOT/$runtime_source" <(git show "HEAD:$runtime_source") || {
    echo "v5 runtime source differs from Git HEAD: $runtime_source" >&2
    exit 2
  }
done
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/downstreams:${PYTHONPATH:-}"

assert_no_active_registered_v5_training() {
  local fold
  for fold in 0 1 2 3 4; do
    if pgrep -f "$ROOT/scripts/train/train.py.*paper_registered_v5_full_150_fold${fold}_" >/dev/null; then
      echo "registered V5 encoder training is active; refuse export until all folds stop" >&2
      exit 5
    fi
  done
}

acquire_registered_v5_export_leases() {
  local lane
  mkdir -p "$LANE_LOCK_ROOT"
  for lane in 0 1 2; do
    local fd
    exec {fd}>"$LANE_LOCK_ROOT/lane_${lane}.lock"
    if ! flock -n "$fd"; then
      echo "registered V5 encoder lane $lane is active; refuse export until all folds stop" >&2
      exit 5
    fi
  done
}

preflight_registered_v5_folds() {
  local fold config checkpoint
  for fold in 0 1 2 3 4; do
    config="$ROOT/configs/paper_registered_v5_20260726/paper_registered_v5_${FAMILY}_fold${fold}_20260726.yaml"
    checkpoint="$(python "$ROOT/scripts/eval/registered_v5_encoder_checkpoint.py" \
      --output-root "$OUTPUT_ROOT" --family "$FAMILY" --fold "$fold")"
    python - "$checkpoint" "$config" <<'PY'
import sys
from pathlib import Path

from scripts.eval.registered_v5_encoder_checkpoint import (
    validate_registered_v5_checkpoint_config_binding,
)

validate_registered_v5_checkpoint_config_binding(
    Path(sys.argv[1]), expected_config_path=Path(sys.argv[2])
)
PY
    [[ -f "$config" && -f "$checkpoint" ]] || {
      echo "missing registered v5 config or best checkpoint for ${FAMILY}/fold${fold}" >&2
      exit 3
    }
  done
}

if [[ "$DRY_RUN" == false ]]; then
  assert_no_active_registered_v5_training
  acquire_registered_v5_export_leases
  # Catch a non-cooperative legacy launch that raced the first process check.
  assert_no_active_registered_v5_training
  preflight_registered_v5_folds
fi

python - "$FAMILY" "$MATRIX" <<'PY'
import sys
from pathlib import Path
from scripts.eval.registered_v5_matrix import load_registered_v5_matrix

family = sys.argv[1]
matrix = load_registered_v5_matrix(Path(sys.argv[2]))
if family not in matrix["families"]:
    raise SystemExit(2)
PY

RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
declare -a PENDING_ENTRIES=()
for fold in 0 1 2 3 4; do
  config="$ROOT/configs/paper_registered_v5_20260726/paper_registered_v5_${FAMILY}_fold${fold}_20260726.yaml"
  suffix="${FAMILY}_fold${fold}_${RUN_ID}"
  declare -a commands=()
  for shard in 0 1 2 3 4 5; do
    printf 'EXPORT protocol=v5_osm_assisted family=%s fold=%s shard=%s device=npu:%s\n' \
      "$FAMILY" "$fold" "$shard" "$shard"
    commands+=("precompute_embeddings.py family=${FAMILY} fold=${fold} shard=${shard} device=npu:${shard}")
  done
  printf 'EXPORT protocol=v5_osm_assisted family=%s fold=%s phase=finalize shards=6 registry=%s\n' \
    "$FAMILY" "$fold" "$REGISTRY"
  if [[ "$DRY_RUN" == true ]]; then
    continue
  fi
  checkpoint="$(python "$ROOT/scripts/eval/registered_v5_encoder_checkpoint.py" \
    --output-root "$OUTPUT_ROOT" --family "$FAMILY" --fold "$fold")"
  config_sha256="$(python - "$checkpoint" "$config" <<'PY'
import sys
import hashlib
from pathlib import Path

from scripts.eval.registered_v5_encoder_checkpoint import (
    validate_registered_v5_checkpoint_config_binding,
)

validate_registered_v5_checkpoint_config_binding(
    Path(sys.argv[1]), expected_config_path=Path(sys.argv[2])
)
print(hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest())
PY
  )"
  [[ -f "$config" && -f "$checkpoint" && -n "$config_sha256" ]] || {
    echo "missing registered v5 config or best checkpoint for ${FAMILY}/fold${fold}" >&2
    exit 3
  }
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
  mkdir -p "$LOG_ROOT"
  # The registered export root is created on first use; individual runs remain exclusive below.
  mkdir -p "$EMBED_ROOT"
  export_name="$(date -u +%Y%m%d)_$(basename "${config%.yaml}")_best_${suffix}"
  export_root="$EMBED_ROOT/$export_name"
  mkdir "$export_root" || {
    echo "refusing to reuse a registered v5 export root: $export_root" >&2
    exit 4
  }
  # Every shard reads this verified snapshot, not a mutable working-tree config path.
  config_snapshot="$export_root/training_config.yaml"
  cp "$config" "$config_snapshot"
  snapshot_sha256="$(sha256sum "$config_snapshot" | awk '{print $1}')"
  [[ "$snapshot_sha256" == "$config_sha256" ]] || {
    echo "registered v5 config changed while creating export snapshot for ${suffix}" >&2
    exit 3
  }
  chmod 0444 "$config_snapshot"
  declare -a pids=()
  for shard in 0 1 2 3 4 5; do
    ASCEND_RT_VISIBLE_DEVICES="$shard" python "$ROOT/downstreams/scripts/precompute_embeddings.py" \
      --config "$config_snapshot" --checkpoint "$checkpoint" --regions haidian --manifest-path "$MANIFEST" \
      --output-root "$EMBED_ROOT" --suffix "$suffix" --months "$MONTH" --center-crop-size 128 \
      --export-name "$export_name" \
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
    exit 6
  }
commands_json="$(python - "${commands[@]}" <<'PY'
import hashlib
import json
import sys
print(json.dumps({str(index): command for index, command in enumerate(sys.argv[1:])}, sort_keys=True))
PY
  )"
  python - "$export_root" "$config" "$config_snapshot" "$config_sha256" "$checkpoint" "$MANIFEST" "$FAMILY" "$fold" "$MONTH" \
    "$commands_json" "$REGISTRY" <<'PY'
import hashlib
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

export_root, config, config_snapshot, config_sha256, checkpoint, manifest, family, fold, month, commands, registry = sys.argv[1:]
export_root = Path(export_root)
config = Path(config)
config_snapshot = Path(config_snapshot)
checkpoint = Path(checkpoint)
manifest = Path(manifest)
registry = Path(registry)
if hashlib.sha256(config_snapshot.read_bytes()).hexdigest() != config_sha256:
    raise ValueError("registered V5 training config snapshot hash changed before finalization")
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
        "config_sha256": config_sha256,
        "training_config_snapshot": str(config_snapshot),
        "training_config_snapshot_sha256": hashlib.sha256(config_snapshot.read_bytes()).hexdigest(),
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
  PENDING_ENTRIES+=("$export_root/pending_registry_entry.json")
done

if [[ "$REGISTER_PENDING" == true ]]; then
  printf 'REGISTER protocol=v5_osm_assisted family=%s entries=5 registry=%s\n' "$FAMILY" "$REGISTRY"
  if [[ "$DRY_RUN" == false ]]; then
    python "$ROOT/scripts/eval/register_registered_v5_embedding_export.py" \
      --registry "$REGISTRY" --pending-entry "${PENDING_ENTRIES[@]}"
  fi
fi
