#!/usr/bin/env bash
# Run only the registered v5 full_150 encoder folds in isolated, auditable attempts.
set -euo pipefail

QUEUE_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$QUEUE_SCRIPT")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_ROOT="$ROOT/configs/paper_registered_v5_20260726"
OUTPUT_ROOT=/data/xuannv_embedding/outputs/paper_registered_v5_20260726
MATRIX="$ROOT/configs/eval/rse_v5_osm_assisted_matrix.json"
SPLIT="$ROOT/configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json"
SUBSETS="$ROOT/configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json"
STATISTICS="$ROOT/configs/eval/haidian_paper_v5_normalization_statistics.json"
EXPORT_REGISTRY="$ROOT/configs/eval/registered_embedding_exports_v5_20260726.json"
LANE_LOCK_ROOT=/data/xuannv_embedding/locks/registered_v5
if [[ -n "${PYTEST_CURRENT_TEST:-}" && "$ROOT" == /tmp/pytest-of-* ]]; then
  LANE_LOCK_ROOT="$ROOT/.pytest_registered_v5_locks"
fi
DRY_RUN=false
RESUME_FOLD=""
RESUME_ATTEMPT=""
RECOVER_FOLD=""
RECOVER_FROM_ATTEMPT=""
RECOVER_CHECKPOINT=""
ONLY_FOLD=""
VERIFY_ATTEMPT=""
MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS:-3}"

declare -a CONFIG_NAMES=(
  paper_registered_v5_full_150_fold0_20260726
  paper_registered_v5_full_150_fold1_20260726
  paper_registered_v5_full_150_fold2_20260726
  paper_registered_v5_full_150_fold3_20260726
  paper_registered_v5_full_150_fold4_20260726
)

usage() {
  cat >&2 <<'EOF'
usage: run_registered_v5_paper_queue.sh [--dry-run] [--output-root PATH]
       run_registered_v5_paper_queue.sh --resume-fold FOLD --resume-attempt N [--dry-run]
       run_registered_v5_paper_queue.sh --recover-fold FOLD --recover-from-attempt N --recover-checkpoint FILE
       run_registered_v5_paper_queue.sh --only-fold FOLD [--dry-run] [--output-root PATH]
       run_registered_v5_paper_queue.sh --verify-attempt PATH

The queue is intentionally limited to the five registered v5 full_150 folds.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --output-root) OUTPUT_ROOT=${2:-}; shift 2 ;;
    --resume-fold) RESUME_FOLD=${2:-}; shift 2 ;;
    --resume-attempt) RESUME_ATTEMPT=${2:-}; shift 2 ;;
    --recover-fold) RECOVER_FOLD=${2:-}; shift 2 ;;
    --recover-from-attempt) RECOVER_FROM_ATTEMPT=${2:-}; shift 2 ;;
    --recover-checkpoint) RECOVER_CHECKPOINT=${2:-}; shift 2 ;;
    --only-fold) ONLY_FOLD=${2:-}; shift 2 ;;
    --verify-attempt) VERIFY_ATTEMPT=${2:-}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

lane_for_fold() {
  case "$1" in
    0|3) printf '0 0,1 35601' ;;
    1|4) printf '1 2,3 35602' ;;
    2) printf '2 4,5 35603' ;;
    *) echo "unknown registered fold: $1" >&2; return 2 ;;
  esac
}

config_for_fold() {
  printf '%s/%s.yaml\n' "$CONFIG_ROOT" "${CONFIG_NAMES[$1]}"
}

require_git_head_file() {
  local path=$1
  local relative
  relative=$(realpath --relative-to="$ROOT" "$path")
  [[ "$relative" != ../* && -f "$path" ]] || {
    echo "registered protocol asset is outside the repository or missing: $path" >&2
    return 2
  }
  git -C "$ROOT" ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 || {
    echo "registered protocol asset is not Git tracked: $relative" >&2
    return 2
  }
  git -C "$ROOT" cat-file -e "HEAD:$relative" 2>/dev/null || {
    echo "registered protocol asset is absent from Git HEAD: $relative" >&2
    return 2
  }
  git -C "$ROOT" diff --quiet HEAD -- "$relative" || {
    echo "registered protocol asset differs from Git HEAD: $relative" >&2
    return 2
  }
}

require_queue_script_at_git_head() {
  local relative
  relative=$(realpath --relative-to="$ROOT" "$QUEUE_SCRIPT")
  [[ "$relative" != ../* && -f "$QUEUE_SCRIPT" ]] || {
    echo "queue script is outside the repository or missing: $QUEUE_SCRIPT" >&2
    return 2
  }
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "queue root is not a Git worktree: $ROOT" >&2
    return 2
  }
  git -C "$ROOT" ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 || {
    echo "queue script is not Git tracked: $relative" >&2
    return 2
  }
  git -C "$ROOT" cat-file -e "HEAD:$relative" 2>/dev/null || {
    echo "queue script is absent from Git HEAD: $relative" >&2
    return 2
  }
  git -C "$ROOT" diff --quiet HEAD -- "$relative" || {
    echo "queue script differs from Git HEAD: $relative" >&2
    return 2
  }
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all -- "$relative")" ]] || {
    echo "queue script owned worktree is not clean: $relative" >&2
    return 2
  }
}

validate_registered_inputs() {
  local path
  require_git_head_file "$ROOT/scripts/eval/registered_v5_encoder_checkpoint.py"
  for path in "${CONFIG_NAMES[@]}"; do
    require_git_head_file "$CONFIG_ROOT/$path.yaml"
  done
  for path in "$MATRIX" "$SPLIT" "$SUBSETS" "$STATISTICS" "$EXPORT_REGISTRY"; do
    require_git_head_file "$path"
  done
  python - "$MATRIX" "$CONFIG_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

matrix = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if matrix.get("protocol_id") != "v5_osm_assisted":
    raise SystemExit("registered matrix is not v5_osm_assisted")
families = matrix.get("families")
if not isinstance(families, dict) or families.get("full_150", {}).get("encoder_fold_count") != 5:
    raise SystemExit("registered matrix must declare exactly five full_150 encoders")
probe = matrix.get("probe")
if not isinstance(probe, dict) or probe.get("expected_job_count") != 90:
    raise SystemExit("registered matrix must declare exactly 90 Xuannv probes")

config_root = Path(sys.argv[2])
for fold in range(5):
    path = config_root / f"paper_registered_v5_full_150_fold{fold}_20260726.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = f"paper_registered_v5_full_150_fold{fold}_20260726"
    experiment = raw.get("experiment", {}) if isinstance(raw, dict) else {}
    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    if experiment.get("name") != expected:
        raise SystemExit(f"registered config name mismatch: {path}")
    if not str(experiment.get("output_dir", "")).startswith(
        "/data/xuannv_embedding/outputs/paper_registered_v5_20260726/"
    ):
        raise SystemExit(f"registered config uses a non-v5 output namespace: {path}")
    if data.get("paper_protocol_name") != "rse_v5_registered_20260726":
        raise SystemExit(f"registered config is not bound to the v5 protocol: {path}")
PY
}

verified_resume_checkpoint() {
  local attempt=$1
  local expected_fold=${2:-}
  local checkpoint_name=${3:-}
  python - "$attempt" "$expected_fold" "$checkpoint_name" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

attempt = Path(sys.argv[1]).resolve()
expected_fold_text = sys.argv[2]
checkpoint_name = sys.argv[3]
expected_fold = None if not expected_fold_text else int(expected_fold_text)
manifest_path = attempt / "attempt_manifest.json"
if attempt.name.startswith("attempt_") is False or not manifest_path.is_file():
    raise SystemExit("attempt manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("protocol_id") != "v5_osm_assisted" or manifest.get("family") != "full_150":
    raise SystemExit("attempt is not a registered v5 full_150 attempt")
if expected_fold is not None and manifest.get("fold") != expected_fold:
    raise SystemExit(
        f"attempt manifest fold {manifest.get('fold')!r} does not match expected fold {expected_fold}"
    )
if Path(str(manifest.get("attempt_dir", ""))).resolve() != attempt:
    raise SystemExit("attempt manifest does not bind this same attempt")
if checkpoint_name:
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise SystemExit("recovery checkpoint name must be a snapshot filename")
    verification_dir = attempt / "checkpoint_verifications"
    record_path = verification_dir / f"{checkpoint_name}.json"
    if verification_dir.is_symlink() or record_path.is_symlink():
        raise SystemExit("recovery verification record must not be a symlink")
    if record_path.resolve().parent != verification_dir.resolve():
        raise SystemExit("recovery verification record is outside its expected directory")
    records = [record_path]
else:
    records = sorted((attempt / "checkpoint_verifications").glob("*.json"))
valid = []
snapshot_dir = attempt / "verified_checkpoints"
if snapshot_dir.is_symlink():
    raise SystemExit("verified checkpoint directory must not be a symlink")
for record_path in records:
    if record_path.is_symlink() or record_path.resolve().parent != (attempt / "checkpoint_verifications").resolve():
        raise SystemExit("checkpoint verification record must remain inside its expected directory")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise SystemExit("checkpoint verification has an unsupported schema version")
    checkpoint = Path(str(record.get("checkpoint_path", ""))).resolve()
    try:
        checkpoint.relative_to(snapshot_dir)
    except ValueError as exc:
        raise SystemExit("verified resume checkpoint must remain inside verified_checkpoints") from exc
    if record_path.name != f"{checkpoint.name}.json":
        raise SystemExit("checkpoint verification filename does not bind the immutable snapshot")
    if Path(str(record.get("attempt_dir", ""))).resolve() != attempt:
        raise SystemExit("checkpoint verification does not bind this same attempt")
    source_checkpoint = Path(str(record.get("source_checkpoint_path", ""))).resolve()
    try:
        source_checkpoint.relative_to(attempt)
    except ValueError as exc:
        raise SystemExit("checkpoint verification source must remain inside the same attempt") from exc
    expected_snapshot_name = (
        f"{source_checkpoint.stem}_{record.get('checkpoint_sha256', '')}{source_checkpoint.suffix}"
    )
    if checkpoint.name != expected_snapshot_name:
        raise SystemExit("verified resume checkpoint is not a content-addressed snapshot")
    if checkpoint.is_symlink() or checkpoint.resolve().parent != snapshot_dir.resolve():
        raise SystemExit("verified resume checkpoint must remain inside its snapshot directory")
    if not checkpoint.is_file():
        raise SystemExit("verified resume checkpoint is missing")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual != record.get("checkpoint_sha256"):
        raise SystemExit("verified resume checkpoint hash changed")
    created_at_ns = record.get("created_at_ns")
    if isinstance(created_at_ns, bool) or not isinstance(created_at_ns, int) or created_at_ns <= 0:
        raise SystemExit("checkpoint verification has an invalid creation timestamp")
    if created_at_ns > time.time_ns() + 300_000_000_000:
        raise SystemExit("checkpoint verification creation timestamp is implausibly in the future")
    valid.append((created_at_ns, checkpoint.name, checkpoint))
if not valid:
    raise SystemExit("attempt has no verified same-attempt recovery checkpoint")
valid.sort(reverse=True)
print(valid[0][2])
PY
}

require_queue_script_at_git_head
cd "$ROOT"

[[ -z "$VERIFY_ATTEMPT" || ( -z "$RESUME_FOLD" && -z "$RESUME_ATTEMPT" && -z "$RECOVER_FOLD" && -z "$RECOVER_FROM_ATTEMPT" && -z "$RECOVER_CHECKPOINT" && -z "$ONLY_FOLD" ) ]] || {
  usage
  exit 2
}

if [[ -n "$VERIFY_ATTEMPT" ]]; then
  checkpoint=$(verified_resume_checkpoint "$VERIFY_ATTEMPT") || exit $?
  printf 'VERIFIED_RESUME attempt=%s checkpoint=%s\n' "$VERIFY_ATTEMPT" "$checkpoint"
  exit 0
fi

[[ -z "$RESUME_FOLD" && -z "$RESUME_ATTEMPT" ]] || [[ -n "$RESUME_FOLD" && -n "$RESUME_ATTEMPT" ]] || {
  usage
  exit 2
}
[[ -z "$RESUME_FOLD" || "$RESUME_FOLD" =~ ^[0-4]$ ]] || { usage; exit 2; }
[[ -z "$RESUME_ATTEMPT" || "$RESUME_ATTEMPT" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ -z "$RECOVER_FOLD" && -z "$RECOVER_FROM_ATTEMPT" && -z "$RECOVER_CHECKPOINT" ]] || \
  [[ -n "$RECOVER_FOLD" && -n "$RECOVER_FROM_ATTEMPT" && -n "$RECOVER_CHECKPOINT" ]] || { usage; exit 2; }
[[ -z "$RECOVER_FOLD" || "$RECOVER_FOLD" =~ ^[0-4]$ ]] || { usage; exit 2; }
[[ -z "$RECOVER_FROM_ATTEMPT" || "$RECOVER_FROM_ATTEMPT" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ -z "$ONLY_FOLD" || "$ONLY_FOLD" =~ ^[0-4]$ ]] || { usage; exit 2; }
[[ -z "$ONLY_FOLD" || ( -z "$RESUME_FOLD" && -z "$RECOVER_FOLD" ) ]] || { usage; exit 2; }
[[ -z "$RECOVER_FOLD" || -z "$RESUME_FOLD" ]] || { usage; exit 2; }
[[ "$MAX_JOB_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_JOB_ATTEMPTS must be positive" >&2; exit 2; }

validate_registered_inputs

write_event() {
  local attempt=$1
  local event=$2
  shift 2
  python - "$attempt" "$event" "$@" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

attempt = Path(sys.argv[1])
event = sys.argv[2]
status_dir = attempt / "status"
status_dir.mkdir(exist_ok=True)
payload = {
    "schema_version": 1,
    "event": event,
    "created_at_ns": time.time_ns(),
    "details": sys.argv[3:],
}
for sequence in range(1000):
    path = status_dir / f"{payload['created_at_ns']}_{sequence:03d}_{event}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        continue
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    break
else:
    raise SystemExit("unable to allocate immutable status event")
PY
}

write_attempt() {
  local attempt=$1
  local config=$2
  local fold=$3
  local parent_attempt=$4
  local recovery_checkpoint=${5:-}
  local recovery_record=${6:-}
  python - "$attempt" "$config" "$fold" "$parent_attempt" "$MATRIX" "$SPLIT" "$SUBSETS" "$STATISTICS" \
    "$EXPORT_REGISTRY" "$(git -C "$ROOT" rev-parse HEAD)" "$recovery_checkpoint" "$recovery_record" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

(
    attempt_text,
    config_text,
    fold_text,
    parent_text,
    matrix_text,
    split_text,
    subsets_text,
    statistics_text,
    exports_text,
    git_head,
    recovery_checkpoint_text,
    recovery_record_text,
) = sys.argv[1:]
attempt = Path(attempt_text).resolve()
config = Path(config_text).resolve()
raw = yaml.safe_load(config.read_text(encoding="utf-8"))
raw["experiment"]["output_dir"] = str(attempt)
raw["experiment"]["wandb_run_name"] = f"{raw['experiment']['wandb_run_name']}__{attempt.name}"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

recovery_parent = None
if recovery_checkpoint_text:
    parent = Path(parent_text).resolve()
    parent_manifest_path = parent / "attempt_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_launch_path = parent / "launch_config.yaml"
    parent_sidecar_path = Path(str(parent_manifest["manifest_sidecar"]["path"])).resolve()
    recovery_parent = {
        "checkpoint_path": str(Path(recovery_checkpoint_text).resolve()),
        "checkpoint_sha256": digest(Path(recovery_checkpoint_text)),
        "verification_record_path": str(Path(recovery_record_text).resolve()),
        "verification_record_sha256": digest(Path(recovery_record_text)),
        "attempt_manifest_path": str(parent_manifest_path.resolve()),
        "attempt_manifest_sha256": digest(parent_manifest_path),
        "launch_config_path": str(parent_launch_path.resolve()),
        "launch_config_sha256": digest(parent_launch_path),
        "manifest_sidecar_path": str(parent_sidecar_path),
        "manifest_sidecar_sha256": digest(parent_sidecar_path),
    }

def write_exclusive(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

launch_config = attempt / "launch_config.yaml"
write_exclusive(launch_config, yaml.safe_dump(raw, allow_unicode=False, sort_keys=False))
assets = {
    name: {"path": str(Path(path).resolve()), "sha256": digest(Path(path))}
    for name, path in {
        "matrix": matrix_text,
        "spatial_split": split_text,
        "subset_registry": subsets_text,
        "statistics_registry": statistics_text,
        "embedding_export_registry": exports_text,
    }.items()
}
manifest = {
    "schema_version": 1,
    "protocol_id": "v5_osm_assisted",
    "family": "full_150",
    "fold": int(fold_text),
    "attempt_dir": str(attempt),
    "git_head": git_head,
    "source_config": {"path": str(config), "sha256": digest(config)},
    "launch_config": {"path": str(launch_config), "sha256": digest(launch_config)},
    "protocol_assets": assets,
    "manifest_sidecar": {
        "path": str(raw["data"]["paper_manifest_audit"]),
        "sha256": str(raw["data"]["paper_manifest_audit_sha256"]),
    },
    "parent_attempt": None if not parent_text else parent_text,
    "recovery_parent": recovery_parent,
    # Concrete hashes are sealed into write-once checkpoint-verification records.
    "checkpoint_hashes": {},
}
write_exclusive(attempt / "attempt_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

seal_checkpoint() {
  local attempt=$1
  local checkpoint=$2
  python - "$attempt" "$checkpoint" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

attempt = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2]).resolve()
try:
    checkpoint.relative_to(attempt)
except ValueError as exc:
    raise SystemExit("checkpoint must remain inside the same attempt") from exc
if not checkpoint.is_file():
    raise SystemExit("checkpoint is missing")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
snapshot_dir = attempt / "verified_checkpoints"
snapshot_dir.mkdir(exist_ok=True)
snapshot = snapshot_dir / f"{checkpoint.stem}_{digest}{checkpoint.suffix}"
def wait_for_snapshot() -> None:
    deadline = time.monotonic() + 300.0
    while True:
        try:
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest:
                return
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    raise SystemExit("verified checkpoint snapshot hash mismatch")

def read_stable_record(path: Path) -> dict:
    deadline = time.monotonic() + 300.0
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise SystemExit("checkpoint verification record was not written atomically")

if not snapshot.exists():
    try:
        descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        # Another recovery path sealed the same content-addressed snapshot first.
        pass
    else:
        with checkpoint.open("rb") as source, os.fdopen(descriptor, "wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
wait_for_snapshot()
record = {
    "schema_version": 1,
    "attempt_dir": str(attempt),
    "checkpoint_path": str(snapshot),
    "checkpoint_sha256": digest,
    "source_checkpoint_path": str(checkpoint),
}
record_path = attempt / "checkpoint_verifications" / f"{snapshot.name}.json"
record_path.parent.mkdir(exist_ok=True)
if record_path.exists():
    existing = read_stable_record(record_path)
    if any(existing.get(key) != value for key, value in record.items()):
        raise SystemExit("existing checkpoint verification disagrees with the checkpoint snapshot")
else:
    record["created_at_ns"] = time.time_ns()
    try:
        descriptor = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = read_stable_record(record_path)
        if any(existing.get(key) != value for key, value in record.items() if key != "created_at_ns"):
            raise SystemExit("existing checkpoint verification disagrees with the checkpoint snapshot")
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
print(snapshot)
PY
}

write_canonical_attempt() {
  local job_root=$1
  local attempt=$2
  local checkpoint=$3
  python - "$job_root" "$attempt" "$checkpoint" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

job_root = Path(sys.argv[1]).resolve()
attempt = Path(sys.argv[2]).resolve()
checkpoint = Path(sys.argv[3]).resolve()
manifest = attempt / "attempt_manifest.json"
payload = {
    "schema_version": 1,
    "attempt_dir": str(attempt),
    "attempt_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
}
payload["sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
pointer = job_root / "canonical_attempt.json"
descriptor = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

canonical_is_verified() {
  local job_root=$1
  [[ -f "$job_root/canonical_attempt.json" ]] || return 1
  python "$ROOT/scripts/eval/registered_v5_encoder_checkpoint.py" --job-root "$job_root" >/dev/null
}

next_attempt_number() {
  local job_root=$1
  local largest=0
  local path number
  shopt -s nullglob
  for path in "$job_root"/attempt_[0-9]*; do
    number=${path##*_}
    [[ "$number" =~ ^[0-9]+$ ]] && (( number > largest )) && largest=$number
  done
  shopt -u nullglob
  printf '%d\n' "$((largest + 1))"
}

latest_recovery_checkpoint() {
  local attempt=$1
  find "$attempt" -maxdepth 1 -type f \( -name 'recovery.pt' -o -name 'epoch_*.pt' \) \
    -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

next_resume_log_number() {
  local attempt=$1
  local largest=0
  local path number
  shopt -s nullglob
  for path in "$attempt"/logs/resume_*.log; do
    number=${path##*/resume_}
    number=${number%.log}
    [[ "$number" =~ ^[1-9][0-9]*$ ]] && (( number > largest )) && largest=$number
  done
  shopt -u nullglob
  printf '%d\n' "$((largest + 1))"
}

seal_log() {
  local temporary=$1
  local final=$2
  ln -- "$temporary" "$final" || {
    echo "refusing to overwrite immutable log: $final" >&2
    return 2
  }
  rm -- "$temporary" || return 2
}

run_attempt() {
  local lane=$1 devices=$2 port=$3 fold=$4 attempt=$5 resume_checkpoint=${6:-}
  local name=${CONFIG_NAMES[$fold]}
  local job_root="$OUTPUT_ROOT/$name"
  local log_tmp log_final log_dir
  local -a resume_args=()
  if [[ -n "$resume_checkpoint" ]]; then
    resume_args=(--resume "$resume_checkpoint")
    log_dir="$attempt/logs"
    mkdir -p "$log_dir"
    local resume_log_number
    resume_log_number=$(next_resume_log_number "$attempt")
    log_tmp=$(mktemp "$attempt/.resume_${resume_log_number}.running.XXXXXX")
    log_final="$log_dir/resume_${resume_log_number}.log"
  else
    log_dir="$job_root/logs"
    mkdir -p "$log_dir"
    log_tmp="$job_root/.${attempt##*/}.running.log"
    log_final="$log_dir/${attempt##*/}.log"
  fi
  write_event "$attempt" start "lane=$lane" "devices=$devices" "resume=${resume_checkpoint:-none}"
  set +e
  ASCEND_RT_VISIBLE_DEVICES="$devices" HCCL_IF_BASE_PORT="$port" \
    torchrun --standalone --master_port "$port" --nproc_per_node=2 "$ROOT/scripts/train/train.py" \
      --config "$attempt/launch_config.yaml" "${resume_args[@]}" \
      >"$log_tmp" 2>&1
  local status=$?
  set -e
  if ! seal_log "$log_tmp" "$log_final"; then
    write_event "$attempt" log_seal_failed "exit=$status" "log=$log_final"
    return 5
  fi
  if (( status == 0 )) && [[ -f "$attempt/best.pt" ]]; then
    local verified
    verified=$(seal_checkpoint "$attempt" "$attempt/best.pt")
    write_event "$attempt" succeeded "checkpoint=$verified" "checkpoint_sha256=$(sha256 "$verified")"
    write_canonical_attempt "$job_root" "$attempt" "$verified"
    return 0
  fi
  local recovery
  recovery=$(latest_recovery_checkpoint "$attempt" || true)
  if [[ -n "$recovery" ]]; then
    local verified
    verified=$(seal_checkpoint "$attempt" "$recovery")
    write_event "$attempt" failed "exit=$status" "checkpoint=$verified" "checkpoint_sha256=$(sha256 "$verified")"
  else
    write_event "$attempt" failed "exit=$status" "checkpoint=none"
  fi
  (( status == 0 )) && status=4
  return "$status"
}

run_fold() {
  local lane=$1 devices=$2 port=$3 fold=$4
  local name=${CONFIG_NAMES[$fold]}
  local job_root="$OUTPUT_ROOT/$name"
  local config
  config=$(config_for_fold "$fold")
  if [[ -f "$job_root/canonical_attempt.json" ]]; then
    canonical_is_verified "$job_root"
    printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s state=canonical\n' \
      "$lane" "$devices" "$fold" "$config"
    return 0
  fi
  if [[ -n "$RESUME_FOLD" ]]; then
    local attempt="$job_root/attempt_$RESUME_ATTEMPT"
    local checkpoint
    checkpoint=$(verified_resume_checkpoint "$attempt" "$fold") || return $?
    printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s attempt=%s resume=%s\n' \
      "$lane" "$devices" "$fold" "$config" "$RESUME_ATTEMPT" "$checkpoint"
    [[ "$DRY_RUN" == true ]] || run_attempt "$lane" "$devices" "$port" "$fold" "$attempt" "$checkpoint"
    return
  fi
  if [[ "$DRY_RUN" == true ]]; then
    local next_attempt=1
    if [[ -d "$job_root" ]]; then next_attempt=$(next_attempt_number "$job_root"); fi
    printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s attempt=%s resume=fresh\n' \
      "$lane" "$devices" "$fold" "$config" "$next_attempt"
    return 0
  fi
  local attempt_number attempt parent="" checkpoint resume_index verification_error reason
  mkdir -p "$job_root"
  attempt_number=$(next_attempt_number "$job_root")
  attempt="$job_root/attempt_$attempt_number"
  if (( attempt_number > 1 )); then parent="$job_root/attempt_$((attempt_number - 1))"; fi
  mkdir "$attempt"
  write_attempt "$attempt" "$config" "$fold" "$parent"
  printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s attempt=%s resume=fresh\n' \
    "$lane" "$devices" "$fold" "$config" "$attempt_number"
  if run_attempt "$lane" "$devices" "$port" "$fold" "$attempt"; then return 0; fi
  for ((resume_index = 1; resume_index <= MAX_JOB_ATTEMPTS; resume_index++)); do
    verification_error=$(mktemp "$attempt/.resume_verification_error.XXXXXX")
    if ! checkpoint=$(verified_resume_checkpoint "$attempt" "$fold" 2>"$verification_error"); then
      reason=$(tr '\n' ' ' < "$verification_error" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')
      rm -f "$verification_error"
      write_event "$attempt" recovery_verification_failed "resume_index=$resume_index" "reason=${reason:-unknown}"
      return 1
    fi
    rm -f "$verification_error"
    printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s attempt=%s resume=%s\n' \
      "$lane" "$devices" "$fold" "$config" "$attempt_number" "$checkpoint"
    if run_attempt "$lane" "$devices" "$port" "$fold" "$attempt" "$checkpoint"; then return 0; fi
  done
  return 1
}

recover_fold_from_snapshot() {
  local lane=$1 devices=$2 port=$3 fold=$4 source_attempt_number=$5 checkpoint_name=$6
  local name=${CONFIG_NAMES[$fold]}
  local job_root="$OUTPUT_ROOT/$name"
  local config source_attempt checkpoint attempt_number attempt
  config=$(config_for_fold "$fold")
  source_attempt="$job_root/attempt_$source_attempt_number"
  python - "$job_root" "$source_attempt" "$config" "$MATRIX" "$SPLIT" "$SUBSETS" "$STATISTICS" \
    "$EXPORT_REGISTRY" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

(
    job_root_text,
    source_attempt_text,
    config_text,
    matrix_text,
    split_text,
    subsets_text,
    statistics_text,
    exports_text,
) = sys.argv[1:]
job_root = Path(job_root_text).resolve()
source_attempt = Path(source_attempt_text)
if source_attempt.is_symlink():
    raise SystemExit("recovery parent attempt must not be a symlink")
resolved_attempt = source_attempt.resolve()
if resolved_attempt.parent != job_root or resolved_attempt.name != source_attempt.name:
    raise SystemExit("recovery parent attempt is outside the expected fold job root")
manifest_path = resolved_attempt / "attempt_manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit("recovery parent attempt manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

config = Path(config_text).resolve()
source_config = manifest.get("source_config")
if not isinstance(source_config, dict):
    raise SystemExit("recovery parent source config is missing")
if Path(str(source_config.get("path", ""))).resolve() != config:
    raise SystemExit("recovery parent source config path differs from the registered config")
if source_config.get("sha256") != digest(config):
    raise SystemExit("recovery parent source config hash differs from the registered config")
launch_config = manifest.get("launch_config")
expected_launch_path = resolved_attempt / "launch_config.yaml"
if not isinstance(launch_config, dict):
    raise SystemExit("recovery parent launch config is missing")
recorded_launch_path = Path(str(launch_config.get("path", "")))
if recorded_launch_path.is_symlink() or recorded_launch_path.resolve() != expected_launch_path:
    raise SystemExit("recovery parent launch config must remain inside the parent attempt")
if not expected_launch_path.is_file() or digest(expected_launch_path) != launch_config.get("sha256"):
    raise SystemExit("recovery parent launch config hash differs from its manifest")
registered_config = yaml.safe_load(config.read_text(encoding="utf-8"))
expected_launch = json.loads(json.dumps(registered_config))
expected_launch["experiment"]["output_dir"] = str(resolved_attempt)
expected_launch["experiment"]["wandb_run_name"] = (
    f"{expected_launch['experiment']['wandb_run_name']}__{resolved_attempt.name}"
)
if yaml.safe_load(expected_launch_path.read_text(encoding="utf-8")) != expected_launch:
    raise SystemExit("recovery parent launch config differs from the registered config")
expected_sidecar = Path(str(registered_config["data"]["paper_manifest_audit"])).resolve()
sidecar = manifest.get("manifest_sidecar")
if not isinstance(sidecar, dict) or Path(str(sidecar.get("path", ""))).resolve() != expected_sidecar:
    raise SystemExit("recovery parent manifest sidecar path differs from the registered config")
if (
    sidecar.get("sha256") != registered_config["data"]["paper_manifest_audit_sha256"]
    or not expected_sidecar.is_file()
    or digest(expected_sidecar) != sidecar.get("sha256")
):
    raise SystemExit("recovery parent manifest sidecar hash differs from the registered config")
expected_assets = {
    "matrix": Path(matrix_text),
    "spatial_split": Path(split_text),
    "subset_registry": Path(subsets_text),
    "statistics_registry": Path(statistics_text),
    "embedding_export_registry": Path(exports_text),
}
assets = manifest.get("protocol_assets")
if not isinstance(assets, dict):
    raise SystemExit("recovery parent protocol assets are missing")
for name, path in expected_assets.items():
    recorded = assets.get(name)
    if (
        not isinstance(recorded, dict)
        or Path(str(recorded.get("path", ""))).resolve() != path.resolve()
        or recorded.get("sha256") != digest(path)
    ):
        raise SystemExit(f"recovery parent protocol asset differs: {name}")
PY
  checkpoint=$(verified_resume_checkpoint "$source_attempt" "$fold" "$checkpoint_name") || return $?
  [[ ! -f "$job_root/canonical_attempt.json" ]] || {
    echo "cannot recover Fold $fold: canonical attempt already exists" >&2
    return 1
  }
  attempt_number=$(next_attempt_number "$job_root")
  attempt="$job_root/attempt_$attempt_number"
  mkdir "$attempt"
  write_attempt "$attempt" "$config" "$fold" "$source_attempt" "$checkpoint" \
    "$source_attempt/checkpoint_verifications/$checkpoint_name.json"
  printf 'ENCODER protocol=v5_osm_assisted family=full_150 lane=%s devices=%s fold=%s config=%s attempt=%s recovery_parent=%s resume=%s\n' \
    "$lane" "$devices" "$fold" "$config" "$attempt_number" "$source_attempt_number" "$checkpoint"
  run_attempt "$lane" "$devices" "$port" "$fold" "$attempt" "$checkpoint"
}

run_lane() {
  local lane=$1 devices=$2 port=$3
  shift 3
  if [[ "$DRY_RUN" == false ]]; then
    assert_lane_is_idle "$lane"
    acquire_lane_lease "$lane"
  fi
  local fold failed=0
  for fold in "$@"; do
    if ! run_fold "$lane" "$devices" "$port" "$fold"; then failed=1; fi
  done
  return "$failed"
}

assert_lane_is_idle() {
  local lane=$1 occupant
  local -a occupants=()
  case "$lane" in
    0) occupants=(0 3) ;;
    1) occupants=(1 4) ;;
    2) occupants=(2) ;;
    *) echo "unknown registered lane: $lane" >&2; return 2 ;;
  esac
  for occupant in "${occupants[@]}"; do
    if pgrep -f "${ROOT}/scripts/train/train.py.*paper_registered_v5_full_150_fold${occupant}_" >/dev/null; then
      echo "cannot start lane $lane: Fold $occupant is active" >&2
      return 1
    fi
  done
}

acquire_lane_lease() {
  local lane=$1
  mkdir -p "$LANE_LOCK_ROOT"
  exec {REGISTERED_V5_LANE_LEASE_FD}>"$LANE_LOCK_ROOT/lane_${lane}.lock"
  if ! flock -n "$REGISTERED_V5_LANE_LEASE_FD"; then
    echo "cannot start registered lane $lane: lane lease is active" >&2
    return 1
  fi
}

if [[ "$DRY_RUN" == false ]]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
  export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/downstreams:${PYTHONPATH:-}"
fi

if [[ -n "$RESUME_FOLD" ]]; then
  read -r lane devices port <<< "$(lane_for_fold "$RESUME_FOLD")"
  run_lane "$lane" "$devices" "$port" "$RESUME_FOLD"
  exit $?
fi

if [[ -n "$RECOVER_FOLD" ]]; then
  read -r lane devices port <<< "$(lane_for_fold "$RECOVER_FOLD")"
  assert_lane_is_idle "$lane"
  acquire_lane_lease "$lane"
  recover_fold_from_snapshot "$lane" "$devices" "$port" "$RECOVER_FOLD" \
    "$RECOVER_FROM_ATTEMPT" "$RECOVER_CHECKPOINT"
  exit $?
fi

if [[ -n "$ONLY_FOLD" ]]; then
  read -r lane devices port <<< "$(lane_for_fold "$ONLY_FOLD")"
  run_lane "$lane" "$devices" "$port" "$ONLY_FOLD"
  exit $?
fi

if [[ "$DRY_RUN" == true ]]; then
  for fold in 0 1 2 3 4; do
    read -r lane devices port <<< "$(lane_for_fold "$fold")"
    run_fold "$lane" "$devices" "$port" "$fold"
  done
  exit 0
fi

run_lane 0 0,1 35601 0 3 & lane0_pid=$!
run_lane 1 2,3 35602 1 4 & lane1_pid=$!
run_lane 2 4,5 35603 2 & lane2_pid=$!
failed=0
for pid in "$lane0_pid" "$lane1_pid" "$lane2_pid"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
