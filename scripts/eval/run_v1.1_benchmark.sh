#!/bin/bash
set -e

# Run full 5-task × 5-fold benchmark for a trained v1.1 checkpoint.
# Usage: scripts/eval/run_v1.1_benchmark.sh [checkpoint_path] [output_root]

CHECKPOINT="${1:-}"
OUTPUT_ROOT="${2:-/data/xuannv_embedding/experiments/v1.1_distill_long_stable_50ep_benchmark}"
CONFIG="${3:-configs/v1.1_extract_labeled_all_stable.yaml}"
EMB_ROOT="/data/xuannv_embedding/embeddings/v1.1_labeled"

WORKTREE="/root/workspace/xuannv/.worktrees/feat-multitask-downstream"
cd "$WORKTREE"

mkdir -p "$OUTPUT_ROOT"

echo "==> Extracting v1.1 embeddings for all labeled patches ..."
CKPT_ARGS=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_ARGS="--checkpoint $CHECKPOINT"
fi
python downstreams/scripts/precompute_embeddings.py \
  --config "$CONFIG" \
  $CKPT_ARGS \
  --regions harbin haidian \
  --output-root "$EMB_ROOT"

# Find the latest extraction directory
EMB_DIR=$(ls -td "$EMB_ROOT"/*/ | head -1)
EMB_DIR="${EMB_DIR%/}"
echo "==> Embeddings: $EMB_DIR"

echo "==> Running 5-fold downstream tasks ..."
for task in construction building_change farm_change rubbish; do
  echo "--- $task ---"
  python downstreams/scripts/train_task.py \
    --config "downstreams/configs/aef_benchmark/${task}.yaml" \
    --task "$task" \
    --embedding-root "$EMB_DIR" \
    --label-root /data/xuannv_embedding/processed/harbin/labels \
    --output-root "$OUTPUT_ROOT/$task"
done

echo "--- construction_joint ---"
python downstreams/scripts/train_task.py \
  --config downstreams/configs/aef_benchmark/construction_joint.yaml \
  --task construction_joint \
  --embedding-root "$EMB_DIR" \
  --label-root /data/xuannv_embedding/processed \
  --regions harbin haidian \
  --output-root "$OUTPUT_ROOT/construction_joint"

echo "==> Generating comparison report ..."
python downstreams/scripts/generate_aef_report.py \
  --aef-root "$OUTPUT_ROOT" \
  --v10-path /data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json \
  --output "$OUTPUT_ROOT/V1.1_vs_AEF_REPORT.md" \
  --plot "$OUTPUT_ROOT/v1.1_vs_aef.png"

echo "==> Done. Report: $OUTPUT_ROOT/V1.1_vs_AEF_REPORT.md"
