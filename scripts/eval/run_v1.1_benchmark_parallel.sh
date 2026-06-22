#!/bin/bash
set -e

# 并行版 5-task × 5-fold benchmark（4 个 harbin 任务并行，construction_joint 串行）。
# Usage: scripts/eval/run_v1.1_benchmark_parallel.sh <checkpoint_path> <output_root> [config]

CHECKPOINT="${1:?请提供 checkpoint 路径}"
OUTPUT_ROOT="${2:?请提供输出根目录}"
CONFIG="${3:-configs/v1.1_extract_labeled_all_stable.yaml}"
EMB_ROOT="/data/xuannv_embedding/embeddings/v1.1_labeled"

WORKTREE="/root/workspace/xuannv/.worktrees/feat-multitask-downstream"
cd "$WORKTREE"

mkdir -p "$OUTPUT_ROOT"

echo "==> Extracting v1.1 embeddings for all labeled patches ..."
python downstreams/scripts/precompute_embeddings.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --regions harbin haidian \
  --output-root "$EMB_ROOT"

EMB_DIR=$(ls -td "$EMB_ROOT"/*/ | head -1)
EMB_DIR="${EMB_DIR%/}"
echo "==> Embeddings: $EMB_DIR"

echo "==> Running 4 harbin tasks in parallel ..."
pids=()
for task in construction building_change farm_change rubbish; do
  mkdir -p "$OUTPUT_ROOT/$task"
  python downstreams/scripts/train_task.py \
    --config "downstreams/configs/aef_benchmark/${task}.yaml" \
    --task "$task" \
    --embedding-root "$EMB_DIR" \
    --label-root /data/xuannv_embedding/processed/harbin/labels \
    --output-root "$OUTPUT_ROOT/$task" > "$OUTPUT_ROOT/$task.log" 2>&1 &
  pids+=("$!")
  echo "  $task -> PID ${pids[-1]}"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "任务 ${pids[$i]} 失败"
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "有下游任务失败，退出"
  exit 1
fi

echo "==> Running construction_joint ..."
python downstreams/scripts/train_task.py \
  --config downstreams/configs/aef_benchmark/construction_joint.yaml \
  --task construction_joint \
  --embedding-root "$EMB_DIR" \
  --label-root /data/xuannv_embedding/processed \
  --regions harbin haidian \
  --output-root "$OUTPUT_ROOT/construction_joint" > "$OUTPUT_ROOT/construction_joint.log" 2>&1

echo "==> Generating comparison report ..."
python downstreams/scripts/generate_aef_report.py \
  --aef-root "$OUTPUT_ROOT" \
  --v10-path /data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json \
  --output "$OUTPUT_ROOT/V1.1_vs_AEF_REPORT.md" \
  --plot "$OUTPUT_ROOT/v1.1_vs_aef.png"

echo "==> Done. Report: $OUTPUT_ROOT/V1.1_vs_AEF_REPORT.md"
