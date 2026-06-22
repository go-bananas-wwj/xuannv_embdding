#!/bin/bash
set -e

# 快速版 5-task × 5-fold benchmark（下游 head 训练 epoch/patience 较少，用于快速验证）。
# Usage: scripts/eval/run_v1.1_benchmark_quick.sh <checkpoint_path> <output_root> [embedding_root]

CHECKPOINT="${1:?请提供 checkpoint 路径}"
OUTPUT_ROOT="${2:?请提供输出根目录}"
EMB_ROOT="${3:-}"

WORKTREE="/root/workspace/xuannv/.worktrees/feat-multitask-downstream"
cd "$WORKTREE"

mkdir -p "$OUTPUT_ROOT"

if [[ -z "$EMB_ROOT" ]]; then
  EMB_ROOT=$(ls -td /data/xuannv_embedding/embeddings/v1.1_labeled/*/ | head -1)
  EMB_ROOT="${EMB_ROOT%/}"
fi

echo "==> Embeddings: $EMB_ROOT"

echo "==> Running 5-fold downstream tasks (quick configs) ..."
for task in construction building_change farm_change rubbish; do
  echo "--- $task ---"
  python downstreams/scripts/train_task.py \
    --config "downstreams/configs/quick_benchmark/${task}.yaml" \
    --task "$task" \
    --embedding-root "$EMB_ROOT" \
    --label-root /data/xuannv_embedding/processed/harbin/labels \
    --output-root "$OUTPUT_ROOT/$task"
done

echo "--- construction_joint ---"
python downstreams/scripts/train_task.py \
  --config downstreams/configs/quick_benchmark/construction_joint.yaml \
  --task construction_joint \
  --embedding-root "$EMB_ROOT" \
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
