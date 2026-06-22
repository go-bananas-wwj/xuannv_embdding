#!/bin/bash
set -e

# 6× NPU 并行版 5-task × 5-fold benchmark。
# Usage: scripts/eval/run_v1.1_benchmark_6card.sh <checkpoint_path> <output_root> [embedding_root]

CHECKPOINT="${1:?请提供 checkpoint 路径}"
OUTPUT_ROOT="${2:?请提供输出根目录}"
EMB_ROOT="${3:-}"

WORKTREE="/root/workspace/xuannv/.worktrees/feat-multitask-downstream"
cd "$WORKTREE"

mkdir -p "$OUTPUT_ROOT"

if [[ -z "$EMB_ROOT" ]]; then
  echo "==> Extracting v1.1 embeddings for all labeled patches ..."
  python downstreams/scripts/precompute_embeddings.py \
    --config configs/v1.1_extract_labeled_all_stable.yaml \
    --checkpoint "$CHECKPOINT" \
    --regions harbin haidian \
    --output-root /data/xuannv_embedding/embeddings/v1.1_labeled
  EMB_ROOT=$(ls -td /data/xuannv_embedding/embeddings/v1.1_labeled/*/ | head -1)
  EMB_ROOT="${EMB_ROOT%/}"
fi

echo "==> Embeddings: $EMB_ROOT"

echo "==> Running 4 harbin tasks in parallel on NPU 0-3 ..."
pids=()
tasks=(construction building_change farm_change rubbish)
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  mkdir -p "$OUTPUT_ROOT/$task"
  ASCEND_RT_VISIBLE_DEVICES="$i" python downstreams/scripts/train_task.py \
    --config "downstreams/configs/aef_benchmark/${task}.yaml" \
    --task "$task" \
    --embedding-root "$EMB_ROOT" \
    --label-root /data/xuannv_embedding/processed/harbin/labels \
    --output-root "$OUTPUT_ROOT/$task" > "$OUTPUT_ROOT/${task}.log" 2>&1 &
  pids+=("$!")
  echo "  $task -> NPU $i, PID ${pids[-1]}"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "任务 ${tasks[$i]} (PID ${pids[$i]}) 失败"
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "有下游任务失败，退出"
  exit 1
fi

echo "==> Running construction_joint on NPU 4 ..."
mkdir -p "$OUTPUT_ROOT/construction_joint"
ASCEND_RT_VISIBLE_DEVICES=4 python downstreams/scripts/train_task.py \
  --config downstreams/configs/aef_benchmark/construction_joint.yaml \
  --task construction_joint \
  --embedding-root "$EMB_ROOT" \
  --label-root /data/xuannv_embedding/processed \
  --regions harbin haidian \
  --output-root "$OUTPUT_ROOT/construction_joint" > "$OUTPUT_ROOT/construction_joint.log" 2>&1

echo "==> Generating V1.0 vs V1.1 comparison report ..."
python downstreams/scripts/generate_aef_report.py \
  --aef-root "$OUTPUT_ROOT" \
  --v10-path /data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json \
  --output "$OUTPUT_ROOT/V1.0_vs_V1.1_REPORT.md" \
  --plot "$OUTPUT_ROOT/v1.0_vs_v1.1.png"

echo "==> Generating V1.1 vs AEF 2025 comparison report/figure ..."
python scripts/eval/generate_v1.1_vs_aef_report.py \
  --v11-root "$OUTPUT_ROOT" \
  --output-dir /root/workspace/report/v1.1_benchmark

echo "==> Done. Reports:"
echo "  $OUTPUT_ROOT/V1.0_vs_V1.1_REPORT.md"
echo "  /root/workspace/report/v1.1_benchmark/V1.1_vs_AEF_REPORT.md"
