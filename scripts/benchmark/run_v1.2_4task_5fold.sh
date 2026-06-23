#!/usr/bin/env bash
# V1.2 national model 的 4-task × 5-fold 下游 benchmark 入口。
# 前置条件：
#   1. full training 完成，checkpoint 位于 $NATIONAL_OUTPUT/v1.2_national_crossmodal/best.pt
#   2. 已运行 precompute_embeddings 生成 embedding_root

set -euo pipefail

NATIONAL_OUTPUT="/data2/xuannv_embedding/national/outputs"
EMB_ROOT="/data2/xuannv_embedding/national/embeddings_v1.2"
LABEL_ROOT="/data/xuannv_embedding/processed"
BENCH_ROOT="/data2/xuannv_embedding/national/downstream_benchmark"
CONFIG_ROOT="downstreams/configs/aef_benchmark"

mkdir -p "$BENCH_ROOT"

# 1) 提取 embedding（如尚未提取）
if [ ! -d "$EMB_ROOT" ]; then
    echo "==> Precomputing embeddings for haidian + harbin"
    python -m downstreams.scripts.precompute_embeddings \
        --config configs/v1.2_extract_labeled.yaml \
        --regions haidian harbin \
        --output-root "$EMB_ROOT" \
        --checkpoint "$NATIONAL_OUTPUT/v1.2_national_crossmodal/best.pt"
fi

# 2) 4-task × 5-fold
declare -A TASK_CONFIG=(
    [construction]="$CONFIG_ROOT/construction.yaml"
    [building_change]="$CONFIG_ROOT/building_change.yaml"
    [farm_change]="$CONFIG_ROOT/farm_change.yaml"
    [rubbish]="$CONFIG_ROOT/rubbish.yaml"
)

for task in construction building_change farm_change rubbish; do
    cfg="${TASK_CONFIG[$task]}"
    out_dir="$BENCH_ROOT/$task"
    echo "==> Benchmark task: $task"
    if [ "$task" = "construction" ]; then
        # construction 在 haidian+harbin 联合训练
        python -m downstreams.scripts.train_task \
            --task "$task" \
            --config "$cfg" \
            --embedding-root "$EMB_ROOT" \
            --label-root "$LABEL_ROOT" \
            --regions haidian harbin \
            --output-root "$out_dir" \
            --months 202512
    else
        # 其他任务仅在 harbin 有标注
        python -m downstreams.scripts.train_task \
            --task "$task" \
            --config "$cfg" \
            --embedding-root "$EMB_ROOT" \
            --label-root "$LABEL_ROOT" \
            --region harbin \
            --output-root "$out_dir" \
            --months 202512
    fi

    # 汇总 5-fold 结果
    python -m downstreams.scripts.aggregate_5fold \
        --experiment-root "$out_dir"
done

# 3) 总表
echo "==> Summarizing all tasks"
python -m downstreams.scripts.summarize_all_tasks \
    --experiment-root "$BENCH_ROOT" \
    --output "$BENCH_ROOT/all_tasks_summary.json"

echo "==> Benchmark complete. Summary: $BENCH_ROOT/all_tasks_summary.json"
