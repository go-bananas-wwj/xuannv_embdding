#!/usr/bin/env bash
# V1.2 national model 的 4-task × 5-fold 下游 benchmark 并行入口。
# 每个 fold 占用一张 NPU，5 folds 并发执行，任务间串行，避免磁盘/I/O 争用。
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
if [ ! -d "$EMB_ROOT" ] || [ -z "$(ls -A "$EMB_ROOT")" ]; then
    echo "==> Precomputing embeddings for haidian + harbin"
    python -m downstreams.scripts.precompute_embeddings \
        --config configs/v1.2_extract_labeled.yaml \
        --regions haidian harbin \
        --output-root "$EMB_ROOT" \
        --checkpoint "$NATIONAL_OUTPUT/v1.2_national_crossmodal/best.pt"
fi

# precompute_embeddings 会按日期创建子目录，取最新一个作为实际 embedding 根目录
EMB_SUBDIR=$(ls -td "$EMB_ROOT"/*/ 2>/dev/null | head -1)
if [ -z "$EMB_SUBDIR" ]; then
    echo "ERROR: no embedding subdirectory found under $EMB_ROOT"
    exit 1
fi
# 去掉末尾斜杠
EMB_SUBDIR="${EMB_SUBDIR%/}"
echo "==> Using embedding root: $EMB_SUBDIR"

# 2) 预生成 split 文件，避免并行 fold 竞争创建
echo "==> Pre-generating 5-fold split files"
python - <<'PY'
from pathlib import Path
from downstreams.data.split import create_stratified_folds, create_combined_stratified_folds

label_root = Path("/data/xuannv_embedding/processed")
tasks = {
    "construction": {"regions": ["haidian", "harbin"]},
    "building_change": {"region": "harbin"},
    "farm_change": {"region": "harbin"},
    "rubbish": {"region": "harbin"},
}

for task, spec in tasks.items():
    task_label = label_root / task
    task_label.mkdir(parents=True, exist_ok=True)
    if "regions" in spec:
        split_path = task_label / "split_joint_5fold.json"
        if split_path.exists():
            print(f"  exists {split_path}")
            continue
        mask_dirs = {r: label_root / r / "labels" / task / "masks" for r in spec["regions"]}
        split = create_combined_stratified_folds(mask_dirs, seed=42)
        split_path.write_text(__import__("json").dumps(split, ensure_ascii=False, indent=2))
        print(f"  wrote {split_path}")
    else:
        split_path = task_label / "split_5fold.json"
        if split_path.exists():
            print(f"  exists {split_path}")
            continue
        mask_dir = label_root / spec["region"] / "labels" / task / "masks"
        split = create_stratified_folds(mask_dir, seed=42)
        split_path.write_text(__import__("json").dumps(split, ensure_ascii=False, indent=2))
        print(f"  wrote {split_path}")
PY

# 3) 4-task × 5-fold，每个任务内 5 folds 并发
declare -A TASK_CONFIG=(
    [construction]="$CONFIG_ROOT/construction.yaml"
    [building_change]="$CONFIG_ROOT/building_change.yaml"
    [farm_change]="$CONFIG_ROOT/farm_change.yaml"
    [rubbish]="$CONFIG_ROOT/rubbish.yaml"
)

# 每任务可用的 NPU 编号（0-4，共 5 张）
GPUS=(0 1 2 3 4)

for task in construction building_change farm_change rubbish; do
    cfg="${TASK_CONFIG[$task]}"
    out_dir="$BENCH_ROOT/$task"
    mkdir -p "$out_dir"
    echo "==> Benchmark task: $task (5 folds in parallel)"

    pids=()
    for fold in {0..4}; do
        gpu="${GPUS[$fold]}"
        fold_out="$out_dir/fold_$fold"
        mkdir -p "$fold_out"
        if [ "$task" = "construction" ]; then
            ASCEND_VISIBLE_DEVICES="$gpu" \
                python -m downstreams.scripts.train_task \
                    --task "$task" \
                    --config "$cfg" \
                    --embedding-root "$EMB_SUBDIR" \
                    --label-root "$LABEL_ROOT" \
                    --regions haidian harbin \
                    --output-root "$fold_out" \
                    --months 202512 \
                    --fold "$fold" \
                    > "$fold_out/train.log" 2>&1 &
        else
            ASCEND_VISIBLE_DEVICES="$gpu" \
                python -m downstreams.scripts.train_task \
                    --task "$task" \
                    --config "$cfg" \
                    --embedding-root "$EMB_SUBDIR" \
                    --label-root "$LABEL_ROOT" \
                    --region harbin \
                    --output-root "$fold_out" \
                    --months 202512 \
                    --fold "$fold" \
                    > "$fold_out/train.log" 2>&1 &
        fi
        pids+=($!)
        echo "  fold $fold -> NPU $gpu (pid ${pids[-1]})"
    done

    # 等待该任务所有 fold 完成
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "ERROR: fold process $pid failed"; exit 1; }
    done

    # 汇总 5-fold 结果
    python -m downstreams.scripts.aggregate_5fold \
        --experiment-root "$out_dir" \
        --out-path "$out_dir/summary_5fold.json"
done

echo "==> Benchmark complete. Results under $BENCH_ROOT"
