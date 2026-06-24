#!/usr/bin/env bash
# V1.2 national model 的 4-task × 5-fold 下游 benchmark（可指定月份组合）。
# 用法示例：
#   MONTHS="202604" OUT_SUFFIX="202604" bash scripts/benchmark/run_v1.2_4task_5fold_by_months.sh
#   MONTHS="202512 202604" OUT_SUFFIX="bitemporal" bash scripts/benchmark/run_v1.2_4task_5fold_by_months.sh
#
# 注意：construction 使用单时相 head（upernet），若传入多个月份会自动取最后一个月份。

set -euo pipefail

NATIONAL_OUTPUT="/data2/xuannv_embedding/national/outputs"
EMB_ROOT="/data2/xuannv_embedding/national/embeddings_v1.2"
LABEL_ROOT="/data/xuannv_embedding/processed"
OUT_SUFFIX="${OUT_SUFFIX:-${MONTHS// /_}}"
BENCH_ROOT="/data2/xuannv_embedding/national/downstream_benchmark_${OUT_SUFFIX}"
CONFIG_ROOT="downstreams/configs/aef_benchmark"

# 月份参数
if [ -z "${MONTHS:-}" ]; then
    echo "ERROR: please set MONTHS, e.g. MONTHS='202604' or MONTHS='202512 202604'"
    exit 1
fi

mkdir -p "$BENCH_ROOT"

# 取最新 embedding 子目录
EMB_SUBDIR=$(ls -td "$EMB_ROOT"/*/ 2>/dev/null | head -1)
if [ -z "$EMB_SUBDIR" ]; then
    echo "ERROR: no embedding subdirectory found under $EMB_ROOT"
    exit 1
fi
EMB_SUBDIR="${EMB_SUBDIR%/}"
echo "==> Using embedding root: $EMB_SUBDIR"
echo "==> Months: $MONTHS"
echo "==> Output root: $BENCH_ROOT"

# 预生成 split 文件（如尚未存在）
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
            continue
        mask_dirs = {r: label_root / r / "labels" / task / "masks" for r in spec["regions"]}
        split = create_combined_stratified_folds(mask_dirs, seed=42)
        split_path.write_text(__import__("json").dumps(split, ensure_ascii=False, indent=2))
        print(f"wrote {split_path}")
    else:
        split_path = task_label / "split_5fold.json"
        if split_path.exists():
            continue
        mask_dir = label_root / spec["region"] / "labels" / task / "masks"
        split = create_stratified_folds(mask_dir, seed=42)
        split_path.write_text(__import__("json").dumps(split, ensure_ascii=False, indent=2))
        print(f"wrote {split_path}")
PY

declare -A TASK_CONFIG=(
    [construction]="$CONFIG_ROOT/construction.yaml"
    [building_change]="$CONFIG_ROOT/building_change.yaml"
    [farm_change]="$CONFIG_ROOT/farm_change.yaml"
    [rubbish]="$CONFIG_ROOT/rubbish.yaml"
)

GPUS=(0 1 2 3 4)
TMP_CONFIG_DIR="$BENCH_ROOT/.tmp_configs"
mkdir -p "$TMP_CONFIG_DIR"

for task in construction building_change farm_change rubbish; do
    cfg="${TASK_CONFIG[$task]}"
    out_dir="$BENCH_ROOT/$task"
    mkdir -p "$out_dir"

    if [ -f "$out_dir/summary_5fold.json" ]; then
        echo "==> $task already done, skipping"
        continue
    fi

    # construction 仅支持单时相；多月份时取最后一个（通常是 202604）
    task_months="$MONTHS"
    if [ "$task" = "construction" ]; then
        n_months=$(echo "$MONTHS" | wc -w)
        if [ "$n_months" -gt 1 ]; then
            task_months=$(echo "$MONTHS" | awk '{print $NF}')
            echo "==> construction uses single month: $task_months"
        fi
    fi

    echo "==> Benchmark task: $task (months=$task_months, 5 folds in parallel)"

    pids=()
    for fold in {0..4}; do
        gpu="${GPUS[$fold]}"
        fold_out="$out_dir/fold_$fold"
        mkdir -p "$fold_out"

        fold_cfg="$TMP_CONFIG_DIR/${task}_fold${fold}.yaml"
        python - "$cfg" "$gpu" "$fold_cfg" <<'PY'
import sys
from pathlib import Path
import yaml
cfg_path, gpu, out_path = sys.argv[1:4]
data = yaml.safe_load(open(cfg_path))
data.setdefault("experiment", {})["device"] = f"npu:{gpu}"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(data, open(out_path, "w"), allow_unicode=True, sort_keys=False)
PY

        if [ "$task" = "construction" ]; then
            python -m downstreams.scripts.train_task \
                --task "$task" \
                --config "$fold_cfg" \
                --embedding-root "$EMB_SUBDIR" \
                --label-root "$LABEL_ROOT" \
                --regions haidian harbin \
                --output-root "$fold_out" \
                --months $task_months \
                --fold "$fold" \
                > "$fold_out/train.log" 2>&1 &
        else
            python -m downstreams.scripts.train_task \
                --task "$task" \
                --config "$fold_cfg" \
                --embedding-root "$EMB_SUBDIR" \
                --label-root "$LABEL_ROOT/harbin/labels" \
                --region harbin \
                --output-root "$fold_out" \
                --months $task_months \
                --fold "$fold" \
                > "$fold_out/train.log" 2>&1 &
        fi
        pids+=($!)
        echo "  fold $fold -> NPU $gpu (pid ${pids[-1]})"
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "ERROR: fold process $pid failed"; exit 1; }
    done

    python -m downstreams.scripts.aggregate_5fold \
        --experiment-root "$out_dir" \
        --out-path "$out_dir/summary_5fold.json"
done

echo "==> Benchmark complete. Results under $BENCH_ROOT"
