# V2 Post-Training Scale Workflow

Date: 2026-06-26

## AEF Comparison

Generated:

- `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/comparison_vs_aef.md`
- `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/comparison_vs_aef.json`

Summary:

| Metric | V2 gated fusion | AEF | Delta | Winner |
|---|---:|---:|---:|---|
| macro AUC | 0.7618 | 0.7848 | -0.0229 | AEF |
| macro F1-best | 0.1579 | 0.1305 | +0.0274 | V2 |
| macro F1@0.5 | 0.0594 | 0.0846 | -0.0252 | AEF |
| macro mIoU | 0.0749 | 0.0491 | +0.0258 | V2 |

Task-level read:

- `construction`: V2 wins all metrics by a large margin.
- `construction_joint`: AEF remains stronger, especially F1@0.5.
- `building_change`: AEF is stronger, but both are weak.
- `farm_change`: V2 has higher AUC, but AEF is slightly better on F1/mIoU.
- `rubbish`: AEF is stronger, but both are weak.

The change-task interpretation should remain cautious because Harbin labels
were found to be non-georeferenced and often duplicated across months.

## Output Visualizations

Generated 20 output strips:

- `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/visualizations/`
- Index:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/visualizations/index.md`

Each strip contains six horizontal panels:

1. high-resolution optical before (`202512`)
2. high-resolution optical after (`202605`)
3. embedding PCA before
4. embedding PCA after
5. PDA / delta embedding PCA
6. model prediction probability

Observation: the delta embedding PCA panel is nearly black in checked samples,
which means the exported monthly embedding maps are nearly identical between
`202512` and `202605`. This is a separate issue from downstream decoder quality
and should be investigated before relying on change-task conclusions.

## Experiment Directory Index

Created a non-destructive timestamped index:

- `/data/xuannv_embedding/experiments/by_date/README.md`
- `/data/xuannv_embedding/experiments/by_date/inventory.json`

The original experiment directories were not moved. The `by_date/` directory
contains timestamp-suffixed symlinks such as:

```text
aef_benchmark_20260621_130325 -> /data/xuannv_embedding/experiments/aef_benchmark
v2_202512_202605_20260626_144358 -> /data/xuannv_embedding/experiments/v2_202512_202605
```

Future post-training runs should use `run_name_YYYYMMDD_HHMMSS` output names.

## Scale Workflow

Added:

- `scripts/scale/post_training_eval.py`
- `scripts/scale/organize_experiment_dirs.py`
- `scripts/report/compare_benchmark_to_aef.py`
- `scripts/report/visualize_downstream_outputs.py`

Recommended command after every embedding export:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{embedding_run} \
  --run-name {experiment_name} \
  --npu 0 \
  --fold 0 \
  --samples-per-task 4
```

This runs quick downstream fold-0 evaluation for all five tasks, saves
predictions for visualization, compares against AEF and V1.0 gates, and writes
output strips.

For an existing benchmark directory:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/20260626_v2_embedding_haidian_harbin_202512_202605_gated_fusion_best_acceptance \
  --run-name v2_gated_fusion_best_acceptance \
  --benchmark-root /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance \
  --skip-training \
  --samples-per-task 4
```

Refresh the experiment directory index:

```bash
python scripts/scale/organize_experiment_dirs.py \
  --root /data/xuannv_embedding/experiments
```
