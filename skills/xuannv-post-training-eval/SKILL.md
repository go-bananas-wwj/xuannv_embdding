---
name: xuannv-post-training-eval
description: Run Xuannv embedding post-training evaluation after a checkpoint or embedding export, including quick downstream folds, AEF/V1 comparison reports, output-strip visualizations, and timestamped experiment indexing. Use when asked to evaluate a newly trained Xuannv embedding model, compare it with AEF, generate downstream prediction visualizations, or organize experiment outputs.
---

# Xuannv Post-Training Eval

Use this skill after a Xuannv embedding checkpoint has been exported to frozen
embedding maps. The workflow is script-driven and keeps experiment outputs under
`/data/xuannv_embedding/experiments/`.

## Preconditions

- Work from `/root/workspace/xuannv`.
- Ensure the embedding export directory exists, for example:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/{embedding_run}`.
- Keep code/config/docs in git; keep data, predictions, images, and benchmark
  outputs under `/data/xuannv_embedding/`.
- Preserve CANN paths when setting `PYTHONPATH`:
  `PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams:$PYTHONPATH`.

## Standard Workflow

Run the scale workflow:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{embedding_run} \
  --run-name {experiment_name} \
  --npu 0 \
  --fold 0 \
  --samples-per-task 4
```

This performs:

1. quick downstream fold-0 evaluation for five tasks;
2. prediction GeoTIFF export for visualization;
3. AEF comparison via `scripts/report/compare_benchmark_to_aef.py`;
4. V1 acceptance comparison via `scripts/report/compare_v2_acceptance.py`;
5. horizontal output strips via `scripts/report/visualize_downstream_outputs.py`;
6. a `POST_TRAINING_REPORT.md` in the benchmark directory.

Default tasks:

- `construction`
- `building_change`
- `farm_change`
- `rubbish`
- `construction_joint`

## Existing Benchmark

If downstream summaries already exist and only reports/visualizations need to be
refreshed:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{embedding_run} \
  --run-name {experiment_name} \
  --benchmark-root /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/{benchmark_run} \
  --skip-training \
  --samples-per-task 4
```

## Full Acceptance

Use `--all-folds` only after quick fold-0 results and label QA look plausible:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{embedding_run} \
  --run-name {experiment_name}_full \
  --all-folds \
  --samples-per-task 4
```

## Output Checklist

After completion, report these paths to the user:

- benchmark root;
- `comparison_vs_aef.md`;
- `comparison_vs_v1.0.md`;
- `visualizations/index.md`;
- `POST_TRAINING_REPORT.md`.

Also summarize:

- task-level AUC/F1_best/mIoU against AEF;
- macro metrics;
- whether delta embedding visualizations show temporal sensitivity;
- any missing source imagery or embedding files from `visualizations/metadata.json`.

## Experiment Index

Refresh the non-destructive timestamped experiment index:

```bash
python scripts/scale/organize_experiment_dirs.py \
  --root /data/xuannv_embedding/experiments
```

The index lives at:

- `/data/xuannv_embedding/experiments/by_date/README.md`
- `/data/xuannv_embedding/experiments/by_date/inventory.json`

Do not move old experiment directories unless the user explicitly approves a
destructive cleanup.
