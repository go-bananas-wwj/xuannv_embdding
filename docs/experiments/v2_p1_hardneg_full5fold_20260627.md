# V2 P1 Hard-Negative Full 5-Fold Evaluation - 2026-06-27

Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
Benchmark root: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751`
Evaluation mode: full 5-fold, downstream `concat_diff`, predictions and visualizations enabled.

## AEF Comparison

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | Note |
|---|---:|---:|---:|---:|---:|---|
| construction | 0.9098 | 0.3705 | 0.1585 | 0.3476 | 0.2119 | beats AEF on all listed metrics |
| building_change | 0.9620 | 0.2524 | 0.0794 | 0.1161 | 0.0627 | beats AEF; absolute mIoU remains modest |
| farm_change | 0.9450 | 0.1603 | 0.0543 | 0.1032 | 0.0575 | substantial full-fold improvement vs quick fold0 |
| rubbish | 0.9954 | 0.4023 | 0.1825 | 0.2825 | 0.1710 | strongest sparse-task result |
| construction_joint | 0.9139 | 0.3180 | 0.1038 | 0.2983 | 0.1770 | AUC improves; mIoU nearly ties AEF, but AEF still wins F1 metrics |

Macro vs AEF: AUC `0.9452` vs `0.7848` (+0.1604), F1_best `0.3007` vs `0.1305` (+0.1702), F1@0.5 `0.1157` vs `0.0846` (+0.0311), mIoU `0.1360` vs `0.0491` (+0.0870).

## Threshold Calibration

- Validation-threshold calibration improves every task.
- Gains: construction `+0.1890`, building_change `+0.0367`, farm_change `+0.0488`, rubbish `+0.1000`, construction_joint `+0.1945`.
- `construction_joint` mIoU is almost tied with AEF (`0.1770` vs `0.1776`), but `F1@0.5` remains far lower (`0.1038` vs `0.2966`), so calibration/default-threshold behavior is still a major problem.

## Decision

P1B is validated as a stronger full-fold model than AEF at the macro level and stronger than the P0 sparse baseline on the target sparse tasks. It should become the current upgrade baseline. The next fix should focus on calibration and `construction_joint` threshold/default operating point, not on another broad sampler change.

## Artifacts

- AEF comparison: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751/comparison_vs_aef.md`
- Threshold report: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751/threshold_calibration.md`
- Visualization index: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751/visualizations/index.md`
- Post-training report: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751/POST_TRAINING_REPORT.md`
