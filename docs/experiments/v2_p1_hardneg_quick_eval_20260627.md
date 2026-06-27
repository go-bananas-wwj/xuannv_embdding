# V2 P1 Hard-Negative Quick Evaluation - 2026-06-27

Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
Benchmark root: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_20260627_090500_20260627_121032`
Evaluation mode: quick fold-0, downstream `concat_diff`, predictions and visualizations enabled.

## AEF Comparison

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | Note |
|---|---:|---:|---:|---:|---:|---|
| construction | 0.9209 | 0.3696 | 0.1506 | 0.3647 | 0.2230 | beats AEF on all listed metrics |
| building_change | 0.9688 | 0.2732 | 0.1211 | 0.1241 | 0.0662 | beats AEF, but absolute mIoU remains low |
| farm_change | 0.8346 | 0.0534 | 0.0131 | 0.0134 | 0.0067 | beats AEF but still weak; fold0 val is extremely sparse |
| rubbish | 0.9990 | 0.5359 | 0.2624 | 0.4329 | 0.2763 | strong quick-fold improvement |
| construction_joint | 0.9273 | 0.3420 | 0.0958 | 0.2880 | 0.1683 | AUC improves, but AEF still wins F1_best/F1@0.5/mIoU |

Macro vs AEF: AUC `0.9301` vs `0.7848` (+0.1454), F1_best `0.3148` vs `0.1305` (+0.1844), F1@0.5 `0.1286` vs `0.0846` (+0.0440), mIoU `0.1481` vs `0.0491` (+0.0990).

## Threshold Calibration

- Calibration remains important: `construction`, `rubbish`, and `construction_joint` gain more than 0.17 F1 from validation-selected thresholds.
- `building_change` has a large oracle gap (`0.1491`), so validation threshold selection is still not finding the best test threshold reliably.
- `farm_change` remains too sparse/noisy for stable quick-fold conclusions.

## Artifacts

- AEF comparison: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_20260627_090500_20260627_121032/comparison_vs_aef.md`
- Threshold report: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_20260627_090500_20260627_121032/threshold_calibration.md`
- Visualization index: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_20260627_090500_20260627_121032/visualizations/index.md`
- Post-training report: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_20260627_090500_20260627_121032/POST_TRAINING_REPORT.md`

## Decision

This P1B quick fold result is promising enough to run full 5-fold validation. The strongest evidence is the macro AEF win and the large rubbish improvement. The main caution is that `farm_change` fold0 has extremely sparse validation positives, and `construction_joint` still trails AEF on segmentation metrics despite better AUC.
