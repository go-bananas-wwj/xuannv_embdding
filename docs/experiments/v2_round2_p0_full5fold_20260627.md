# V2 Round2 P0 Full 5-Fold Evaluation - 2026-06-27

## Run

- Baseline: Round2 supervised-change checkpoint.
- Embedding root:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_round2_supervised_change_20260627_050500_best_round2_supervised_best`
- Benchmark root:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_p0_full5fold_20260627_075306`
- Log:
  `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500/logs/p0_full5fold_eval_20260627_075306.log`
- Mode: full 5-fold, quick concat-diff downstream head config.
- Output size: 137M.

## Outputs

- AEF comparison:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_p0_full5fold_20260627_075306/comparison_vs_aef.md`
- Threshold calibration:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_p0_full5fold_20260627_075306/threshold_calibration.md`
- Visualizations:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_p0_full5fold_20260627_075306/visualizations/index.md`

## Metrics

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU |
|---|---:|---:|---:|---:|---:|
| construction | 0.8943 | 0.3272 | 0.1574 | 0.2975 | 0.1755 |
| building_change | 0.9548 | 0.1586 | 0.0607 | 0.1241 | 0.0662 |
| farm_change | 0.9318 | 0.1711 | 0.0625 | 0.1142 | 0.0616 |
| rubbish | 0.9644 | 0.1461 | 0.0430 | 0.1101 | 0.0593 |
| construction_joint | 0.9530 | 0.4408 | 0.1599 | 0.3962 | 0.2476 |
| macro | 0.9397 | 0.2487 | 0.0967 | 0.2084 | 0.1220 |

## Comparison Against AEF

| Metric | Round2 P0 full 5-fold | AEF | Delta |
|---|---:|---:|---:|
| macro AUC | 0.9397 | 0.7848 | +0.1549 |
| macro F1_best | 0.2487 | 0.1305 | +0.1183 |
| macro F1@0.5 | 0.0967 | 0.0846 | +0.0121 |
| macro mIoU | 0.1220 | 0.0491 | +0.0730 |

Round2 P0 beats AEF on every macro metric. It also beats AEF on every task/metric
pair except `construction_joint` F1@0.5, where AEF is still higher
(`0.2966` vs `0.1599`). The same task beats AEF by best-threshold F1 and mIoU,
so this remains primarily a calibration issue.

## Threshold Calibration

| Task | F1@0.5 | F1@val_thr | Gain | F1_best | Oracle gap | Mean val threshold |
|---|---:|---:|---:|---:|---:|---:|
| construction | 0.1574 | 0.2975 | +0.1401 | 0.3272 | 0.0297 | 0.8346 |
| building_change | 0.0607 | 0.1241 | +0.0633 | 0.1586 | 0.0345 | 0.7989 |
| farm_change | 0.0625 | 0.1142 | +0.0517 | 0.1711 | 0.0569 | 0.6542 |
| rubbish | 0.0430 | 0.1101 | +0.0671 | 0.1461 | 0.0360 | 0.7149 |
| construction_joint | 0.1599 | 0.3962 | +0.2363 | 0.4408 | 0.0446 | 0.9447 |

The largest practical gain is threshold calibration. Macro F1 improves from
0.0967 at threshold 0.5 to 0.2084 at validation-selected thresholds.

## Findings

- The full 5-fold result confirms that Round2 is a real improvement over AEF,
  not a fold-0 accident.
- AUC is strong across all tasks, especially sparse Harbin change tasks. This
  means ranking signal exists even when default-threshold F1 is low.
- `construction_joint` is the strongest task. It is stable across folds and has
  the largest calibration gain.
- Sparse Harbin tasks show high fold variance. `farm_change` and `rubbish` have
  folds that almost fail while other folds show strong signal. P1 should inspect
  per-fold positive-pixel distribution and add positive-aware sampling.
- The next model upgrade should keep P0 as the frozen baseline, then implement
  balanced supervised change training with positive-aware sampling, per-task
  positive weighting, and a lightweight delta auxiliary head.
