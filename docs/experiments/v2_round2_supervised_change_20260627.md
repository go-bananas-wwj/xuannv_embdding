# V2 Round2 Supervised Change Evaluation - 2026-06-27

## Run

- Config: `configs/v2_round2_supervised_change_20260627_050500.yaml`
- Training output: `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500`
- Best checkpoint: `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500/best.pt`
- Final training epoch: 100
- Best validation loss: 2.090356 at epoch 99

## Export

- Embedding root:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_round2_supervised_change_20260627_050500_best_round2_supervised_best`
- Export coverage:
  - Haidian: 320 patches, complete.
  - Harbin: 424 patches, complete.
  - Each patch has 6 monthly scene embeddings and 6 monthly 128x128 embedding maps.

## Quick Downstream Evaluation

- Benchmark root:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_20260627_050500_20260627_072030`
- Mode: quick fold 0, `concat_diff`, months 202512 and 202605.
- Report:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_20260627_050500_20260627_072030/POST_TRAINING_REPORT.md`
- AEF comparison:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_20260627_050500_20260627_072030/comparison_vs_aef.md`
- Visualizations:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_20260627_050500_20260627_072030/visualizations/index.md`

## Metrics

| Task | AUC | F1_best | F1@0.5 | mIoU |
|---|---:|---:|---:|---:|
| construction | 0.8752 | 0.2891 | 0.1625 | 0.1660 |
| building_change | 0.9611 | 0.1459 | 0.0728 | 0.0566 |
| farm_change | 0.9828 | 0.1850 | 0.1144 | 0.0594 |
| rubbish | 0.9850 | 0.1935 | 0.0675 | 0.0934 |
| construction_joint | 0.9715 | 0.5319 | 0.1541 | 0.2259 |
| macro | 0.9551 | 0.2691 | 0.1143 | 0.1203 |

## Comparison Against AEF

| Metric | Round2 | AEF | Delta |
|---|---:|---:|---:|
| macro AUC | 0.9551 | 0.7848 | +0.1704 |
| macro F1_best | 0.2691 | 0.1305 | +0.1386 |
| macro F1@0.5 | 0.1143 | 0.0846 | +0.0296 |
| macro mIoU | 0.1203 | 0.0491 | +0.0712 |

Round2 beats AEF on all macro metrics in quick fold 0. It also beats AEF on every
task/metric pair except `construction_joint` F1@0.5, where AEF remains higher
(`0.2966` vs `0.1541`). `construction_joint` still wins by best-threshold F1 and
mIoU.

## Notes

- This is a quick fold-0 result, not a full 5-fold evaluation.
- Training emitted several AMP gradient-overflow messages around epoch 42, then
  recovered after reducing loss scale to 32768.
- The supervised change alignment objective produced a much stronger downstream
  signal than Round1. The largest remaining weakness is calibration at threshold
  0.5 for `construction_joint` and still-low absolute mIoU on sparse Harbin
  change classes.
