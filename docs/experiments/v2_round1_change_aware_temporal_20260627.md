# V2 Round1 Change-Aware Temporal Contrast - 2026-06-27

## Purpose

Round1 tested a conservative temporal upgrade for the V2 monthly embedding
model. The change-aware temporal contrast loss encourages embeddings to stay
close in stable pixels and separate in pixels with large multi-source temporal
delta. The older temporal endpoint loss remains disabled for this run.

## Training

- Config: `configs/v2_round1_change_aware_temporal_20260627_023510.yaml`
- Output directory:
  `/data/xuannv_embedding/outputs/v2_round1_change_aware_temporal_20260627_023510`
- Hardware: Ascend NPU 0-5, 6-card DDP
- Epochs: 100
- Best checkpoint:
  `/data/xuannv_embedding/outputs/v2_round1_change_aware_temporal_20260627_023510/best.pt`
- Best epoch: 99
- Best validation loss: 2.088745

Validation loss improved throughout training:

| Epoch | Val loss |
| --- | ---: |
| 9 | 2.732803 |
| 19 | 2.598259 |
| 29 | 2.482980 |
| 39 | 2.358243 |
| 49 | 2.244358 |
| 59 | 2.187983 |
| 69 | 2.122833 |
| 79 | 2.099609 |
| 89 | 2.090799 |
| 99 | 2.088745 |

AMP produced several gradient-overflow/loss-scale reductions during training,
but the run recovered automatically and completed normally.

## Embedding Export

- Embedding root:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_round1_change_aware_temporal_20260627_023510_best_round1_change_aware_best`
- Exported monthly embedding maps: 4,464 (`744 patches x 6 months`)
- Metadata: `meta.json` in the embedding root

## Quick Downstream Evaluation

- Benchmark root:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round1_change_aware_temporal_20260627_023510_20260627_042300`
- Report:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round1_change_aware_temporal_20260627_023510_20260627_042300/POST_TRAINING_REPORT.md`
- AEF comparison:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round1_change_aware_temporal_20260627_023510_20260627_042300/comparison_vs_aef.md`
- Visualizations:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round1_change_aware_temporal_20260627_023510_20260627_042300/visualizations/index.md`

Macro comparison against the bug-fixed baseline:

| Metric | Bug-fixed baseline | Round1 |
| --- | ---: | ---: |
| Macro AUC | 0.7843 | 0.8005 |
| Macro F1_best | 0.1254 | 0.1276 |
| Macro mIoU | 0.0253 | 0.0270 |

Macro comparison against AEF:

| Metric | Round1 | AEF |
| --- | ---: | ---: |
| Macro AUC | 0.8005 | 0.7848 |
| Macro F1_best | 0.1276 | 0.1305 |
| Macro mIoU | 0.0270 | 0.0491 |

Per-task Round1 vs AEF:

| Task | AUC | F1_best | mIoU | Notes |
| --- | ---: | ---: | ---: | --- |
| construction | 0.8781 vs 0.8651 | 0.3067 vs 0.2116 | 0.0877 vs 0.0554 | Still strong. |
| building_change | 0.5955 vs 0.6828 | 0.0075 vs 0.0245 | 0.0037 vs 0.0062 | Weak. |
| farm_change | 0.8058 vs 0.6706 | 0.0101 vs 0.0147 | 0.0000 vs 0.0023 | AUC improved but segmentation unusable. |
| rubbish | 0.8453 vs 0.8124 | 0.0155 vs 0.0365 | 0.0058 vs 0.0038 | Sparse positive prediction remains weak. |
| construction_joint | 0.8780 vs 0.8929 | 0.2981 vs 0.3650 | 0.0379 vs 0.1776 | Improved over baseline F1, still far from AEF mIoU. |

## Takeaways

Round1 is a small positive step, not a breakthrough. It improved macro AUC,
F1_best, and mIoU slightly over the bug-fixed baseline, and it made the
monthly embedding PCA visibly more time-varying. However, the learned temporal
difference is still not aligned well enough with labeled change regions.

Round2 should not simply increase the same contrast loss weight. It should
target change-label alignment more directly, with special attention to:

- sparse positives and threshold calibration in change tasks;
- better positive/negative temporal pair mining;
- high-resolution optical/SAR temporal cues;
- construction_joint mIoU, where AEF remains much stronger.
