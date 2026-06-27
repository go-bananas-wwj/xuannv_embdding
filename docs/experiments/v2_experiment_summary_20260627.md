# V2 Experiment Summary - 2026-06-27

This document summarizes the recent V2 embedding experiments and downstream
benchmarks. It is intended as the quick reference for deciding which checkpoint
is the current baseline and what to improve next.

## Version Timeline

| Version | Main Change | Checkpoint / Embedding | Evaluation | Decision |
|---|---|---|---|---|
| Round2 P0 | Supervised change alignment baseline after P0 fixes | `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500/best.pt` | Full 5-fold | Strong baseline; beats AEF macro, but sparse tasks and threshold calibration still weak. |
| P1A | Label/fold diagnostics only | No model change | Diagnostics | Confirmed sparse Harbin folds have very few positives; model changes alone cannot solve all instability. |
| P1B quick | Positive-aware sampler + task-aware hard-negative supervised loss | `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt` | Quick fold-0 | Promising; macro beats AEF, strong rubbish improvement, but fold0 farm_change is too sparse for final judgment. |
| P1B full | Same P1B checkpoint, full validation | same as above | Full 5-fold | Current best upgrade baseline; macro beats AEF and improves target sparse tasks. |

## Macro Results

| Version | Eval Mode | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | vs AEF Summary |
|---|---|---:|---:|---:|---:|---:|---|
| AEF | Reference | 0.7848 | 0.1305 | 0.0846 | n/a | 0.0491 | Official comparison baseline. |
| Round2 P0 | Full 5-fold | 0.9397 | 0.2487 | 0.0967 | 0.2084 | 0.1220 | Wins all macro metrics. |
| P1B quick | Fold 0 | 0.9301 | 0.3148 | 0.1286 | n/a | 0.1481 | Quick check only; very promising but fold-sensitive. |
| P1B full | Full 5-fold | 0.9452 | 0.3007 | 0.1157 | n/a | 0.1360 | Current best validated macro result. |

Note: `F1@val_thr` is tracked in threshold reports. It is not always included in
AEF macro comparison tables, but it is operationally important because the
model is poorly calibrated at the default 0.5 threshold.

## Full 5-Fold Task Results

### Round2 P0

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU |
|---|---:|---:|---:|---:|---:|
| construction | 0.8943 | 0.3272 | 0.1574 | 0.2975 | 0.1755 |
| building_change | 0.9548 | 0.1586 | 0.0607 | 0.1241 | 0.0662 |
| farm_change | 0.9318 | 0.1711 | 0.0625 | 0.1142 | 0.0616 |
| rubbish | 0.9644 | 0.1461 | 0.0430 | 0.1101 | 0.0593 |
| construction_joint | 0.9530 | 0.4408 | 0.1599 | 0.3962 | 0.2476 |
| macro | 0.9397 | 0.2487 | 0.0967 | 0.2084 | 0.1220 |

### P1B Full 5-Fold

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | AEF Result |
|---|---:|---:|---:|---:|---:|---|
| construction | 0.9098 | 0.3705 | 0.1585 | 0.3476 | 0.2119 | Wins all metrics. |
| building_change | 0.9620 | 0.2524 | 0.0794 | 0.1161 | 0.0627 | Wins all metrics, but mIoU remains modest. |
| farm_change | 0.9450 | 0.1603 | 0.0543 | 0.1032 | 0.0575 | Wins AEF, but slightly below P0 on some metrics. |
| rubbish | 0.9954 | 0.4023 | 0.1825 | 0.2825 | 0.1710 | Large win; best P1B improvement. |
| construction_joint | 0.9139 | 0.3180 | 0.1038 | 0.2983 | 0.1770 | AUC wins; AEF still wins F1 metrics and narrowly wins mIoU. |
| macro | 0.9452 | 0.3007 | 0.1157 | n/a | 0.1360 | Wins all macro comparison metrics. |

## What Improved

- P1B clearly improves sparse target tasks overall, especially `rubbish` and
  `building_change`.
- Macro F1_best improves from Round2 P0 `0.2487` to P1B full `0.3007`.
- Macro mIoU improves from Round2 P0 `0.1220` to P1B full `0.1360`.
- AUC remains very strong, meaning the embedding has useful ranking signal.
- P1A diagnostics were correct: sparse fold imbalance is real and must be
  handled explicitly.

## What Regressed Or Remains Weak

- `construction_joint` regressed from P0 (`mIoU 0.2476`) to P1B (`mIoU 0.1770`).
  P1B almost ties AEF mIoU (`0.1770` vs `0.1776`), but still loses F1_best and
  F1@0.5.
- `farm_change` remains unstable. P1B full still beats AEF, but it does not
  clearly beat P0 on all metrics.
- Default threshold `0.5` is still not a good operating point. P1B full gains
  substantially from validation threshold calibration:
  - construction: `+0.1890` F1
  - building_change: `+0.0367` F1
  - farm_change: `+0.0488` F1
  - rubbish: `+0.1000` F1
  - construction_joint: `+0.1945` F1

## Current Recommended Baseline

Use P1B full as the current upgrade baseline for sparse-change focused work:

- Config: `configs/v2_p1_sparse_sampler_hardneg_20260627_090500.yaml`
- Checkpoint: `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- Full benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751`

Caveat: if the product priority is `construction_joint`, Round2 P0 remains a
strong comparison point because its construction_joint metrics are better.

## Recommended Next Step

Do not start with another broad sampler change. The next P2 should focus on
calibration and operating point quality:

1. Add a calibration report and calibrated inference artifact as a required
   output, not just an analysis table.
2. Add per-task threshold presets learned from validation folds, especially for
   `construction_joint` and `rubbish`.
3. Test downstream-side loss/head changes for `construction_joint` before
   retraining the embedding again.
4. Add the expanded capability suite promised in the plan: WorldCover land-cover
   segmentation and patch-level land-use classification first, then road/static
   building extraction once labels are available.

## Key Reports

- Round2 P0 full: `docs/experiments/v2_round2_p0_full5fold_20260627.md`
- P1 label diagnostics: `docs/experiments/v2_p1_label_fold_diagnostics_20260627.md`
- P1B quick: `docs/experiments/v2_p1_hardneg_quick_eval_20260627.md`
- P1B full: `docs/experiments/v2_p1_hardneg_full5fold_20260627.md`
