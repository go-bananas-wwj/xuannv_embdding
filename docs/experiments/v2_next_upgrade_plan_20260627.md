# V2 Next Upgrade Plan - 2026-06-27

## Current Baseline

The current frozen baseline is Round2 supervised-change training:

- Config: `configs/v2_round2_supervised_change_20260627_050500.yaml`
- Checkpoint: `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500/best.pt`
- Embedding export:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_round2_supervised_change_20260627_050500_best_round2_supervised_best`
- Benchmark:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_round2_supervised_change_20260627_050500_20260627_072030`

Quick fold-0 macro metrics: AUC 0.9551, F1_best 0.2691, F1@0.5 0.1143,
mIoU 0.1203. It beats AEF on all macro metrics, but `construction_joint`
F1@0.5 is still weaker than AEF. The gap between best-threshold F1 and F1@0.5
shows that calibration is now a high-value target.

## External References

- SatMAE: https://arxiv.org/abs/2207.08051
  - Useful idea: masked autoencoding for temporal and multispectral satellite
    imagery, with temporal embeddings and independent masking across time.
- CROMA: https://arxiv.org/abs/2311.00566
  - Useful idea: combine radar-optical contrastive learning with masked
    reconstruction for aligned optical/SAR samples.
- Scale-MAE: https://ai-climate.berkeley.edu/scale-mae-website/
  - Useful idea: make scale explicit during pretraining and reconstruct both
    low-frequency and high-frequency targets.
- Prithvi-EO-2.0: https://arxiv.org/html/2412.02732v3
  - Useful idea: larger multi-temporal EO pretraining, temporal/location
    embeddings, repeated benchmarking, and small-label fine-tuning validation.
- CDMamba: https://arxiv.org/html/2406.04207v2
  - Useful idea: change detection benefits from explicit bi-temporal interaction
    and combining global context with local detail.
- Remote sensing foundation model survey:
  https://www.mdpi.com/2072-4292/17/2/179
  - Useful idea: remote sensing needs domain-specific foundation models because
    natural-image foundation models do not directly match EO data geometry,
    modalities, scales, and sparse labels.

## Proposed Upgrade Order

### P0: Lock the Baseline More Rigorously

Before changing the model again, run the same post-training process in a fuller
evaluation mode:

1. Run full 5-fold downstream evaluation for Round2.
2. Report mean/std per task and compare against AEF.
3. Add threshold calibration per task, because current best-threshold F1 is much
   higher than F1@0.5.
4. Generate the same visualization strips with GT for all representative tasks.

Expected benefit: reduce uncertainty and recover practical F1@0.5 without
touching the embedding backbone.

### P1: Round3 Balanced Supervised Change Training

Keep the current architecture mostly intact and strengthen the supervision path:

1. Add positive-aware patch sampling so sparse change labels appear more often.
2. Add per-task positive weighting in supervised change alignment loss.
3. Add a lightweight auxiliary delta head from `embedding_t2 - embedding_t1` to
   predict change masks during pretraining.
4. Keep the original reconstruction and embedding objectives so the embedding
   remains general-purpose.

Expected benefit: improve sparse Harbin classes and reduce unchanged-area false
positives.

### P2: Temporal Masked Reconstruction

Borrow the safest part of SatMAE/Prithvi:

1. Randomly hide one or more months during training.
2. Ask the model to reconstruct the missing month or missing modality features.
3. Preserve 128x128 output resolution.

Expected benefit: make embeddings depend on temporal evidence instead of only
single-date appearance.

### P3: Cross-Modal Optical/SAR Alignment

Borrow the safest part of CROMA:

1. Treat aligned optical/SAR observations as positive pairs.
2. Add a small cross-modal contrastive loss between optical-only and SAR-only
   branches.
3. Keep modality dropout so inference remains robust when a source is missing.

Expected benefit: improve cloudy/noisy scene robustness and reduce modality
overfitting.

### P4: Scale and Context Upgrade

Borrow from Scale-MAE and dense change detection models:

1. Feed a larger context crop while keeping the final embedding map at 128x128.
2. Add low-frequency and high-frequency reconstruction targets.
3. Test only after P0-P3, because this has more engineering risk.

Expected benefit: improve boundary quality and large-object context reasoning.

## Recommended Next Step

Execute P0 first. If the full evaluation confirms the Round2 gains, run P1 as
the next training experiment. P2/P3 should be introduced one at a time after P1,
otherwise it will be hard to know which change caused a metric movement.
