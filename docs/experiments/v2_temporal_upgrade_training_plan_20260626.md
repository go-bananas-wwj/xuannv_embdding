# V2 Temporal-Sensitive Embedding Upgrade Plan

Date: 2026-06-26

Status: proposal only. Do not execute training until approved.

## Executive Summary

The current V2 gated-fusion embedding has a likely P0 correctness issue:
`MonthlyEmbeddingModule` defaults to `ref_year=2025, ref_month=1`, while the V2
experiment uses months `202512..202605`. With `num_months=6`, these timestamps
map outside the module range and are replaced by the same learned
`missing_token`. This explains why exported `202512` and `202605` embedding maps
are visually and numerically almost identical.

Before adding larger architecture changes, fix the month reference path, add
sanity checks, and run a short temporal-sensitivity verification. Then add
masked-month reconstruction, change-aware contrastive objectives, and better
high-resolution temporal handling.

## Evidence From Code Review

Current V2 config:

- `configs/v2_embedding_haidian_harbin_202512_202605.yaml`
- months: `2025-12` to `2026-05`
- `num_months: 6`
- `stp.temporal_fusion: gated_sum`

Likely root issue:

- `AEFModel` constructs `MonthlyEmbeddingModule` with only `num_months`.
- `MonthlyEmbeddingModule` defaults to `ref_year=2025, ref_month=1`.
- Therefore:
  - `202512` maps to relative index `11`;
  - `202605` maps to relative index `16`;
  - both are out of range for `num_months=6`;
  - all months are filled by the same `missing_token`.

Additional temporal-smoothing factors:

- High-resolution optical/SAR are currently aggregated into one weighted
  temporal mean and copied to all months.
- WorldCover is static and copied to all months.
- Uniformity flattens `(B,T,D)` without same-scene temporal structure.
- vMF bottleneck normalizes output direction; this is allowed, but weak temporal
  differences can be compressed if the upstream signal is already missing.

## Literature And Best-Practice References

- [SatMAE](https://arxiv.org/abs/2207.08051): masked autoencoding for temporal
  and multispectral satellite imagery; uses temporal embeddings and independent
  masking across time.
- [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732): multi-temporal EO
  foundation model trained on 4.2M HLS time-series samples; incorporates
  temporal and location embeddings.
- [CROMA](https://arxiv.org/abs/2311.00566): combines contrastive
  radar-optical alignment with masked reconstruction for multimodal remote
  sensing representation learning.
- [CACo](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf):
  change-aware sampling and contrastive learning; useful for treating true
  long-term change as hard negatives while avoiding over-penalizing seasonal
  variation.
- [SeCo](https://arxiv.org/abs/2103.16607): seasonal contrast pretraining for
  remote sensing using temporal/seasonal natural augmentations.
- [AnySat](https://arxiv.org/abs/2412.14123): multimodal JEPA with
  resolution-adaptive encoders for heterogeneous EO data.
- [DOFA](https://arxiv.org/abs/2403.15356): wavelength-conditioned dynamic
  architecture for unified multimodal EO inputs.
- [Scale-MAE](https://arxiv.org/abs/2212.14532): scale-aware masked
  autoencoding; useful for handling input GSD differences while keeping a fixed
  output grid.

## Upgrade Roadmap

### Phase 0: Correctness Fix And Diagnostics

Goal: make monthly embeddings actually receive `202512..202605` observations.

Changes:

1. Add `ref_year/ref_month` or `start_month` to model config.
2. Pass the first configured month into `AEFModel -> MonthlyEmbeddingModule`.
3. Add a timestamp range assertion before training/export:
   - fail if all timestamps are out of monthly range;
   - log per-batch valid monthly mask count.
4. Add export-time temporal diagnostics:
   - `cosine(emb_202512, emb_202605)` mean/std;
   - pixel L2 delta mean/std;
   - percentage of pixels with delta above small epsilon;
   - delta PCA strip in post-training visualization.

Acceptance gate:

- A small forward pass must show nonzero monthly valid masks.
- Exported `202512` vs `202605` embeddings should no longer be bitwise
  identical.
- Existing tests plus one new month-reference test must pass.

Risk: low. This is a correctness fix.

### Phase 1: Re-run V2 With Correct Month Reference

Goal: establish a clean baseline before adding new objectives.

Run:

1. 1-epoch sanity on 6 NPU.
2. Full V2 training using current architecture.
3. Export embeddings.
4. Run `$xuannv-post-training-eval`.

Expected outcome:

- Construction should remain strong.
- Change tasks may improve because `concat_diff` will finally have nonzero
  input.
- If change tasks remain weak, inspect label QA and temporal diagnostics before
  changing architecture.

Risk: low-to-medium; training time cost only.

### Phase 2: Month-Aware High-Resolution Handling

Goal: stop high-resolution sources from forcing static embeddings.

Changes:

1. Bin high-resolution optical/SAR by month when files have usable dates.
2. Feed month-specific high-resolution features where available.
3. Use target masks so high-resolution reconstruction loss is applied only for
   months with that high-resolution source.
4. Keep fallback static high-resolution mean only for sources without dates.

Risk: medium. It touches batch preparation, model fusion, and reconstruction
targets.

### Phase 3: Masked Multi-Source Reconstruction

Goal: follow SatMAE/Prithvi-style temporal pretraining instead of plain
always-visible reconstruction.

Changes:

1. Add source/month/patch masking:
   - random spatial patch mask;
   - random whole-month mask;
   - random whole-source dropout.
2. Reconstruct masked S2/S1/Landsat values from the monthly embedding.
3. Use L1/Huber for continuous bands and CE for discrete labels.
4. Keep output resolution `128x128`.

Acceptance gate:

- Reconstruction loss on masked data improves during sanity training.
- Temporal delta diagnostics increase on known changing regions but stay small
  on stable regions.

Risk: medium.

### Phase 4: Change-Aware Temporal Objectives

Goal: make embedding sensitive to true change while not overreacting to season,
snow, clouds, or sensor noise.

Training signals:

1. Stable pairs: same patch adjacent months with low spectral/SAR difference;
   pull together.
2. Candidate change pairs: high NDBI/NDVI/S1/high-resolution difference or
   downstream change labels; push apart or predict delta.
3. Seasonal/noise pairs: medium difference; weak positive or ignored.

Implementation:

- Add a projection head for contrastive losses so the main embedding map is not
  over-constrained.
- Start with scene-level pooled embeddings, then add local patch/pixel contrast
  if stable.
- Use CACo-style hard mining after each short training cycle.

Risk: medium-to-high. Needs careful pair sampling and label/noise handling.

### Phase 5: Multimodal Alignment And Sensor Metadata

Goal: improve source robustness and sensor transfer.

Options:

1. CROMA-style cross-modal alignment:
   - S1/S2 same month positive;
   - S2/Landsat same patch-month positive;
   - source-specific projections.
2. DOFA-inspired metadata:
   - source id;
   - GSD;
   - wavelength/band group embedding.
3. Scale-MAE/AnySat-inspired GSD-aware fusion:
   - keep fixed `128x128` output;
   - condition source features on meter-per-pixel and sensor type.

Risk: high. Do only after Phases 0-3 are validated.

## Proposed First Execution After Approval

Step A: implement Phase 0 only.

Files likely touched:

- `src/xuannv_embedding/config.py`
- `src/xuannv_embedding/models/model.py`
- `src/xuannv_embedding/models/blocks.py`
- `scripts/train/train.py`
- `downstreams/downstreams/inference.py`
- tests under `tests/`
- V2 YAML config, self-contained.

Step B: run diagnostics without full training.

Commands:

```bash
PYTHONPATH=/root/workspace/xuannv/src:$PYTHONPATH \
pytest tests/test_model.py tests/test_training.py -q
```

Then run a small forward/export check that prints:

- timestamps;
- valid monthly mask count;
- `202512` vs `202605` embedding cosine/L2 on one batch.

Step C: 1-epoch 6-NPU sanity.

Step D: if sanity passes, run full training in tmux.

Step E: export embeddings and invoke:

```bash
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{new_embedding_run} \
  --run-name v2_month_ref_fix \
  --npu 0 \
  --fold 0 \
  --samples-per-task 4
```

## Decision Points

Ask for approval before:

1. implementing Phase 0 code changes;
2. launching full 6-NPU training;
3. implementing Phase 2 high-resolution temporal changes;
4. implementing Phase 4 change-aware objectives.

Recommended approval now: Phase 0 only.
