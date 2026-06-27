# V2 P1 Sparse Change Fix Plan - 2026-06-27

## Why This Round

P0 full 5-fold confirmed that Round2 is a real improvement over AEF, but it also
exposed the next core problem:

- Macro AUC is strong: 0.9397.
- Macro F1 at threshold 0.5 is weak: 0.0967.
- Macro F1 at validation-selected thresholds is much better: 0.2084.
- Sparse Harbin tasks have high fold variance. Some folds nearly fail even when
  other folds have useful signal.

This means the embedding has ranking signal, but sparse positives, unstable
sampling, and threshold calibration are limiting practical segmentation quality.

## References

- Focal Loss for dense detection: https://arxiv.org/abs/1708.02002
  - Useful idea: down-weight easy background examples and focus learning on hard
    positives/negatives.
- Generalized Dice for imbalanced segmentation:
  https://arxiv.org/abs/1707.03237
  - Useful idea: compensate for class-size imbalance in dense masks.
- Unified Focal Loss: https://arxiv.org/abs/2102.04525
  - Useful idea: combine distribution-based and region-based losses for highly
    imbalanced segmentation.
- HRANet hard region adaptive network:
  https://ieeexplore.ieee.org/document/9847221
  - Useful idea: change detection benefits from mining hard regions instead of
    averaging all easy background pixels.
- ChangeStar: https://arxiv.org/abs/2108.07002
  - Useful idea: introduce an explicit change branch/head for remote-sensing
    change detection while sharing image representations.
- ChangeStar2 / STAR: https://arxiv.org/abs/2406.04207
  - Useful idea: train on richer temporal pairs and strengthen temporal-change
    supervision when exact dense labels are limited.

## P1 Goal

Improve sparse change stability without throwing away the current strong Round2
baseline. The target is not only higher quick fold-0 metrics; the target is
better full 5-fold metrics, especially:

- `building_change`, `farm_change`, `rubbish` F1_best and mIoU.
- Macro F1 at validation threshold.
- Lower fold-to-fold variance.

## Step 1: Label and Split Diagnostics

Before changing training, add a diagnostic script that reports:

1. Positive-pixel count per patch and per task.
2. Positive-patch count per fold for each downstream split.
3. Train/val/test positive-pixel totals per fold.
4. Patches in failing folds, especially `farm_change` fold1 and `rubbish` fold2.
5. A markdown report plus CSV/JSON output.

Expected output:

- `/data/xuannv_embedding/experiments/v2_202512_202605/diagnostics/p1_label_fold_diagnostics_YYYYMMDD_HHMMSS/`
- Repo-side summary in `docs/experiments/`.

Why first: if a fold has almost no positive pixels in train or val, model changes
will not fix the evaluation instability by themselves.

## Step 1.5: Expanded Downstream Capability Suite

Add a separate downstream suite so the embedding is evaluated beyond sparse
change labels. This suite should be reported alongside, but not blindly averaged
with, the P0/P1 sparse-change benchmark because label quality and task semantics
are different.

### Immediate Tasks From Existing Labels

1. `landcover_multiclass_segmentation`
   - Label source: existing `worldcover` labels under
     `/data/xuannv_embedding/processed/{haidian,harbin}/labels/worldcover`.
   - Head: linear probe or lightweight segmentation head on frozen embeddings.
   - Metrics: mean IoU, macro F1, pixel accuracy, per-class IoU.
   - Caveat: WorldCover-like labels are also part of the pretraining target, so
     this is a representation sanity probe rather than a fully independent
     held-out capability claim.
2. `lulc_patch_classification`
   - Label source: patch-level majority class derived from `worldcover`.
   - Head: classification head using pooled scene/patch embeddings.
   - Metrics: accuracy, macro F1, per-class recall.
   - Purpose: test whether the embedding carries global land-use context, not
     only dense pixel boundaries.

### Tasks Requiring Label Inventory or Preparation

1. `building_extraction`
   - Preferred source: local static building footprint masks, if present.
   - Fallback source: derive a weak built-up mask from WorldCover built-up class
     for quick probing, clearly labeled as weak/pseudo supervision.
   - Better source for later: rasterized building footprints from a trusted
     local GIS layer, OSM, Microsoft building footprints, or manual labels.
   - Metrics: F1_best, F1 at validation threshold, mIoU, precision/recall.
2. `road_extraction`
   - Preferred source: local road masks or rasterized vector road centerlines.
   - If no local road labels exist, prepare a small, versioned label-building
     step before running the benchmark. Do not treat generated pseudo labels as
     equal to hand-checked GT.
   - Metrics: F1_best, F1 at validation threshold, mIoU, precision/recall. Add
     a relaxed-distance or skeleton-aware metric later if road masks are thin.

### Reporting Rules

- Keep two macro summaries:
  - `sparse_change_macro`: construction/change labels used for P0/P1.
  - `capability_macro`: landcover, building, road, and patch classification.
- Mark each task as one of:
  - `gt`: human-checked or trusted independent label.
  - `weak_gt`: coarse external label such as WorldCover-derived built-up.
  - `pseudo`: generated labels used only for debugging.
- Every post-training report should include a task availability table so we can
  see which downstream tasks were actually run for that checkpoint.

## Step 2: Positive-Aware Pretraining Sampler

Current Round2 uses ordinary shuffled patches. P1 should cache per-patch
supervised positive counts and sample more positive patches during embedding
training.

Implementation plan:

1. Add dataset-side label statistics cache:
   - patch id
   - region
   - per-task mask availability
   - per-task positive pixels
   - total supervised positive pixels
2. Add config fields:
   - `data.supervised_sampling.enabled`
   - `data.supervised_sampling.positive_boost`
   - `data.supervised_sampling.max_weight`
   - `data.supervised_sampling.task_weights`
3. Add a distributed-safe weighted sampler for 6-card DDP:
   - sample a deterministic global index list per epoch
   - shard by rank
   - keep epoch seeding compatible with DDP
4. Keep a conservative cap, for example `max_weight: 8.0`, so the model still
   sees enough stable/background patches.

Expected effect: batches contain enough rare positive examples for the supervised
change loss to matter every epoch.

## Step 3: Rebalance Supervised Change Alignment Loss

Current `supervised_change_alignment_loss` aggregates positives and negatives
across all tasks. P1 should make it task-aware and hard-negative-aware.

Implementation plan:

1. Compute loss per task, then average with configurable task weights.
2. Keep positive hinge:
   - positives should have endpoint distance above `pos_margin`.
3. Replace all-negative averaging with hard-negative mining:
   - compute negative hinge map
   - keep only top-k or top-ratio negative pixels per task
   - default `hard_negative_ratio: 0.05`
4. Add separate weights:
   - `supervised_change_pos_weight`
   - `supervised_change_neg_weight`
   - `supervised_change_task_weights`
5. Log per-task:
   - positive pixels
   - negative pixels sampled
   - positive loss
   - hard-negative loss

Expected effect: easy unchanged pixels no longer dominate the gradient, while
false-positive-prone areas still receive pressure.

## Step 4: Add Optional Delta Auxiliary Head

After Steps 1-3 are stable, add a small training-only change head that predicts
sparse masks from endpoint embeddings.

Input candidates:

- `embedding_t1`
- `embedding_t2`
- `abs(embedding_t2 - embedding_t1)`
- optionally signed difference

Head:

- lightweight 1x1/3x3 conv head
- one output per supervised task
- preserves 128x128 spatial resolution

Loss:

- BCE + Dice/Tversky or focal-style sparse mask loss
- only applied where labels exist
- low weight, e.g. `delta_aux_weight: 0.02-0.05`

Expected effect: force the embedding space to contain linearly decodable change
information, without making the exported embedding depend on the downstream head.

## Step 5: Calibration as a First-Class Report

Keep threshold calibration in every post-training evaluation. Add explicit
acceptance metrics:

- `F1@0.5`
- `F1@val_thr`
- `F1_best`
- `oracle_gap = F1_best - F1@val_thr`
- fold standard deviation

Expected effect: avoid rejecting a good representation because the default 0.5
threshold is bad.

## Proposed Experiment Order

### P1A: Diagnostics Only

- Add label/fold diagnostic script.
- No model changes.
- Commit and push.

### P1B: Sampler + Task-Aware Hard-Negative Loss

- Add positive-aware sampler.
- Add task-aware supervised alignment loss.
- Add self-contained config:
  `configs/v2_p1_sparse_change_sampler_loss_YYYYMMDD_HHMMSS.yaml`
- Run a short sanity train first.
- Export embeddings.
- Run quick downstream + threshold calibration.

### P1C: Full P1B Validation

- If quick eval improves or keeps macro AUC while improving sparse F1, run full
  5-fold.
- Compare against P0 full 5-fold and AEF.

### P1D: Delta Auxiliary Head

- Only after P1B is understood.
- Add optional head and config switch.
- Run sanity, quick eval, then full 5-fold if promising.

## Initial Hyperparameter Proposal

```yaml
data:
  supervised_sampling:
    enabled: true
    positive_boost: 4.0
    max_weight: 8.0
    task_weights:
      building_change: 1.5
      farm_change: 1.5
      rubbish: 1.5
      construction_joint: 1.0

training:
  supervised_change_weight: 0.06
  supervised_change_pos_weight: 1.5
  supervised_change_neg_weight: 0.5
  supervised_change_hard_negative_ratio: 0.05
  supervised_change_task_weights:
    building_change: 1.5
    farm_change: 1.5
    rubbish: 1.5
    construction_joint: 1.0
```

Delta auxiliary head should start only after P1B:

```yaml
model:
  delta_aux:
    enabled: true
    hidden_dim: 64
    tasks:
    - building_change
    - farm_change
    - rubbish
    - construction_joint

training:
  delta_aux_weight: 0.03
  delta_aux_loss: bce_tversky
```

## Acceptance Criteria

P1B is successful if it satisfies at least two of:

- Sparse-task macro F1_best improves over P0.
- Sparse-task macro `F1@val_thr` improves over P0.
- Fold standard deviation drops for `farm_change` or `rubbish`.
- Macro mIoU stays above P0's 0.1220 or improves.
- AUC does not drop by more than 0.02 macro.

P1D is successful only if it improves P1B full 5-fold; otherwise the delta head
should remain optional and disabled by default.

## Risks and Guardrails

- Oversampling positives can overfit sparse masks. Use a capped sampler and keep
  reconstruction losses active.
- Hard-negative mining can focus on label noise. Start with `top 5%`, not a large
  fraction.
- Delta auxiliary head can make embeddings too task-specific. Keep low weight
  and compare against P1B before adopting.
- All actual experiment YAML files must remain self-contained and must not use
  `_base_`.
