# V2 Embedding Experiment: Haidian + Harbin 202512-202605

## Goal

Train a V2 embedding model on Haidian and Harbin New Area imagery from
2025-12 through 2026-05, then validate frozen embeddings against the V1.0
downstream benchmark. The experiment must preserve the 128x128 embedding map
resolution and keep V1.0's change-task strengths.

## Branch And Commit Discipline

- Branch: `codex/exp-v2-embedding-haidian-harbin-202512-202605`.
- Push every independent code, config, script, or documentation step.
- Stage files explicitly. Do not use `git add .`.
- Do not stage existing untracked directories such as `.superpowers/`, `fonts/`,
  or `reports/`.
- Data, checkpoints, embeddings, and large logs stay under
  `/data/xuannv_embedding/` or ignored output directories.

## Canonical Paths

- Runnable config:
  `/root/workspace/xuannv/configs/v2_embedding_haidian_harbin_202512_202605.yaml`
- V2 manifests and metadata:
  `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/`
- Training outputs:
  `/data/xuannv_embedding/outputs/v2_embedding_202512_202605/{run_id}/`
- Exported embeddings:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/{run_id}/`
- Downstream benchmark reports:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/{run_id}/`

`run_id` should include timestamp, config name, and short git SHA.

## Data Rules

- Allowed months: `202512`, `202601`, `202602`, `202603`, `202604`, `202605`.
- Regions: `haidian`, `harbin`.
- Preferred low-resolution source names:
  `s2`, `s1`, `landsat`, `worldcover`.
- High-resolution source names remain region-specific because channel counts
  and native resolutions must be validated per region:
  `highres_optical_haidian` has 3 channels at 427x427,
  `highres_optical_harbin` has 3 channels at 2560x2560, and
  `highres_sar_haidian` has 1 channel.
- Missing high-resolution observations are represented by availability masks and
  recorded in manifest metadata. Do not synthesize missing months.
- Before training, run a dry-run that reports per-region/per-source month
  coverage, missing files, tensor shapes, and empty modality rates.

## Model And Loss Defaults

- Keep the current AEFModel/STP/vMF backbone and 128x128 embedding map output.
- Do not introduce `einops`.
- Primary objective: multimodal reconstruction plus WorldCover classification.
- Corrected uniformity loss is configurable and warmed from `0.001` to `0.01`.
- Modality dropout starts light: `0.1-0.2`.
- Temporal dropout is disabled by default or kept at `0.05` only for ablations.
- Same-month noisy/complete consistency is an ablation, not a default main loss.
- AEF pixel distillation is disabled by default and used only as a negative
  control.
- Weak relation distillation and temporal change consistency are later
  mask-gated ablations.

## Downstream Acceptance

Run the same five V1.0 tasks with frozen V2 embeddings:

- `construction`
- `building_change`
- `farm_change`
- `rubbish`
- `construction_joint`

The benchmark must use fixed folds, fixed months `[202512, 202605]`, paired
V1.0/V2 comparison, validation-only threshold selection, and report mean/std
over all folds.

Passing thresholds:

- Macro average must exceed V1.0:
  - `avg_auc_roc > 0.8433`
  - `avg_f1_best > 0.1524`
  - `avg_miou > 0.0434`
- Protected tasks must not regress:
  - `building_change`: AUC `>= 0.8828`, F1_best `>= 0.1163`, mIoU `>= 0.0327`
  - `farm_change`: AUC `>= 0.8876`, F1_best `>= 0.1208`, mIoU `>= 0.0385`
  - `rubbish`: AUC `>= 0.8876`, F1_best `>= 0.1208`, mIoU `>= 0.0384`
  - `construction_joint`: AUC `>= 0.8123`, F1_best `>= 0.1904`,
    mIoU `>= 0.0799`

## Required Monitoring

- Embedding norm mean/std and vMF stability.
- Batch pairwise cosine statistics, effective rank, and PCA variance.
- Per-source reconstruction losses.
- Region-separated train/validation metrics.
- Embedding delta between `202512` and `202605`, separated by change and
  non-change masks when labels are available.
