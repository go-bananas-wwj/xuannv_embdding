# Artifact Retention Policy

This project keeps the Git repository small and stores heavyweight artifacts on
the data disk.

## Git Repository

Keep in Git:

- Source code under `src/`, `scripts/`, and `downstreams/`
- Self-contained experiment configs under `configs/`
- Small documentation under `docs/`
- Small indexes and manifests that point to data-disk artifacts

Do not keep in Git:

- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- Raster data: `*.tif`, `*.tiff`
- Generated embeddings
- Generated reports, slides, PDFs, videos, and large images
- Temporary annotation folders, fonts, local caches, or worktree outputs

## Data Disk Layout

Use `/data/xuannv_embedding/` for data and generated artifacts:

```text
/data/xuannv_embedding/
  raw/                  # Original source data, treated as read-only.
  processed/            # Standard processed datasets and labels.
  outputs/              # Training runs and best checkpoints.
  embeddings/           # Exported embeddings kept only when actively needed.
  experiments/          # Evaluation outputs, metrics, and visualizations.
  cache/                # Re-downloadable caches.
  archive/              # Archived reports or repo-local artifacts.
  cleanup_YYYYMMDD/     # Cleanup inventories and audit logs.
```

## Checkpoints

For old runs, keep only the best checkpoint:

```text
best.pt
```

Delete periodic checkpoints such as:

```text
epoch_*.pt
checkpoint_epoch_*.pt
```

The best checkpoint is enough for downstream evaluation and reporting. Periodic
checkpoints should be kept only during an active training run.

## Embeddings

Generated embeddings can be deleted after their metrics and visualizations are
saved. They are usually reproducible from `best.pt` and the fixed export script.

Exception: keep small external baseline embeddings only when they are actively
used for comparison, such as:

```text
/data/xuannv_embedding/embeddings/aef_official_2025_annual
```

## Run Naming

Use date-prefixed names:

```text
YYYYMMDD_model_region_task_stage_shortdesc
```

Examples:

```text
20260703_p5h_haidian_embedding_best
20260703_aef_haidian_downstream_pixel_mlp
20260703_p5h_vs_aef_haidian_visual_report
```

Every important run should have a short manifest or summary that records:

- Config path
- Checkpoint path
- Embedding path, if any
- Evaluation path
- Visualization path
- Notes about whether artifacts are reproducible or must be retained
