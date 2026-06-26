# V2 Downstream Speed And Label QA

Date: 2026-06-26

## Speed Fixes

Full downstream acceptance was slow because each task ran 5 folds with up to
100 epochs, validation every epoch, `num_workers: 0`, and GeoTIFF probability
export after every fold.

Implemented quick-evaluation support:

- `downstreams/scripts/train_task.py`
  - `--skip-predictions` skips test probability GeoTIFF export.
  - `--eval-every N` validates every N epochs instead of every epoch.
  - DataLoaders now support `persistent_workers` and `prefetch_factor` when
    `num_workers > 0`.
  - `best_epoch` is tracked explicitly so sparse validation keeps correct
    metadata.
- `downstreams/configs/v2_acceptance_quick_concat_diff.yaml`
  - Self-contained quick config.
  - 30 epochs, patience 6, batch size 16, 4 DataLoader workers, validation
    every 2 epochs, and prediction export disabled.

Recommended workflow:

1. Use the quick config for model/debug iterations and label QA.
2. Run only one fold first with `--fold 0`.
3. Run the full 5-fold `v2_acceptance_concat_diff.yaml` only after quick
   metrics and labels look plausible.

Example:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams:$PYTHONPATH \
python downstreams/scripts/train_task.py \
  --task building_change \
  --config downstreams/configs/v2_acceptance_quick_concat_diff.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/20260626_v2_embedding_haidian_harbin_202512_202605_gated_fusion_best_acceptance \
  --label-root /data/xuannv_embedding/processed/harbin/labels/building_change \
  --region harbin \
  --output-root /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/quick_debug/building_change \
  --fold 0 \
  --skip-predictions
```

Smoke result:

- Command above was run for `construction` fold 0 on NPU 0.
- Runtime: about 39 seconds (`16:07:04` to `16:07:43`).
- Early stopped at epoch 17; best epoch was 5.
- Test metrics: `AUC=0.9143`, `F1_best=0.3931`, `mIoU=0.0912`.
- No prediction GeoTIFF directory was written, confirming
  `--skip-predictions` works.
- Output:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/quick_smoke/construction/summary_5fold.json`

## Label Visualization

Added:

- `scripts/qa/visualize_downstream_labels.py`

The script selects positive masks, overlays them on high-resolution optical
and Sentinel-2 imagery, and writes both PNG figures and metadata. It supports
region-prefixed mask ids such as `haidian_patch_000198` and
`harbin_patch_000217`.

Generated QA outputs:

- `/data/xuannv_embedding/qa/downstream_label_visuals_20260626/`
- Contact sheet:
  `/data/xuannv_embedding/qa/downstream_label_visuals_20260626/contact_sheet.png`

## QA Findings

| Task | Mask files | Positive files | Patch groups | Mask CRS | Identity no-CRS masks | Multi-month identical pairs |
|---|---:|---:|---:|---|---:|---:|
| construction | 62 | 51 | 62 | EPSG:32650 | 0 | 0/0 |
| building_change | 573 | 131 | 191 | None | 573 | 551/573 |
| farm_change | 573 | 98 | 191 | None | 573 | 563/573 |
| rubbish | 382 | 111 | 191 | None | 382 | 186/191 |
| construction_joint | 253 | 134 | 253 | mixed | 191 | 0/0 |

Interpretation:

- Haidian `construction` masks carry real CRS/transform and visually align
  with source imagery.
- Harbin downstream masks carry no CRS and use an identity pixel transform
  (`0..128` bounds). Overlay QA can still check pixel-grid alignment by patch
  id, but the mask files themselves cannot prove geospatial alignment.
- Harbin change-task labels are highly suspicious for temporal evaluation:
  most same-patch multi-month masks are exactly identical. This makes
  `202512 -> 202605` change labels ambiguous and likely explains weak
  `building_change`, `farm_change`, and `rubbish` downstream metrics.
- The representative overlays do not show an obvious global flip or scale
  mismatch, but the lack of georeferencing and repeated monthly masks must be
  fixed before treating change-task metrics as a model-quality signal.

## Recommended Fix

1. Regenerate Harbin task masks with the same CRS, transform, and bounds as the
   corresponding 128x128 S2 patch.
2. Store single-event labels separately from month-specific labels. If a mask
   is valid for a whole patch rather than a specific month, do not duplicate it
   as separate monthly change targets.
3. For change tasks, define explicit source and target months in the label
   metadata, then train/evaluate against the matching embedding pair.
4. Re-run the quick downstream config first; only rerun full 5-fold acceptance
   after label QA passes.
