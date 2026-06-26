# V2 Embedding Execution Status

Branch:

`codex/exp-v2-embedding-haidian-harbin-202512-202605`

## Implemented

- V2 manifest builder and validation metadata.
- Mixed Haidian/Harbin dataset support with region-specific statistics.
- Corrected configurable uniformity loss with warmup.
- Train-only source dropout that preserves reconstruction targets.
- Self-contained V2 embedding config:
  `configs/v2_embedding_haidian_harbin_202512_202605.yaml`
- Self-contained 6-NPU sanity config:
  `configs/v2_embedding_haidian_harbin_202512_202605_gated_sanity.yaml`
- Source-aware gated temporal fusion is enabled for the V2 run
  (`stp.temporal_fusion: gated_sum`).
- Bitemporal downstream dataset support for `[202512, 202605]` and
  `concat_diff`.
- ROC-AUC, validation threshold selection, and V1.0 acceptance comparison
  tooling.

## Data Outputs

- Manifest:
  `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/manifest_v2_202512_202605.json`
- Manifest metadata:
  `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/manifest_v2_202512_202605.meta.json`

## Verified

```bash
PYTHONPATH=/root/workspace/xuannv/src pytest tests/test_dataset.py tests/test_training.py tests/test_train_entry.py -q
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams pytest downstreams/tests -q
PYTHONPATH=/root/workspace/xuannv/src python - <<'PY'
from xuannv_embedding.config import Config
cfg = Config.from_yaml('configs/v2_embedding_haidian_harbin_202512_202605.yaml')
print(cfg.experiment.name)
print(cfg.data.manifest_path)
print(cfg.model.num_months, cfg.data.months)
print(cfg.training.uniformity_weight, cfg.training.uniformity_warmup_epochs)
PY
```

Latest verification results:

- Main training/data tests: `38 passed`
- Downstream tests: `54 passed`
- V2 config: no `_base_`; manifest points to `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/`; months are `2025-12` through `2026-05`.

6-NPU gated fusion sanity result on 2026-06-26:

- Command used NPU `0,1,2,3,4,5` with per-NPU batch size `4`.
- Correct environment form:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=/root/workspace/xuannv/src:$PYTHONPATH \
torchrun --nproc_per_node=6 scripts/train/train.py \
  --config configs/v2_embedding_haidian_harbin_202512_202605_gated_sanity.yaml
```

- Important environment note: do not replace `PYTHONPATH` with only the project
  `src` path on Ascend. CANN's `tbe` path must remain in `PYTHONPATH`; otherwise
  the first NPU convolution fails with `ModuleNotFoundError: No module named
  'tbe'`.
- Result: `epoch 0: train_loss=3.350532, val_loss=3.328531`.
- Checkpoints:
  `/data/xuannv_embedding/outputs/v2_embedding_202512_202605/gated_fusion_sanity_bs4/epoch_1.pt`
  and `best.pt`.
- Observed HBM during training was about `10.4 GB / 65.5 GB` per NPU; process
  memory was about `6.8-7.0 GB` per rank.
- NPU `as_strided` and non-finite-check warnings were emitted by `torch_npu`,
  but the 1-epoch sanity completed successfully.

Full 6-NPU gated fusion training result on 2026-06-26:

- Command used NPU `0,1,2,3,4,5` with per-NPU batch size `4`.
- Training was launched in tmux session `xuannv_v2_train`; the session exited
  after successful completion.
- Output directory:
  `/data/xuannv_embedding/outputs/v2_embedding_202512_202605/gated_fusion_bs4_uniformity_dropout`
- Validation trajectory:
  - epoch 9: `2.685731`
  - epoch 19: `2.522732`
  - epoch 29: `2.341984`
  - epoch 39: `2.171747`
  - epoch 49: `2.085322`
  - epoch 59: `2.034969`
  - epoch 69: `2.013439`
  - epoch 79: `2.005691`
  - epoch 89: `2.000825`
  - epoch 99: `2.001047`
- Best checkpoint: `best.pt` from epoch 89 with `val_loss=2.000825`.
- Final checkpoint: `epoch_100.pt` with `val_loss=2.001047`.
- Recommendation: use `best.pt` for embedding export because it is slightly
  better than the final epoch.
- NPU memory remained well within capacity; typical per-rank process memory was
  about `19.2 GB`, and NPU memory returned to idle after training.

Embedding export result on 2026-06-26:

- Checkpoint:
  `/data/xuannv_embedding/outputs/v2_embedding_202512_202605/gated_fusion_bs4_uniformity_dropout/best.pt`
- Output directory:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260626_v2_embedding_haidian_harbin_202512_202605_gated_fusion_best_acceptance`
- Exported patches:
  - `haidian`: 320 patches
  - `harbin`: 424 patches
- Exported embedding map files:
  - `haidian`: 1920 files (`320 patches x 6 months`)
  - `harbin`: 2544 files (`424 patches x 6 months`)
- Export size: about `105 GB`.
- Integrity spot check:
  - `202512_embedding_map.pt` and `202605_embedding_map.pt` shapes are
    `(64, 128, 128)`.
  - scene embedding shape is `(64,)`.
  - pixel-wise embedding norms averaged `1.0` for checked Haidian and Harbin
    samples.

Downstream 5-fold acceptance result on 2026-06-26:

- Frozen embedding run:
  `/data/xuannv_embedding/embeddings/v2_202512_202605/20260626_v2_embedding_haidian_harbin_202512_202605_gated_fusion_best_acceptance`
- Benchmark root:
  `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance`
- Logs:
  `/data/xuannv_embedding/logs/v2_202512_202605/downstream_20260626_gated_fusion_best_acceptance`
- Downstream config:
  `downstreams/configs/v2_acceptance_concat_diff.yaml`
- Input representation: months `[202512, 202605]`, temporal mode
  `concat_diff`, embedding channels `192`.
- All five downstream tasks completed and wrote `summary_5fold.json`.
- V1.0 comparison reports:
  - `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/comparison_vs_v1.0.json`
  - `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/20260626_gated_fusion_best_acceptance/comparison_vs_v1.0.md`

| Task | Region | AUC | F1_best | mIoU | Acceptance |
|---|---|---:|---:|---:|---|
| construction | haidian | 0.9280 | 0.4442 | 0.2256 | pass |
| construction_joint | haidian+harbin | 0.8638 | 0.3217 | 0.1441 | pass |
| building_change | harbin | 0.5717 | 0.0058 | 0.0017 | fail |
| farm_change | harbin | 0.7122 | 0.0076 | 0.0022 | fail |
| rubbish | harbin | 0.7335 | 0.0101 | 0.0010 | fail |

Macro acceptance:

- Overall pass: `False`.
- Macro AUC: `0.7618` vs threshold `0.8433` -> fail.
- Macro F1_best: `0.1579` vs threshold `0.1524` -> pass.
- Macro mIoU: `0.0749` vs threshold `0.0434` -> pass.

Interpretation:

- The exported gated-fusion embedding is usable for construction-related
  segmentation. Haidian construction has strong ranking quality
  (`AUC=0.9280`), and joint Haidian+Harbin construction remains above the
  acceptance threshold (`AUC=0.8638`) despite large fold variance.
- Construction thresholds are high: mean selected threshold is about `0.874`
  for Haidian construction and `0.777` for joint construction, while mean
  best-F1 thresholds are about `0.969` and `0.966`. This suggests the frozen
  embedding separates positives mainly at the high-confidence tail; downstream
  deployment should tune thresholds per task instead of using `0.5`.
- Harbin change-style tasks (`building_change`, `farm_change`, `rubbish`) do
  not yet produce usable masks. They have very low AP/F1/mIoU, with best epochs
  concentrated near the beginning, indicating the current frozen embedding plus
  shallow decoder is not learning stable rare-change segmentation.
- The non-random AUC on `farm_change` and `rubbish` (`0.71-0.73`) indicates
  weak ordering signal is present, but the class imbalance and label sparsity
  dominate thresholded segmentation quality.

Recommended next experiment:

1. Keep the gated-fusion embedding as the current construction baseline.
2. For Harbin change tasks, run task-specific diagnostics before another long
   embedding training job: inspect positive-pixel rates per fold, visualize
   prediction probability maps, and verify label-month semantics for the
   `202512 -> 202605` pair.
3. Try downstream-side ablations first because they are cheaper than retraining
   the embedding: per-task threshold calibration, lower/controlled positive
   weighting, focal/Tversky-style loss, and `single_month` or `diff_only`
   temporal inputs.
4. If change-task metrics remain near zero after downstream ablations, add a
   change-aware objective or harder temporal contrast during embedding training.

## Next Commands

Training on 6 Ascend NPUs:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=/root/workspace/xuannv/src:$PYTHONPATH \
torchrun --nproc_per_node=6 scripts/train/train.py \
  --config configs/v2_embedding_haidian_harbin_202512_202605.yaml
```

Export frozen embeddings after selecting a checkpoint:

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python downstreams/scripts/precompute_embeddings.py \
  --config configs/v2_embedding_haidian_harbin_202512_202605.yaml \
  --checkpoint /data/xuannv_embedding/outputs/v2_embedding_202512_202605/gated_fusion_bs4_uniformity_dropout/best.pt \
  --regions haidian harbin \
  --output-root /data/xuannv_embedding/embeddings/v2_202512_202605 \
  --suffix acceptance
```

Run one downstream task after embedding export:

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python downstreams/scripts/train_task.py \
  --task building_change \
  --config downstreams/configs/v2_acceptance_concat_diff.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/{run_id} \
  --label-root /data/xuannv_embedding/processed/harbin/labels/building_change \
  --region harbin \
  --output-root /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/{run_id}/building_change
```

Compare all five task summaries against V1.0:

```bash
python scripts/report/compare_v2_acceptance.py \
  --benchmark-root /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/{run_id} \
  --output /data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/{run_id}/comparison_vs_v1.0.json
```
