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
  --checkpoint /data/xuannv_embedding/outputs/v2_embedding_202512_202605/baseline_uniformity_dropout/best.pt \
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

## Not Yet Run

Embedding export and five-task 5-fold acceptance have not been run in this
implementation pass.
