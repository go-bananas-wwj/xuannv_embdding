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

## Next Commands

Training on 6 Ascend NPUs:

```bash
PYTHONPATH=/root/workspace/xuannv/src \
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

Full 6-NPU training, embedding export, and five-task 5-fold acceptance have not
been run in this implementation pass.
