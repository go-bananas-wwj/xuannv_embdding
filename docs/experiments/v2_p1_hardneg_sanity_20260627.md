# V2 P1 Hard-Negative Sanity - 2026-06-27

Config: `configs/v2_p1_sparse_sampler_hardneg_sanity_20260627_091500.yaml`

Output: `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_sanity_20260627_091500`

## Result

The 6-NPU sanity run completed successfully after fixing the launch environment.

- Devices: `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5`
- Patches: 96
- Epochs: 2
- Supervised sampler: enabled
- Sampler weights: min `1.000`, max `5.000`, mean `1.708`
- Positive-weighted samples: `17/96`
- Epoch 0: train_loss `3.378740`, val_loss `3.380136`
- Epoch 1: train_loss `3.383979`, val_loss `3.377903`
- Best checkpoint: `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_sanity_20260627_091500/best.pt`

## Notes

The first launch failed because the tmux command overwrote `PYTHONPATH` and
removed CANN/TBE paths, causing `ModuleNotFoundError: No module named 'tbe'`.
The successful rerun sources `/usr/local/Ascend/cann-9.0.0/set_env.sh` and
prepends the repo `src` path to the existing `PYTHONPATH` instead of replacing it.

This validates the P1B training path before starting the full run.
