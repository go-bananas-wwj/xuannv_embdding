# P2C Semantic Hard Negative Sanity - 2026-06-27

## Hypothesis

P2B showed that forcing a purely linear training probe is too blunt: it slightly helped Haidian road simple probes but hurt Harbin and change tasks. P2C returns to the P2A MLP semantic probe and adds a small hard-negative penalty for semantic background pixels that the probe most wants to classify as positive.

The goal is to reduce broad red false positives without destroying P2A's Harbin transfer.

## Code and Config

- Code commit: `a3dd09e feat: add semantic hard negative loss for p2c`
- Sanity config: `configs/v2_p2c_semantic_hardneg_sanity_20260627.yaml`
- Full config: `configs/v2_p2c_semantic_hardneg_full_20260627.yaml`
- Init checkpoint: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- Hard-negative ratio: `0.03`
- Hard-negative weight: `0.25`

The config is self-contained and does not use `_base_`.

## Sanity Run

- Output: `/data/xuannv_embedding/outputs/v2_p2c_semantic_hardneg_sanity_20260627`
- Log: `/data/xuannv_embedding/outputs/v2_p2c_semantic_hardneg_sanity_20260627/logs/train.log`
- Device: single NPU 0
- Epochs: 5

| Epoch | Train Loss | Val Loss |
|---:|---:|---:|
| 0 | 1.840038 | 1.553593 |
| 1 | 1.755356 | 1.645097 |
| 2 | 1.739458 | 1.576926 |
| 3 | 1.711701 | 1.525813 |
| 4 | 1.711993 | 1.536498 |

Sanity completed without NaN, OOM, or training crashes. Best checkpoint was saved at epoch 4.

## Decision

The code path is valid. Proceed to the 6-NPU full P2C training run using NPU 0-5.

