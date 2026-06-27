# P2B Linear Semantic Probe Sanity - 2026-06-27

## Hypothesis

P2A improved U-Net downstream heads, but linear/MLP probes were mixed, especially on Haidian. One likely reason is that the training-only semantic probe was an MLP, so semantic information could remain nonlinearly encoded.

P2B changes only one main thing: set the training-only semantic probe to a pure linear 1x1 readout with `semantic_probe_hidden_dim: 0`. The goal is to make building/road semantics easier for downstream linear and MLP probes to read.

## Code and Config

- Code commit: `6bc1f97 feat: add linear semantic probe option for p2b`
- Sanity config: `configs/v2_p2b_linear_semantic_probe_sanity_20260627.yaml`
- Full config: `configs/v2_p2b_linear_semantic_probe_full_20260627.yaml`
- Init checkpoint: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`

The config is self-contained and does not use `_base_`.

## Sanity Run

- Output: `/data/xuannv_embedding/outputs/v2_p2b_linear_semantic_probe_sanity_20260627`
- Log: `/data/xuannv_embedding/outputs/v2_p2b_linear_semantic_probe_sanity_20260627/logs/train.log`
- Device: single NPU 0
- Epochs: 5

| Epoch | Train Loss | Val Loss |
|---:|---:|---:|
| 0 | 1.812003 | 1.547742 |
| 1 | 1.746580 | 1.602571 |
| 2 | 1.692466 | 1.532093 |
| 3 | 1.676034 | 1.511173 |
| 4 | 1.661273 | 1.527009 |

Sanity completed without NaN, OOM, or training crashes. Best checkpoint was saved at epoch 4.

## Decision

The code path is valid. Proceed to the 6-NPU full P2B training run using NPU 0-5.

