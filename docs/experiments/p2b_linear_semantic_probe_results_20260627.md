# P2B Linear Semantic Probe Results - 2026-06-27

## Round Hypothesis

P2A improved U-Net downstream results, but linear/MLP probes were mixed. P2B tested one focused hypothesis: if the training-only semantic probe is changed from a small MLP to a pure 1x1 linear probe, the embedding may become more linearly readable for building and road extraction.

This round changed only the semantic probe readout shape:

- `semantic_probe_hidden_dim: 0`
- Init checkpoint: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- Full config: `configs/v2_p2b_linear_semantic_probe_full_20260627.yaml`
- Best checkpoint: `/data/xuannv_embedding/outputs/v2_p2b_linear_semantic_probe_full_20260627/best.pt`
- Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p2b_linear_semantic_probe_full_20260627_best_p2b_linear_semantic_probe_full_best_20260627`
- Change benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2b_linear_semantic_probe_full_quick_20260627_214900`
- OSM benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2b_linear_semantic_probe_full_quick_20260627_214900`

## Training

Full training used NPU 0-5. The best checkpoint was the final checkpoint at epoch 99 with `val_loss=1.539060`.

The exported embedding cache is complete:

| Region | Patch Dirs | Monthly Embedding Maps |
|---|---:|---:|
| haidian | 320 | 1920 |
| harbin | 424 | 2544 |

## OSM Building and Road

Main comparison is against P2A, because P2A was the previous strongest semantic checkpoint. P1B is retained as the older baseline.

`pred+` means predicted positive pixel ratio at the tuned validation threshold, computed from `(tp + fp) / (tp + fp + tn + fn)`.

| Head | Task | P1B F1 | P2A F1 | P2B F1 | ΔP2B-P2A | P2B F1@0.5 | P2B F1@thr | P2B mIoU | P2B AUPRC | P2B pred+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | haidian_building_osm | 0.3057 | 0.2894 | 0.3056 | +0.0162 | 0.1590 | 0.3008 | 0.1770 | 0.1934 | 0.2130 |
| linear | haidian_road_osm | 0.3176 | 0.2953 | 0.3225 | +0.0272 | 0.2736 | 0.3187 | 0.1896 | 0.1937 | 0.6512 |
| linear | harbin_building_osm | 0.1373 | 0.1752 | 0.1360 | -0.0391 | 0.0348 | 0.1335 | 0.0715 | 0.0618 | 0.4051 |
| linear | harbin_road_osm | 0.1887 | 0.2254 | 0.2160 | -0.0095 | 0.1360 | 0.2113 | 0.1181 | 0.1173 | 0.3906 |
| mlp | haidian_building_osm | 0.2938 | 0.2897 | 0.2871 | -0.0026 | 0.1588 | 0.2855 | 0.1665 | 0.1876 | 0.2233 |
| mlp | haidian_road_osm | 0.3020 | 0.3024 | 0.3238 | +0.0214 | 0.2736 | 0.3197 | 0.1903 | 0.2331 | 0.3465 |
| mlp | harbin_building_osm | 0.2860 | 0.4044 | 0.2715 | -0.1329 | 0.2161 | 0.2706 | 0.1565 | 0.1889 | 0.0849 |
| mlp | harbin_road_osm | 0.3183 | 0.3648 | 0.3254 | -0.0394 | 0.2458 | 0.3243 | 0.1935 | 0.2297 | 0.1782 |
| unet | haidian_building_osm | 0.3590 | 0.4225 | 0.3916 | -0.0309 | 0.2218 | 0.3785 | 0.2334 | 0.3156 | 0.1249 |
| unet | haidian_road_osm | 0.3869 | 0.4491 | 0.4347 | -0.0144 | 0.2737 | 0.4339 | 0.2770 | 0.4275 | 0.2032 |
| unet | harbin_building_osm | 0.4283 | 0.6121 | 0.4802 | -0.1319 | 0.2710 | 0.4449 | 0.2861 | 0.4270 | 0.1107 |
| unet | harbin_road_osm | 0.3654 | 0.5469 | 0.4625 | -0.0844 | 0.1965 | 0.4615 | 0.2999 | 0.4454 | 0.0967 |

Takeaway: P2B improves Haidian road simple probes, but it loses too much on Harbin and also reduces U-Net performance. This is not a valid upgrade over P2A.

## Change Tasks

| Task | AEF F1 | P2A F1 | P2B F1 | ΔP2B-P2A | P2B F1@0.5 | P2B F1@thr | P2B mIoU | P2B AUPRC | P2B pred+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| construction | 0.2116 | 0.2978 | 0.2576 | -0.0402 | 0.0000 | 0.2573 | 0.1477 | 0.1719 | 0.0681 |
| building_change | 0.0245 | 0.3229 | 0.3081 | -0.0147 | 0.1785 | 0.1997 | 0.1109 | 0.1522 | 0.0076 |
| farm_change | 0.0147 | 0.1421 | 0.0131 | -0.1291 | 0.0088 | 0.0000 | 0.0000 | 0.0026 | 0.0003 |
| rubbish | 0.0365 | 0.5971 | 0.5677 | -0.0294 | 0.2423 | 0.4595 | 0.2983 | 0.6202 | 0.0018 |
| construction_joint | 0.3650 | 0.3371 | 0.3477 | +0.0106 | 0.1269 | 0.3331 | 0.1998 | 0.2978 | 0.0563 |

Takeaway: P2B still beats AEF on several tasks, but it regresses from P2A on 4 of 5 change tasks. `farm_change` nearly collapses, so P2B fails the rule that change detection must not clearly degrade.

## Visualization

Diagnostics:

- Linear: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2b_linear_semantic_probe_full_quick_20260627_214900/diagnostics_linear/index.md`
- MLP: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2b_linear_semantic_probe_full_quick_20260627_214900/diagnostics_mlp/index.md`
- U-Net: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2b_linear_semantic_probe_full_quick_20260627_214900/diagnostics_unet/index.md`

Visual inspection shows that P2B does not fix broad false positives. It is somewhat more conservative in Harbin, but this mostly removes true positives rather than cleanly fixing over-segmentation. The same stale title issue remains in the visualization sheets: the PCA panel says `P1B Emb PCA`, but the actual input embedding root is P2B.

## Decision

P2B is rejected as a new baseline.

Plain-language summary:

- What changed: the training-only semantic probe was forced to be linear.
- Why: we wanted the embedding to be easier for linear/MLP heads to decode.
- What happened: Haidian road simple probes improved a little, but Harbin building/road and most change tasks got worse.
- Visualization issue: red predictions are still too broad or too fragmented; the change did not make masks cleaner.
- Baseline decision: keep P2A as the current stronger semantic checkpoint, not P2B.

## Next Round Direction

Do not continue with pure linear semantic probe as the main path. The next round should go back to P2A as the base and target the actual visual failure mode:

1. Keep the MLP semantic probe from P2A.
2. Add hard-negative sampling or false-positive penalty for building/road semantic loss.
3. Add region-balanced semantic sampling so Haidian improvements do not trade off against Harbin.
4. Keep supervised change loss unchanged or slightly stronger, because P2B showed semantic tuning can damage change tasks.

One focused next hypothesis: P2C should add semantic hard-negative mining on top of P2A, not linear-only probing.

