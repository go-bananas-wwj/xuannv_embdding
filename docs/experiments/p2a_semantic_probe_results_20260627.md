# P2A Semantic Probe Results - 2026-06-27

## Run Summary

- Model: `v2_p2a_semantic_probe_full_20260627`
- Init checkpoint: `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- Best checkpoint: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- Training log: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/logs/train.log`
- Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p2a_semantic_probe_full_20260627_best_p2a_semantic_probe_full_best_20260627`
- Change benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2a_semantic_probe_full_quick_20260627_185700`
- OSM benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2a_semantic_probe_full_quick_20260627_185700`

Training completed on 6 NPUs. The best validation checkpoint was the final checkpoint at epoch 99, with `val_loss=1.706879`.

The exported embeddings cover both regions:

| Region | Patch Dirs | Monthly Embedding Maps |
|---|---:|---:|
| haidian | 320 | 1920 |
| harbin | 424 | 2544 |

## Change Tasks vs AEF

This is quick fold-0 evaluation for P2A, compared with the existing AEF 5-fold mean.

| Task | AEF F1 Mean | P2A F1 Fold0 | Delta | P2A AUC | P2A mIoU |
|---|---:|---:|---:|---:|---:|
| construction | 0.2116 | 0.2978 | +0.0862 | 0.8744 | 0.1593 |
| building_change | 0.0245 | 0.3229 | +0.2983 | 0.9579 | 0.1363 |
| farm_change | 0.0147 | 0.1421 | +0.1274 | 0.8961 | 0.0000 |
| rubbish | 0.0365 | 0.5971 | +0.5606 | 0.9989 | 0.2917 |
| construction_joint | 0.3650 | 0.3371 | -0.0279 | 0.9480 | 0.2021 |

Takeaway: P2A is clearly stronger than AEF on most quick change tasks, especially `building_change` and `rubbish`. The exception is `construction_joint`, where it is slightly below AEF on F1 in this fold.

## OSM Building and Road Tasks vs P1B

P2A was evaluated with linear, MLP, and U-Net downstream heads. The linear/MLP heads are the stricter test of whether the embedding itself is already semantically easy to decode.

| Head | Task | P1B F1 | P2A F1 | Delta | P2A mIoU |
|---|---|---:|---:|---:|---:|
| linear | haidian_building_osm | 0.3057 | 0.2894 | -0.0163 | 0.1667 |
| linear | haidian_road_osm | 0.3176 | 0.2953 | -0.0223 | 0.1715 |
| linear | harbin_building_osm | 0.1373 | 0.1752 | +0.0379 | 0.0952 |
| linear | harbin_road_osm | 0.1887 | 0.2254 | +0.0367 | 0.1266 |
| mlp | haidian_building_osm | 0.2938 | 0.2897 | -0.0041 | 0.1688 |
| mlp | haidian_road_osm | 0.3020 | 0.3024 | +0.0004 | 0.1741 |
| mlp | harbin_building_osm | 0.2860 | 0.4044 | +0.1184 | 0.2455 |
| mlp | harbin_road_osm | 0.3183 | 0.3648 | +0.0465 | 0.2219 |
| unet | haidian_building_osm | 0.3590 | 0.4225 | +0.0635 | 0.2611 |
| unet | haidian_road_osm | 0.3869 | 0.4491 | +0.0623 | 0.2886 |
| unet | harbin_building_osm | 0.4283 | 0.6121 | +0.1837 | 0.4393 |
| unet | harbin_road_osm | 0.3654 | 0.5469 | +0.1815 | 0.3731 |

Takeaway:

- U-Net improves all 4 OSM tasks over P1B, with large gains on Harbin.
- MLP improves Harbin clearly, but does not improve Haidian meaningfully.
- Linear probe improves Harbin but regresses slightly on Haidian.

This means P2A added useful semantic signal, but the signal is not uniformly linearly separable across regions yet.

## Visualization Check

Diagnostics were generated here:

- Linear: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2a_semantic_probe_full_quick_20260627_185700/diagnostics_linear/index.md`
- MLP: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2a_semantic_probe_full_quick_20260627_185700/diagnostics_mlp/index.md`
- U-Net: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2a_semantic_probe_full_quick_20260627_185700/diagnostics_unet/index.md`

Each head produced 24 diagnostic sheets. The inspected U-Net samples are not pure background false positives; the model is responding to real building/road structure. However, it still over-segments broad areas. For example, some building and road predictions become large red blocks even at tuned high thresholds. So P2A should not be treated as solved just because F1 improved.

Note: the OSM visualization title still says `P1B Emb PCA`; that is a stale display label in the visualization script, not the embedding source. The run used the P2A embedding root above.

## Decision

P2A is a good upgrade candidate over P1B for feature-rich downstream heads and for Harbin semantic transfer. It should not fully replace P1B as the final baseline yet, because the simple linear/MLP probes on Haidian did not clearly improve and the visual checks show over-prediction.

Recommended next round:

1. Keep P2A as the current strongest semantic checkpoint for reporting demos.
2. Run full 5-fold on OSM U-Net only if we need robust presentation numbers.
3. For the next model upgrade, add calibration and boundary-aware constraints:
   - stronger hard negatives from non-building/non-road high-confidence areas;
   - boundary or edge-aware semantic loss for building/road labels;
   - region-balanced semantic batches so Haidian does not regress;
   - explicit false-positive penalty on broad over-segmentation.

