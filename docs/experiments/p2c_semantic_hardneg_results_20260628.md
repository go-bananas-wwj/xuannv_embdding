# P2C Semantic Hard Negative Results - 2026-06-28

## Round Hypothesis

P2B failed because a pure linear training probe was too blunt. P2C returned to the P2A MLP semantic probe and added one focused change: semantic hard-negative loss on the highest-loss background pixels for building/road labels.

Goal: reduce broad false positives while keeping P2A's semantic transfer.

## Run Summary

- Code commit: `a3dd09e feat: add semantic hard negative loss for p2c`
- Sanity doc: `docs/experiments/p2c_semantic_hardneg_sanity_20260627.md`
- Full config: `configs/v2_p2c_semantic_hardneg_full_20260627.yaml`
- Init checkpoint: `/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- Best checkpoint: `/data/xuannv_embedding/outputs/v2_p2c_semantic_hardneg_full_20260627/best.pt`
- Embedding root: `/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p2c_semantic_hardneg_full_20260627_best_p2c_semantic_hardneg_full_best_20260627`
- Change benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2c_semantic_hardneg_full_quick_20260627_214900`
- OSM benchmark: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2c_semantic_hardneg_full_quick_20260627_214900`

Training completed on NPU 0-5. The final checkpoint at epoch 99 was best with `val_loss=1.536930`.

Embedding export completed:

| Region | Patch Dirs | Monthly Embedding Maps |
|---|---:|---:|
| haidian | 320 | 1920 |
| harbin | 424 | 2544 |

Note: the quick evaluation directory uses the timestamp `20260627_214900` because the reused eval script had a fixed timestamp. It is still a distinct P2C output directory and does not overwrite P2B.

## OSM Building and Road

`pred+` is predicted positive pixel ratio at validation threshold.

| Head | Task | P1B F1 | P2A F1 | P2B F1 | P2C F1 | ΔP2C-P2A | F1@0.5 | F1@thr | mIoU | AUPRC | pred+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | haidian_building_osm | 0.3057 | 0.2894 | 0.3056 | 0.3118 | +0.0224 | 0.1598 | 0.3048 | 0.1798 | 0.2021 | 0.1908 |
| linear | haidian_road_osm | 0.3176 | 0.2953 | 0.3225 | 0.3221 | +0.0268 | 0.2736 | 0.3199 | 0.1904 | 0.1926 | 0.6354 |
| linear | harbin_building_osm | 0.1373 | 0.1752 | 0.1360 | 0.1500 | -0.0252 | 0.0422 | 0.1497 | 0.0809 | 0.0702 | 0.3036 |
| linear | harbin_road_osm | 0.1887 | 0.2254 | 0.2160 | 0.2322 | +0.0068 | 0.1354 | 0.2293 | 0.1295 | 0.1420 | 0.3023 |
| mlp | haidian_building_osm | 0.2938 | 0.2897 | 0.2871 | 0.2901 | +0.0004 | 0.1588 | 0.2869 | 0.1674 | 0.1911 | 0.2094 |
| mlp | haidian_road_osm | 0.3020 | 0.3024 | 0.3238 | 0.3243 | +0.0219 | 0.2736 | 0.3218 | 0.1918 | 0.2327 | 0.3695 |
| mlp | harbin_building_osm | 0.2860 | 0.4044 | 0.2715 | 0.3792 | -0.0252 | 0.2326 | 0.3776 | 0.2327 | 0.3147 | 0.0584 |
| mlp | harbin_road_osm | 0.3183 | 0.3648 | 0.3254 | 0.3492 | -0.0156 | 0.2494 | 0.3473 | 0.2101 | 0.2749 | 0.1518 |
| unet | haidian_building_osm | 0.3590 | 0.4225 | 0.3916 | 0.4323 | +0.0098 | 0.2249 | 0.4244 | 0.2693 | 0.3931 | 0.1033 |
| unet | haidian_road_osm | 0.3869 | 0.4491 | 0.4347 | 0.4402 | -0.0089 | 0.2737 | 0.4377 | 0.2802 | 0.4334 | 0.1837 |
| unet | harbin_building_osm | 0.4283 | 0.6121 | 0.4802 | 0.6314 | +0.0193 | 0.3437 | 0.6313 | 0.4612 | 0.6530 | 0.0484 |
| unet | harbin_road_osm | 0.3654 | 0.5469 | 0.4625 | 0.5215 | -0.0255 | 0.2006 | 0.5177 | 0.3492 | 0.5036 | 0.0992 |

Takeaway: P2C is better than P2B and improves several P2A points, especially Haidian linear probes and building U-Net. But it still underperforms P2A on Harbin building MLP and Harbin road U-Net, so it is not a clean replacement.

## Change Tasks

| Task | AEF F1 | P2A F1 | P2B F1 | P2C F1 | ΔP2C-P2A | F1@0.5 | F1@thr | mIoU | AUPRC | pred+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| construction | 0.2116 | 0.2978 | 0.2576 | 0.2650 | -0.0328 | 0.0000 | 0.2643 | 0.1522 | 0.1813 | 0.0736 |
| building_change | 0.0245 | 0.3229 | 0.3081 | 0.3033 | -0.0195 | 0.1916 | 0.1242 | 0.0662 | 0.1571 | 0.0159 |
| farm_change | 0.0147 | 0.1421 | 0.0131 | 0.1599 | +0.0178 | 0.1470 | 0.0000 | 0.0000 | 0.0495 | 0.0004 |
| rubbish | 0.0365 | 0.5971 | 0.5677 | 0.5581 | -0.0390 | 0.2398 | 0.4408 | 0.2827 | 0.5911 | 0.0018 |
| construction_joint | 0.3650 | 0.3371 | 0.3477 | 0.3608 | +0.0237 | 0.1241 | 0.3550 | 0.2158 | 0.3216 | 0.0487 |

Takeaway: P2C fixes P2B's `farm_change` collapse and improves `construction_joint`, but it is still below P2A on construction, building_change, and rubbish. It passes AEF on most tasks but fails the rule that change detection should not clearly regress from the current best baseline.

## Visualization

Diagnostics:

- Linear: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2c_semantic_hardneg_full_quick_20260627_214900/diagnostics_linear/index.md`
- MLP: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2c_semantic_hardneg_full_quick_20260627_214900/diagnostics_mlp/index.md`
- U-Net: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2c_semantic_hardneg_full_quick_20260627_214900/diagnostics_unet/index.md`

Visual inspection shows the hard-negative direction reduced predicted positive ratio for several tasks, especially building U-Net. However, road predictions still show broad false-positive regions, and building masks still contain large red areas at tuned thresholds. The failure mode is improved but not solved.

## Decision

P2C is a useful partial improvement, but it is not promoted to the new baseline.

Plain-language summary:

- What changed: kept P2A's semantic probe, added a small penalty on the hardest background pixels.
- Why: P2A's main visual bug was too much red false positive area.
- Better: Haidian simple probes improved; building U-Net improved; predicted positive ratios are lower.
- Worse: several change tasks still regress from P2A; Harbin simple building/road remains below P2A.
- Decision: keep P2A as the current baseline; keep P2C as evidence that hard-negative mining is useful but needs balancing.

## Next Round Direction

P2D should keep the P2C hard-negative idea but protect change detection and Harbin transfer:

1. Start from P2A again.
2. Use semantic hard-negative loss, but lower its weight or warm it up later.
3. Add region-balanced semantic loss weights so Harbin does not lose simple-probe performance.
4. Increase supervised change weight slightly or delay semantic hard negatives until after the change loss stabilizes.

One focused next hypothesis: delayed hard-negative warmup, not stronger hard-negative pressure.

