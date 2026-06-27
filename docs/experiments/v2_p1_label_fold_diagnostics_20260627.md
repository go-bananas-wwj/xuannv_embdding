# V2 P1 Label Fold Diagnostics - 2026-06-27

Data report: `/data/xuannv_embedding/experiments/v2_202512_202605/diagnostics/p1_label_fold_diagnostics_20260627_085632`

## Availability

| task | kind | masks | split | status |
|---|---|---:|---:|---|
| construction | binary | 62 | True | ok |
| building_change | binary | 573 | True | ok |
| farm_change | binary | 573 | True | ok |
| rubbish | binary | 382 | True | ok |
| construction_joint | binary | 253 | True | ok |
| haidian_worldcover | multiclass | 320 | False | ok |
| harbin_worldcover | multiclass | 424 | False | ok |

## Key Findings

- Existing sparse-change labels are complete for the current splits: no missing masks in train/val/test.
- Sparse Harbin tasks are extremely imbalanced. `farm_change` has only 2,604 positive pixels across 191 patches; `rubbish` has 2,470; `building_change` has 4,885.
- Some validation folds contain almost no positives, so threshold selection can be unstable before model quality is considered.
- `worldcover` exists for Haidian and Harbin and can support immediate land-cover segmentation and patch-level LULC classification probes.
- Static road extraction labels were not found in the processed label inventory. Static building extraction needs either a trusted footprint mask or a clearly marked weak built-up proxy.

## Lowest Validation Positive Counts

| task | fold | val positive patches | val positive pixels | val positive pixel ratio |
|---|---:|---:|---:|---:|
| farm_change | 0 | 1 | 11 | 0.00004476 |
| farm_change | 1 | 1 | 15 | 0.00006104 |
| rubbish | 2 | 2 | 30 | 0.00012207 |
| building_change | 1 | 3 | 111 | 0.00045166 |
| building_change | 2 | 3 | 119 | 0.00048421 |
| rubbish | 1 | 5 | 176 | 0.00071615 |
| rubbish | 0 | 4 | 198 | 0.00080566 |
| rubbish | 3 | 4 | 208 | 0.00084635 |
| farm_change | 3 | 3 | 252 | 0.00102539 |
| building_change | 0 | 4 | 285 | 0.00115967 |
| rubbish | 4 | 3 | 285 | 0.00115967 |
| building_change | 3 | 4 | 412 | 0.00167643 |

## WorldCover Class Distribution

| task | class id | pixels | ratio |
|---|---:|---:|---:|
| haidian_worldcover | 1 | 96324 | 0.018374 |
| haidian_worldcover | 2 | 227487 | 0.043393 |
| haidian_worldcover | 3 | 413 | 0.000079 |
| haidian_worldcover | 4 | 499247 | 0.095232 |
| haidian_worldcover | 5 | 3488678 | 0.665468 |
| haidian_worldcover | 6 | 3102 | 0.000592 |
| haidian_worldcover | 8 | 927189 | 0.176862 |
| harbin_worldcover | 1 | 935947 | 0.134730 |
| harbin_worldcover | 2 | 81050 | 0.011667 |
| harbin_worldcover | 3 | 625267 | 0.090008 |
| harbin_worldcover | 4 | 2574381 | 0.370584 |
| harbin_worldcover | 5 | 2208052 | 0.317851 |
| harbin_worldcover | 6 | 12908 | 0.001858 |
| harbin_worldcover | 7 | 200 | 0.000029 |
| harbin_worldcover | 8 | 509011 | 0.073273 |

## Implication For P1

P1B should not rely on ordinary random sampling. It should add positive-aware patch sampling and task-aware hard-negative loss, then evaluate with validation-selected thresholds. For the expanded capability suite, implement WorldCover segmentation/classification first because labels already exist; road and static building extraction should start with a label acquisition or weak-label generation step.
