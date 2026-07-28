# Registered downstream results (preliminary)

## Status

This document records the completed, sealed computational outputs for the RSE draft. It is not
paper-admissible evidence: the required external release-admission record is still absent. Labels
overlap the OSM weak-semantic channel, and the AEF comparison is annual 2025 versus monthly 2026-04;
neither result supports independent-transfer or information-matched claims.

## Protocol completion

All 90 Xuannv V5 probe cells completed: building, road, and water; 5/10 support patches; five
spatial folds; and seeds 42/43/44. The same 90-cell contextual AEF protocol completed. The paired
comparison uses 10,000 hierarchical resamples over fold, complete 2x2 geographic cluster, and
support-schedule seed.

## Xuannv V5 aggregation

Values are arithmetic means of per-seed fold means (15 fold-seed observations per condition).

| Task | Support patches | F1 | AP | ROC-AUC | mIoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Building | 5 | 0.239 | 0.189 | 0.748 | 0.140 | 0.161 | 0.562 |
| Building | 10 | 0.267 | 0.213 | 0.786 | 0.159 | 0.185 | 0.523 |
| Road | 5 | 0.286 | 0.250 | 0.627 | 0.170 | 0.235 | 0.458 |
| Road | 10 | 0.295 | 0.266 | 0.647 | 0.176 | 0.265 | 0.377 |
| Water | 5 | 0.176 | 0.188 | 0.718 | 0.100 | 0.157 | 0.440 |
| Water | 10 | 0.213 | 0.215 | 0.745 | 0.126 | 0.178 | 0.427 |

## Contextual AEF comparison

The following is a time-inequivalent contextual readout, not an AEF superiority or label-efficiency
claim. The final column is the hierarchical geographic-cluster bootstrap F1 point estimate and its
95% interval; it is intentionally distinct from the descriptive difference of per-seed fold means.

| Task | Support patches | Xuannv F1 | AEF F1 | Difference (95% CI) |
|---|---:|---:|---:|---:|
| Building | 5 | 0.239 | 0.288 | -0.057 [-0.094, -0.013] |
| Building | 10 | 0.267 | 0.330 | -0.056 [-0.095, -0.014] |
| Road | 5 | 0.286 | 0.420 | -0.126 [-0.155, -0.103] |
| Road | 10 | 0.295 | 0.450 | -0.153 [-0.192, -0.118] |
| Water | 5 | 0.176 | 0.550 | -0.400 [-0.535, -0.214] |
| Water | 10 | 0.213 | 0.597 | -0.438 [-0.528, -0.293] |

## Source artifacts

- Xuannv aggregation: `/data/xuannv_embedding/experiments/paper_registered_v5_20260726/reports/xuannv_v5_full150_preliminary_aggregate_20260728.json`
- AEF comparison and bootstrap: `/data/xuannv_embedding/experiments/paper_registered_v5_20260726/reports/aef_annual2025_vs_xuannv_v5_contextual_preliminary_20260728.json`
