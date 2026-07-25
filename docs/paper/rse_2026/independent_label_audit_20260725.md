# Independent-Label Audit for the RSE Study

**Date:** 2026-07-25  
**Status:** evidence inventory; not a result-admission record.

## Decision

No label currently on disk supports a claim that the full XuannvEarth recipe transfers
independently to building-footprint or road extraction. The primary OSM building, road, water,
and construction readouts are useful **OSM-assisted** diagnostics only. The ESRI 2023 land-cover
labels are available over all 320 patches, but they overlap with the categorical WorldCover
supervision used by the full recipe and have a 2023 versus 2025--2026 time mismatch.

The registered fold-specific encoders and the P10C product must be kept separate:

- Registered encoders use their matching train-fold manifest, while validation, test, and buffer
  patches are excluded from upstream training. This establishes geographic holdout.
- The full registered recipe nevertheless includes a WorldCover categorical target and OSM
  semantic probes. A downstream task sharing those ontologies is not task-ontology-independent.
- P10C was trained over the full Haidian product region. It is transductive and cannot be used
  for the RSE spatial-generalisation table.

## Available Label Sources

| Source | Coverage | Suitable use | Critical limitation |
| --- | --- | --- | --- |
| ESRI 2023 LULC | 320/320 patches, 10 m | Annual land-cover diagnostics: water, trees, crops, bare, rangeland, built area | Same land-cover source family as the full recipe's WorldCover target; 2023 temporal mismatch; built area is not a building footprint |
| OSM building/road/water | 320/320 patches | OSM-assisted few-shot readout and ablations | Overlaps full-recipe semantic probe ontology |
| LabelMe construction | 62/320 patches, reference imagery dated 2026-04-30 | Small construction diagnostic | Overlaps OSM construction semantic probe; not yet double annotated or independently adjudicated |
| Dynamic World | no files in the local directory | none | Cannot be used until downloaded and quality controlled |

Exact local paths and mask metadata are maintained under
`/data/xuannv_embedding/processed/haidian/labels/`; large labels remain outside Git.

## Admissible Analyses

1. Use the OSM tasks only with the phrase **OSM-assisted spatially held-out readout**. Report
   their ontology overlap explicitly and do not claim generic semantic transfer from them alone.
2. Evaluate `no_osm_150` against ESRI water and land-cover masks as a reconstruction-only
   diagnostic. It did not train on OSM or WorldCover semantic targets, but ESRI's annual time
   mismatch and land-cover semantics remain limitations.
3. Compare the full recipe, WorldCover-only recipe, and no-OSM recipe on ESRI labels only as an
   ontology-alignment/weak-supervision analysis. It is not evidence that the full recipe learned
   an unseen task.
4. Keep P10C visualisations, retrieval, and ESRI diagnostics in the product case-study or
   Supplementary Material, not in the inductive main table.

## Required Before Independent Main Results

To support independent building, road, and water claims, freeze a new held-out annotation set
with the following minimum evidence:

- patch list selected before model comparison and disjoint from encoder train/validation/buffer
  geography for the relevant fold;
- imagery/date source, label ontology, rasterisation rule, annotator guide, and checksums;
- two independent annotations or documented adjudication and agreement statistics;
- an append-only test-access log and immutable release manifest;
- at least enough geographically distributed test patches to report block-level uncertainty.

Until that set exists, the evidence ledger remains `pending` for ontology-independent transfer.
