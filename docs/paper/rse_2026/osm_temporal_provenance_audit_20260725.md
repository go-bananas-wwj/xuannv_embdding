# OSM Temporal-Provenance Audit for the RSE Study

**Date:** 2026-07-25  
**Status:** evidence inventory; not a result-admission record.

## Decision

The OSM-derived training and downstream masks are re-rasterisable from locally retained caches
and recorded label-generation rules, but they are **not** tied to a versioned historical OSM
snapshot. The retained material cannot reproduce the same upstream OSM query response. These
masks must therefore be described only as weak auxiliary semantics and **OSM-assisted spatially
held-out readouts**. They are not temporally contemporaneous ground truth for the December
2025--May 2026 imagery, nor evidence of independent semantic transfer or monthly change.

## Evidence Located

| Use | Local cache evidence | Local file time (UTC) | Limitation |
| --- | --- | --- | --- |
| P10C OSM semantic targets | `/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/` | 2026-06-29 | No historical snapshot identity retained |
| OSM building downstream mask | `/data/xuannv_embedding/processed/haidian/labels/building_osm/` | 2026-06-27 | No query response hash or historical OSM version retained |
| OSM road downstream mask | `/data/xuannv_embedding/processed/haidian/labels/road_osm/` | 2026-06-27 | No query response hash or historical OSM version retained |
| OSM water downstream mask | `/data/xuannv_embedding/processed/haidian/labels/osm_water/` | 2026-06-29 | Derived from the same 2026-06-29 thematic cache |

The stated local file times only show that the retained artifacts had been written by those
times; they do not identify the online query time or an OSM feature-state date. The P10C
configuration explicitly uses OSM semantic targets; see
`configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml`.
The generators call online `osmnx.features_from_polygon(...)` directly in
`scripts/data/build_osm_weak_semantic_labels.py` and `scripts/data/build_osm_labels.py`.
Neither generator accepts a historical OSM date, replication sequence, fixed PBF/OSH file, or
content-addressed query response. Local file timestamps establish only the last write time of
the currently retained files; they do not establish the feature state at image acquisition time.

## Required Manuscript Wording

> OpenStreetMap-derived labels were used as weak auxiliary semantics and as OSM-assisted
> spatially held-out readouts. The locally retained OSM caches do not encode a versioned
> historical snapshot aligned with the 2025--2026 imagery; consequently, these evaluations are
> not interpreted as temporally contemporaneous ground-truth validation or independent
> semantic-transfer evidence.

## Requirement for Temporal Claim Admission

Any future temporally aligned OSM analysis must retain a downloadable historical `.osh.pbf` or
dated PBF snapshot, source URL, snapshot date, download UTC, SHA-256, AOI/query rule, and
rasterisation-version hash. Those fields must be bound into both the encoder and downstream
result registries before evaluation.
