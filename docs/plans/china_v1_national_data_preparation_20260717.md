# Xuannv China V1: National Data Preparation

## Decision

China V1 will retain the P10C principles that mattered in Haidian: 1280 m
chips at 10 m / 128 x 128 output resolution, multi-source monthly observations,
pixel-valid reconstruction losses, quality filtering before training, structured
modality/month/spatial masking, and broad OSM semantics rather than downstream
manual labels. It will not copy the Haidian-only high-resolution sources until a
legally usable nationwide source is frozen.

The first archive covers **April 2025 through April 2026 inclusive (13 monthly
bins)**. Training samples are six-month contiguous windows drawn from that
archive; archiving a longer period must not be confused with sending thirteen
months into the P10C architecture at once. The base sample gives every
quality-eligible chip an expected one-percent inclusion probability. For a
10 x 10 macrocell with `n` eligible chips, the seeded-hash winner is accepted
with probability `n / 100`. This avoids treating a macrocell with only a few
cloud-free candidates as if it deserved the same sample count as a full one.

China has roughly 9.6 million km2 of land. At 1.6384 km2 per chip, one-percent
coverage is approximately 58,600 chips before coastline and eligibility filters.
The initial target is 80,000 to 100,000 chips after supplemental strata are
added.

## Why This Sampling Design

The one-percent backbone is aligned with the scale of AlphaEarth Foundations:
its training data cover about 1.1% of Earth land area, while performance still
improves as unique observations increase. AEF also uses valid per-pixel source
weights, which is essential for monthly cloudy observations.

Clay uses a comparable two-part strategy: a global sampling backbone plus
land-cover balancing. It selects locations from WorldCover class statistics and
explicitly notes that human-made features are smaller and unevenly distributed.
OLMoEarth and Prithvi-EO likewise use globally distributed multi-temporal
samples rather than a few large urban regions.

Sources:

- [AlphaEarth Foundations paper](https://arxiv.org/abs/2507.22291)
- [Clay data sampling](https://clay-foundation.github.io/model/release-notes/data_sampling.html)
- [OLMoEarth pretraining repository](https://github.com/allenai/olmoearth_pretrain)
- [Prithvi-EO-2.0 paper](https://arxiv.org/abs/2412.02732)

## Data Layers

| Layer | Role | China V1 rule |
|---|---|---|
| Sentinel-2 L2A | Main 10 m optical time series | Keep the best two quality-qualified observations per patch-month; use SCL and pixel masks in loss. |
| Sentinel-1 RTC | Cloud-robust SAR time series | Composite valid observations per month; retain availability masks; lower reconstruction weight than optical. |
| Landsat C2 L2 | Independent optical observation source | Keep quality-qualified observations and reconstruct with a lower weight than S2. |
| ESA WorldCover | Coarse static semantic balance | Use chip-level fractions for sampling and audit only in the first national pilot. |
| Copernicus DEM GLO-30 | Static terrain diversity / optional input | Use elevation, slope and aspect summaries for strata; do not confuse it with a monthly target. |
| OSM | Broad weak semantics and rare-feature sampling | Rasterize only from a checksum-verified snapshot no later than 2025-04-01; unknown is not background. |
| High-resolution optical/SAR | Fine-boundary target | Deliberately deferred until a nationwide licensed source, dates and coverage are documented. |

The public Planetary Computer STAC endpoint exposes Sentinel-2 L2A and Landsat
collections, but anonymous access can be throttled. It is suitable for atlas and
pilot creation, not an unbounded country-scale downloader. The implementation
should use signed STAC assets, bounded workers and resumable 5,000-chip shards.

The already interrupted `china-20260601` partial download is deliberately
quarantined and cannot enter training: an OSM edit from after the image period
would leak future information. The preparation pipeline will first verify an
archived China extract dated no later than 2025-04-01, then record its checksum,
coverage statistics and rasterization rules. OSM blanks remain unknown rather
than background.

## Sampling Policy

The checked-in policy is
`configs/national/china_v1_sampling_policy_20260717.json`. It starts with the
one-percent backbone, then adds quotas for rare WorldCover covers, sparse OSM
themes, regional diversity and observation-hard examples. Quotas are targets,
not duplicated samples: a wetland airport chip can satisfy several strata while
appearing only once in the registry.

Every candidate atlas record must carry:

- a stable `patch_id`, MGRS `grid_id`, tile-local EPSG, integer `grid_row` /
  `grid_col`, WGS84 bounds and a geometry hash;
- the declared number of physical candidates in its 10 x 10 macrocell, so an
  incomplete atlas cannot silently alter sampling probabilities;
- per-source/month availability and quality summary;
- WorldCover fractions, terrain / ecoregion / province summaries;
- OSM coverage quality plus broad OSM strata; and
- `eligible=true` only if it passes the frozen core-source gate.

`scripts/data/build_national_sampling_registry.py` consumes this atlas JSONL
and writes a selected JSONL plus a hash-linked report. It never downloads
imagery, so the sample can be inspected, versioned and frozen before the costly
materialization phase.

Example:

```bash
python scripts/data/build_national_sampling_registry.py \
  --atlas /data2/xuannv_embedding/china_v1/atlas/candidate_atlas.jsonl \
  --policy configs/national/china_v1_sampling_policy_20260717.json \
  --output /data2/xuannv_embedding/china_v1/registry/train_registry.jsonl \
  --report /data2/xuannv_embedding/china_v1/registry/train_registry_report.json
```

## Materialization Order

1. Freeze the China land AOI and MGRS tile inventory. Use MGRS tile-local UTM
   grids; a single national UTM CRS would distort or misalign chips across zones.
2. Build the metadata-only candidate atlas from WorldCover, DEM, province /
   ecoregion overlays, OSM snapshot and STAC scene metadata.
3. Run the registry builder, inspect coverage tables and maps, then freeze its
   input/policy hashes. Supplemental quotas are measured as final coverage in
   unique selected chips, capped at 90,000 total chips, and cannot concentrate
   unboundedly in one macrocell or regional group.
4. Materialize in resumable 5,000-chip shards directly from COG windows. Do
   not first download full-country NetCDF mosaics.
5. For every shard, run the existing S2/Landsat quality scorer, select the best
   observations per month, create valid-target masks, and audit source
   alignment on random chips.
6. Train a 2,000-chip pilot from the exact national pipeline before committing
   the full 80,000 to 100,000-chip acquisition.

## Storage and Safety Gate

The quality-filtered Haidian P10C core sources occupy about 8.2 MB per patch
for six months (S2 + S1 + Landsat), excluding high-resolution sources. A
thirteen-month archive is estimated at about 17.7 MB per patch: 80,000 chips
need about 1.4 TB for core outputs before masks, indexes and working files.
China V1 is therefore rooted at `/data2/xuannv_embedding/china_v1/`, which has
about 3.3 TB free, and uses compressed 5,000-chip shards without a duplicated
national raw cache. A nationwide high-resolution copy remains blocked pending
source licensing and storage approval.

## Gates Before Full Download

1. Freeze the national AOI boundary and treatment of offshore islands.
2. Verify the MGRS atlas has approximately uniform one-percent coverage per
   macrocell and no province/biome is accidentally absent.
3. Verify the pre-period OSM snapshot checksum and compute mapping-coverage
   diagnostics; blank OSM areas are ignored for weak supervision.
4. Verify imagery source licenses, rate limits and a resumable acquisition
   budget.
5. Run visual alignment, per-window cloud-mask QA and source-CRS audits on a
   2,000-chip pilot covering all supplement strata.
6. Decide whether a nationwide high-resolution source is legally available.
   If not, China V1 must be explicitly described as a core-sensor model rather
   than a direct like-for-like P10C high-resolution replica.
