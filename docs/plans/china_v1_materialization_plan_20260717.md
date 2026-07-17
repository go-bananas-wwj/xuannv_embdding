# China V1: Data Materialization Plan

## Scope

This plan materializes the **60,500 provisional China V1 locations** for the
13-month archive from **2025-04 through 2026-04**. It does not start national
model training. Training draws six-month contiguous windows from this archive
only after the pilot and quality gates pass.

The candidate registry currently contains:

| Sampling tier | Patches | Purpose |
| --- | ---: | --- |
| Spatial backbone | 57,405 | Approximately one-percent, boundary-valid national coverage. |
| Spatial-stratified supplement | 2,595 | Fills spatial strata after the base selection. |
| Ocean-adjacent supplement | 500 | Adds coastal land/water interfaces; an independent ocean polygon is required, so land borders are excluded. |
| Total | 60,500 | Static candidates before monthly quality gating. |

The final eligible registry can be smaller: a candidate without enough valid
monthly observations is rejected or replaced. No cloudy scene is retained just
to keep a nominal patch count.

## Sources And Access

All network jobs run through `scripts/data/run_china_v1_direct.sh`, which clears
proxy variables. Direct connectivity was tested from this host.

| Data | Primary source | Backup | Role | Planned local form |
| --- | --- | --- | --- | --- |
| Sentinel-2 L2A | Microsoft Planetary Computer `sentinel-2-l2a` STAC + signed COG assets | Copernicus Data Space `sentinel-2-l2a` STAC | Main 10 m optical source; SCL masks clouds | Monthly quality-selected composites plus pixel masks |
| Sentinel-1 RTC | Planetary Computer `sentinel-1-rtc`, only with an approved Planetary Computer account | CDSE Sentinel-1 GRD only after an explicitly documented RTC-equivalent preprocessing path | Cloud-robust radar source | Monthly composites plus availability masks |
| Landsat C2 L2 | Planetary Computer `landsat-c2-l2` | USGS Landsat STAC | Independent optical source and temporal redundancy | Monthly quality-selected composites plus QA_PIXEL masks |
| WorldCover / DEM | Official or Planetary Computer static layers | None during pilot | Sampling/audit strata only; not fine downstream labels | Per-patch summaries, not full duplicated country mosaics |
| OSM | Geofabrik China `2025-01-01` PBF, SHA-256 locked | None | Broad masked weak semantics and rare-class strata | Raster labels only where coverage/confidence is valid |
| Ocean polygons | Natural Earth 10m Ocean, SHA-256 checked | None | Coastal supplement only | Static vector metadata |

Planetary Computer STAC is publicly available but anonymous asset requests can
be throttled. Its Sentinel-1 RTC collection additionally requires an account.
The pipeline therefore uses bounded concurrency, signed-asset renewal and
resumable shards. CDSE is a metadata/access fallback, not an unreviewed mixing
of source products.

References: [Planetary Computer data access](https://planetarycomputer.microsoft.com/docs),
[Planetary Computer STAC filtering](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/),
and [CDSE STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html).

## Storage Budget

Measured Haidian P10C quality-selected storage was 8.18 MB/patch for six
months across S2, S1 and Landsat. Scaling this measured rate to 13 months gives
about 17.7 MB/patch.

| Artifact | Estimate | Rule |
| --- | ---: | --- |
| S2 monthly composites | 235 GB | Keep selected composites only. |
| S1 monthly composites | 336 GB | Keep selected composites only. |
| Landsat monthly composites | 502 GB | Keep selected composites only. |
| Core sensor total | 1.07 TB | 60,500 patches x 13 months. |
| Pixel masks, manifests, OSM/static labels, QA | 70-120 GB | Compressed, shard-local. |
| Temporary working space | <=150 GB | Deleted after each verified shard. |
| Planned peak | 1.30-1.40 TB | No full-country NetCDF or duplicate raw tile cache. |

The data root is `/data2/xuannv_embedding/china_v1/`, which currently has more
than 3 TiB free. Materialization is therefore feasible with substantial room
for retries and later embeddings.

## Execution Table

| Stage | Work | Output | Parallelism | Expected duration | Gate to proceed |
| --- | --- | --- | ---: | ---: | --- |
| 0 | Freeze 60,500-point candidate registry, hashes and static source locks | Immutable registry manifest | CPU | Complete | All IDs unique; source SHA-256 files exist |
| 1 | Query/cache 13-month STAC metadata by UTM zone and month | Minimal STAC JSONL catalogs | 6 CPU workers, max 12 requests | 2-6 h | Catalog completeness and rate-limit audit |
| 2 | Create 2,000-patch pilot in four 500-patch shards | S2/S1/Landsat composites + masks | 6 CPU/I/O workers | 4-8 h | Alignment/cloud/missingness visual QA passes |
| 3 | Recalibrate actual transfer and processing rate from pilot | ETA report and storage report | CPU | <30 min | Budget remains under 1.40 TB; no source systematically fails |
| 4 | Materialize remaining 58,500 candidates in 117 resumable 500-patch shards | Quality-filtered source shards | 6 CPU/I/O workers, adaptive 6-12 remote requests | 30-60 h | Every shard hash/manifest/quality report passes |
| 5 | Build exact quality-eligible atlas and replacement registry | Final registry, replacements, coverage maps | CPU | 2-4 h | Every six-month window satisfies required-source gate |
| 6 | Review 2 random patches per source/month and registry coverage tables | QA sheets and signed readiness report | CPU | 2-4 h | Explicit approval before NPU training |

The full-run 30-60 h estimate is intentionally conservative. It assumes
approximately 1.5-2.2 TB of remote COG-window transfer, throttling/retry
overhead and raster reprojection. After Stage 2, ETA is replaced by the measured
pilot throughput:

`full ETA = pilot wall time x 60,500 / 2,000 x 1.3 retry factor`.

## Quality And Safety Rules

1. Never download a national full-scene NetCDF mosaic. Read signed COG windows
   into a 500-patch shard and delete transient files after verification.
2. Use S2 SCL, Landsat QA_PIXEL and source-specific valid masks before
   resampling; categorical/bitmask bands use nearest-neighbor resampling.
3. For each six-month training window, require S2, S1 and Landsat coverage per
   the frozen policy: at least four complete months per source and at least 14
   source-months in total. Missing masks fail closed; poor scenes are not kept
   as fallbacks.
4. S2/Landsat retain at most the two best observations per patch-month, using
   scene metadata as a cheap prefilter and pixel-level quality as the decision.
5. Preserve OSM blank regions as unknown. Weak semantic loss is applied only
   to its confidence mask; OSM is not a negative-label raster.
6. Log the executing Git commit and `xuannv_embedding.__file__` in every shard
   launcher, with `PYTHONPATH=/root/workspace/xuannv/src`, preventing imports
   from the older worktree.
7. Keep high-resolution optical/SAR out of China V1 until a nationwide,
   licensed, temporally documented source is separately approved.

## What Starts Only After Approval

Stages 0-1 are metadata preparation. Stage 2 is a contained 2,000-patch
download pilot. The 58,500-patch full materialization and all NPU training wait
for the pilot QA, measured ETA and explicit go-ahead.
