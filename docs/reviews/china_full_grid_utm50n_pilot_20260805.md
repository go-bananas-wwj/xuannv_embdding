# China Full 1,280 m Grid: UTM 50N Pilot Evidence

Date: 2026-08-06

## Scope

This review records the corrected production pilot written to
`/data/xuannv_embedding/outputs/china_full_1280m_grid_pilot_utm50_20260805`.
It uses the densified inventory configured at
`/data/xuannv_embedding/raw/china_v1/national_macrocell_inventory_densified_20260806.jsonl`.

The pilot covers all 8,825 inventory macrocells owned by `utm50n`; each macrocell is a
10 by 10 group of 1,280 m parent patches. It is a sizing and package-integrity pilot,
not a national count claim.

## Generation Measurement

The production command completed with exit code 0:

```bash
python scripts/data/build_china_full_grid.py \
  --config configs/national/china_full_1280m_grid_v1_20260805.json \
  --zones 50 \
  --output /data/xuannv_embedding/outputs/china_full_1280m_grid_pilot_utm50_20260805
```

| Measurement | Exact value |
| --- | ---: |
| Parent patches (`all`) | 844,345 |
| Sampled patches | 9,052 |
| Unsampled patches | 835,293 |
| Parent-patch partition check | 844,345 = 9,052 + 835,293 |
| GeoParquet batches per partition | 9 |
| Generation elapsed time | 4,462.654205230065 s (74 min 22.654205230065 s) |
| Generation peak RSS | 722,928 KiB (705.984375 MiB) |
| Parent-package bytes (all + sampled + unsampled) | 870,834,801 B |
| Parent-package bytes per `all` patch | 1,031.3731957908201 B |

The source measurements are `generation_metrics.json` and `generation_summary.json`
under the pilot output root.

## Canonical Strict Verification

After the corrected generation, the canonical full-output command was run without a
zone selector, as required by the production CLI:

```bash
python scripts/data/build_china_full_grid.py \
  --config configs/national/china_full_1280m_grid_v1_20260805.json \
  --output /data/xuannv_embedding/outputs/china_full_1280m_grid_pilot_utm50_20260805 \
  --verify-only
```

It exited 0 and wrote `china_full_grid_membership_audit.json` with `passed: true`.
The audit reports `all_count=844345`, `sampled_count=9052`, `unsampled_count=835293`,
`matched=9052`, `missing_sampled_count=0`, and an empty `missing` list. Every failure
counter is zero, including duplicate keys, sampled-flag and exact-partition mismatches,
child metadata/geometry mismatches, all three hash mismatch counters, stored bounds and
footprint-coordinate mismatches, invalid geometry, owner-zone mismatch, and overlap
violations. The maximum footprint-coordinate difference and maximum cross-zone overlap
fraction are both `0.0`.

## Independent Full Readback

An independent readback ran after the canonical audit and completed in
137.1940674900543 s. It used `pyarrow.parquet` for every GeoParquet part and
`pyogrio.read_dataframe` for every Shapefile part. It checked CRS metadata, every row's
`UTM_EPSG`, projected 1,280 m width/height within 0.001 m, area within 1 m2, and the
partition membership flag.

| Format and partition | Parts | Rows | Sampled true / false | Bad CRS parts | Bad EPSG | Bad dimensions | Bad area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoParquet `all` | 9 | 844,345 | 9,052 / 835,293 | 0 | 0 | 0 | 0 |
| GeoParquet `sampled` | 9 | 9,052 | 9,052 / 0 | 0 | 0 | 0 | 0 |
| GeoParquet `unsampled` | 9 | 835,293 | 0 / 0 | 0 | 0 | 0 | 0 |
| Shapefile `all` | 29 | 844,345 | 9,052 / 835,293 | 0 | 0 | 0 | 0 |
| Shapefile `sampled` | 28 | 9,052 | 9,052 / 0 | 0 | 0 | 0 | 0 |
| Shapefile `unsampled` | 29 | 835,293 | 0 / 0 | 0 | 0 | 0 | 0 |

All 113 read parts declare EPSG:4326. The `all` partitions intentionally contain both
membership values; the two child partitions contain only their named membership class.

## Storage and Compression

The byte measurements include all files below each partition directory, including
GeoParquet and every Shapefile sidecar. A fresh ZIP archive using DEFLATE level 9 was
created from those three partition directories, measured, and removed.

| Artifact | Bytes | MiB |
| --- | ---: | ---: |
| `all/utm50n` | 434,496,706 | 414.36835 |
| `sampled/utm50n` | 5,493,890 | 5.23938 |
| `unsampled/utm50n` | 430,844,205 | 410.88505 |
| All three partitions, raw | 870,834,801 | 830.49278 |
| ZIP archive | 297,425,842 | 283.64739 |

The ZIP/raw ratio is `0.3415410611271609`; the raw/ZIP compression ratio is
`2.9279056424424614:1`, a 65.8458938872839% reduction. The archive is a measurement
artifact only and is not retained in the pilot output directory.

## Measured Configuration Limits

`configs/national/china_full_1280m_grid_v1_20260805.json` retains the values exercised
by this pilot:

| Config key | Retained value | Evidence |
| --- | ---: | --- |
| `output.batch_size` | 100,000 | 844,345 records wrote as eight full batches plus one 44,345-record batch (9 parts per GeoParquet partition). |
| `output.shapefile_component_cap_bytes` | 1,800,000,000 | Required production safety cap; generator validates this exact value. |
| `output.shapefile_safe_fraction` | 0.95 | The enforced split threshold is 1,710,000,000 B per component; every generated Shapefile component remained below it. |

No configuration value was changed after this pilot because the existing production
values are the measured values and the generator rejects drift from the 1.8 GB cap or
the 95% safe fraction.

## Macrocell-Based National Extrapolation

The densified national inventory has exactly 60,482 macrocells across UTM zones 43N--53N;
the UTM50N pilot contains exactly 8,825. The scaling multiplier is therefore
`60,482 / 8,825 = 6.853484419263456...`. The values below are intentionally a
macrocell-count extrapolation, not a promise that other zones have UTM50N's coastline,
sampling, or Shapefile fragmentation profile.

| Measure | Pilot | Macrocell-scaled national estimate |
| --- | ---: | ---: |
| Parent patches | 844,345 | 5,786,705.301983003 (round up: 5,786,706) |
| Sampled patches | 9,052 | 62,037.740963173 (round up: 62,038) |
| Unsampled patches | 835,293 | 5,724,667.561019830 (round up: 5,724,668) |
| Generation time | 4,462.654205230065 s | 30,584.731064104792 s (8.495758628918 h) |
| All partition bytes | 434,496,706 B | 2,977,816,404.7922946 B |
| Sampled partition bytes | 5,493,890 B | 37,652,289.51614731 B |
| Unsampled partition bytes | 430,844,205 B | 2,952,784,046.0974503 B |
| Raw package bytes | 870,834,801 B | 5,968,252,740.405892 B |
| ZIP bytes | 297,425,842 B | 2,038,403,374.0333144 B |

## Concerns and Next Gate

1. The output count and storage extrapolations are arithmetic projections from one owner
   zone. The complete national run must report its actual zone counts and bytes rather
   than promote these values to a final atlas-size claim.
2. The full canonical geometry/membership audit is substantially more expensive than a
   count-only check at this scale. Run it as a dedicated, blocking post-write step for
   the nationwide package and retain its emitted audit JSON.
3. This pilot proves grid packaging only. It does not validate imagery availability,
   training-data quality, or downstream embedding performance.
