# Task 2 Report: Streaming National Grid Writers

## Scope

Implemented Task 2 only in the requested worktree and branch. No nationwide grid was generated.

## Delivered

- Added `ZoneWriteSummary`, `write_zone_batch()`, and `write_zone_records()` to
  `scripts/data/china_full_grid.py`.
- `write_zone_records()` consumes an iterable with a bounded in-memory batch (default 100,000)
  and enforces `all_count == sampled_count + unsampled_count` after streaming completes.
- Every non-empty batch produces WGS84 GeoParquet parts under
  `<output-root>/{all,sampled,unsampled}/<grid-id>/`. GeoParquet keeps the canonical record
  fields, complete version/hash fields, an explicit `sampled` flag, and WGS84 parent-cell
  polygon geometry.
- Per-zone Shapefiles are written under the same partitions with exactly the compact fields
  `PATCH_ID`, `UTM_EPSG`, `GRID_COL`, `GRID_ROW`, `MACRO_ID`, and `SAMPLED` plus geometry.
- Shapefile writes stage a batch first and reject it before publishing when the projected
  `.shp` + `.shx` + `.dbf` total would exceed 1,800,000,000 bytes. A rejected first batch
  leaves no Shapefile behind.
- Added `scripts/data/build_china_full_grid.py`. The CLI reads a WGS84-compatible boundary,
  Task 1 macrocell JSON, sampled parent-key JSON, and writes one requested `grid_id` without
  materializing a nationwide record list.

## TDD Evidence

1. Added `test_writer_partitions_all_sampled_and_unsampled` before its implementation.
2. Ran the required focused test and observed the expected red failure:
   `AttributeError: module 'china_full_grid' has no attribute 'write_zone_records'`.
3. Implemented the bounded writer, then exercised the green path.
4. Added read-back coverage for GeoParquet CRS, WGS84 polygon geometry, complete metadata
   columns, compact Shapefile fields, sampled/unsampled disjointness, and partition algebra.
5. Added regression tests for the 1.8 GB Shapefile cap and for the CLI using only synthetic
   local inputs.

## Verification

- `python -m pytest tests/test_china_full_grid.py -v` -> 12 passed.
- `ruff check scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py` -> passed.
- `black --check scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py` -> passed.
- `git diff --check` -> passed during self-review.

## Self-Review

- The implementation accepts one `grid_id` per streaming call and raises on mixed zones,
  keeping each zone's Shapefile isolated.
- The batch loop retains no prior record batches; only the current bounded batch and summary
  are retained.
- The CLI is deliberately zone-scoped. Running every zone remains a separate orchestration
  step and was intentionally not performed for this task.
- The writer depends on the repository's declared GeoPandas/pyarrow vector stack. Production
  execution should retain the same environment used by the verification command.
