# Task 3 Report: Membership and Geometry Audits

## Status

Implemented Task 3 in the requested worktree and branch. No nationwide grid
was materialized during this task.

## Delivered

- Added `audit_sample_membership()` for exact sampled-key membership, missing
  keys, and duplicate atlas/registry keys.
- Added `audit_grid_package()` to scan `all`, `sampled`, and `unsampled`
  GeoParquet partitions in bounded Arrow batches and return the required JSON
  audit result.
- Added `read_sampled_registry_jsonl()` and `write_grid_package_audit()`.
- Added `build_china_full_grid.py --audit-only --sampled-registry <registry.jsonl>`.
  It writes `china_full_grid_membership_audit.json` by default and exits with
  status 2 when a blocking audit condition is found.
- Audit result includes exact all/sample/unsampled counts, missing sampled
  keys, duplicate parent-key counts, partition algebra, identity/footprint and
  sampled-registry footprint hash mismatches, maximum footprint coordinate
  difference, projected dimensions/area, owner-zone checks, and same-/cross-
  zone overlap results.

## Streaming Design

The audit never materializes nationwide polygons in memory. It scans each
GeoParquet file with `ParquetFile.iter_batches()`. Parent-key partition counts
and the WGS84 footprint spatial index are retained in a temporary on-disk
SQLite database with an RTree index; each candidate geometry is read only when
its bounding box intersects the current streamed geometry. The temporary
directory is removed on success or failure.

## TDD Evidence

1. Added the membership, altered-footprint, overlap, JSONL CLI, and
   recomputed-hash regression tests before their production behavior existed.
2. The first red run failed with the expected missing
   `audit_sample_membership` and `audit_grid_package` attributes.
3. After the first green implementation, a new shifted-footprint test failed
   because a recomputed internal hash could otherwise leave `passed=true`.
4. Added `footprint_coordinate_mismatch_count` as a blocking condition; the
   focused regression then passed.

## Verification

- `python -m pytest tests/test_china_full_grid.py -v` -> `22 passed`.
- `ruff check scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py` -> passed.
- `black --check scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py` -> passed.
- `git diff --check` -> passed.

## Commit

- `2075c2f test: audit national parent grid membership`

## Concerns

- A full nationwide audit was intentionally not run because the Task 3 scope
  did not materialize the 11-zone parent atlas. Production execution needs
  temporary disk capacity for the SQLite parent-key and footprint RTree index,
  in addition to the existing GeoParquet package.
- The current 62,000 candidate registry uses
  `canonical_wgs84_footprint_hash`; the audit accepts that field and the atlas
  `footprint_hash` alias, but a future final registry must retain one of them.
