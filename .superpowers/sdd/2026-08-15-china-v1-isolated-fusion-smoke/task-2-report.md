# Task 2 Report — Read-Only ZIP Selection and Quarterly Tensor Contract

## Scope delivered

- Added `MonthRef`, `PatchSelection`, and `RasterHeader` archive contracts.
- Added stable complete-patch intersection over `pc-s1` and `pc-s2`, years 2020/2021 and all 12 months, using only ZIP central-directory inspection. The implementation opens archives with `ZipFile`, never invokes `extract()` or `extractall()`.
- Captured the selected member CRC and constructed deterministic ZIP member paths.
- Added direct TIFF header inspection with `ZipFile.open` and `rasterio.io.MemoryFile`.
- Added the `PatchYearBatch` loader with quarter-major shapes, S2 scale, S1 VH/VV-to-VV/VH reordering, dB conversion, and per-pixel validity masks. Raster decode validates the 128×128 shape, expected sensor band count, CRS, and non-identity north-up transform.

## TDD evidence

### Archive RED

Command:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke/test_archive.py -q
```

Observed expected failure before implementation:

```text
ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.archive'
```

### Archive GREEN

The archive fixture suite passed after implementing the selector and header API:

```text
4 passed in 0.59s
```

### Data-contract RED

Command:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke/test_data_contract.py -q
```

Observed expected failure before implementation:

```text
ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.data'
```

### GREEN / final verification

Commands:

```bash
ruff check experiments/china_v1_fusion_smoke/archive.py \
  experiments/china_v1_fusion_smoke/data.py \
  tests/isolated_smoke/test_archive.py \
  tests/isolated_smoke/test_data_contract.py
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke -q
```

Output:

```text
All checks passed!
..............                                                           [100%]
14 passed in 18.56s
```

The Task 2 archive/data suites also passed together:

```text
......                                                                   [100%]
6 passed in 17.25s
```

## Real-header inspection evidence

`select_complete_patches(Path('/data2/china_xuannv_embedding/data'), count=4)` returned these stable complete selections, each with 48 references:

| Patch ID | Representative direct header |
| --- | --- |
| `parent_32643:310:3383` | `pc-s1`, 2020-01, 128×128, 2 bands, `float32`, `EPSG:32643` |
| `parent_32643:311:3390` | `pc-s1`, 2020-01, 128×128, 2 bands, `float32`, `EPSG:32643` |
| `parent_32643:312:3346` | `pc-s1`, 2020-01, 128×128, 2 bands, `float32`, `EPSG:32643` |
| `parent_32643:312:3406` | `pc-s1`, 2020-01, 128×128, 2 bands, `float32`, `EPSG:32643` |

The representative transforms are north-up 10 m affine transforms; for example the first is approximately `(10, 0, 396800, 0, -10, 4331520, 0, 0, 1)`.

## Source immutability evidence

For the first real selection, the inspection measured source ZIP inode, byte size, and nanosecond modification time for all 48 source archives immediately before and after `inspect_selection_headers`:

```text
archives_checked 48 headers 48 source_metadata_unchanged True
```

The test fixture also snapshots its entire ZIP tree before and after selection; this regression test passes. No source archive is extracted or written.

## Files

- `experiments/china_v1_fusion_smoke/archive.py`
- `experiments/china_v1_fusion_smoke/data.py`
- `tests/isolated_smoke/test_archive.py`
- `tests/isolated_smoke/test_data_contract.py`

## Commit and push

- Commit: `583aca3 feat: add read-only national patch-year smoke loader`
- Pushed: `origin/codex/china-v1-fusion-smoke`

## Self-review

- Selection uses exact sensor/year/month archive paths and lexically sorts the full intersection before slicing to `count`.
- The central-directory scan and CRC capture operate on `ZipInfo` only; TIFF data is read only by explicit header/decode APIs through `ZipFile.open`.
- Quarterly placement uses `quarter_index = (month - 1) // 3` and an in-quarter index `(month - 1) % 3`, yielding the required `[B, 4, 3, C, 128, 128]` contract.
- The source-specific preprocessing is covered by literal-valued TIFF fixtures: S2 scaling and S1 channel reversal/log conversion are asserted independently of the loader implementation.

## Fix Round 1

### Findings addressed

1. Direct-member parsing now rejects nested paths. CRC collection retains the exact discovered ZIP member filename instead of reconstructing it.
2. `select_complete_patches` now raises `ValueError("requested N complete patches, found M")` when the stable intersection is too small.
3. The loader now validates that every selection has exactly 48 unique `(sensor, year, month)` references matching the complete S1/S2 × 2020/2021 × 12-month contract before it reads any raster.
4. Each decoded raster now requires a north-up 10 m affine transform, and every S1/S2 month in a selection must share the same CRS and affine transform.
5. Validity combines the GDAL validity mask with finite checks across all bands. Non-finite values are zeroed before S2/S1 preprocessing and invalid pixels are zeroed again after preprocessing, so they cannot produce `-60 dB` placeholders or propagate NaN/Inf.

### Covering tests

- `test_selector_ignores_nested_tiff_members`
- `test_selector_rejects_requested_count_larger_than_complete_intersection`
- `test_loader_rejects_duplicate_missing_and_extra_month_references`
- `test_loader_rejects_nonuniform_or_invalid_patch_grid[shifted-transform]`
- `test_loader_rejects_nonuniform_or_invalid_patch_grid[different-crs]`
- `test_loader_rejects_nonuniform_or_invalid_patch_grid[rotated-transform]`
- `test_loader_rejects_nonuniform_or_invalid_patch_grid[non-10m-transform]`
- `test_loader_masks_and_sanitizes_nonfinite_pixels`

### RED

Command:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_archive.py \
  tests/isolated_smoke/test_data_contract.py -q
```

Output before the fixes:

```text
..FF....FFFFFF                                                           [100%]
8 failed, 6 passed in 20.59s
```

The failures were the intended reproductions: nested members entered the intersection, underfilled selection did not raise, duplicate/extra references were accepted, each grid mismatch was accepted, and a GDAL-valid NaN pixel remained marked valid.

### GREEN

Command:

```bash
ruff check experiments/china_v1_fusion_smoke/archive.py \
  experiments/china_v1_fusion_smoke/data.py \
  tests/isolated_smoke/test_archive.py \
  tests/isolated_smoke/test_data_contract.py
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_archive.py \
  tests/isolated_smoke/test_data_contract.py -q
```

Output:

```text
All checks passed!
..............                                                           [100%]
14 passed in 18.42s
```

Final isolated smoke verification:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke -q
```

```text
......................                                                   [100%]
22 passed in 18.21s
```

The real read-only source selection was rechecked after the parser change:

```text
[('parent_32643:310:3383', 48), ('parent_32643:311:3390', 48),
 ('parent_32643:312:3346', 48), ('parent_32643:312:3406', 48)]
```

### Fix Round 1 commit and self-review

- Commit: `04d2e4d fix: harden smoke archive and raster contracts`
- Pushed: `origin/codex/china-v1-fusion-smoke`
- Review: exact selected member names and CRCs now travel together; malformed reference cardinality is rejected before dictionary creation; grid equality is scoped to one patch selection across both sensors and both years; non-finite input has both a false validity mask and finite, zero-filled output.
