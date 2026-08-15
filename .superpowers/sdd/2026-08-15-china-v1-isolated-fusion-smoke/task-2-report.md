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
