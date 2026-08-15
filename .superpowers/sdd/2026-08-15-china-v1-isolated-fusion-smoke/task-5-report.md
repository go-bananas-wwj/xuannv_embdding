# Task 5 Report: Auditable Zarr Export, Checkpoint Reload, and SUCCESS Sealing

## Completed scope

- Added atomic Zarr v2 exporter for `base`, `base_aef`, `base_highres`, and `full`.
  It writes only `embedding.zarr.partial`, verifies the reopened contract, then atomically
  renames it to `embedding.zarr`.
- Enforced the exact arrays `embedding float16 [4,8,64,128,128]`, `valid bool
  [4,8,1,128,128]`, four `patch_id` values, and eight `period` values. Embeddings are
  chunked `(1,1,64,128,128)` with Blosc Zstd. Every Zarr has `synthetic=true`,
  `allowed_use=smoke_test_only`, and both formal-use attributes set to `false`.
- Added a checkpoint writer/reloader that finds a sentinel sandbox for writes, stores
  SHA-256-addressed provenance, and rejects any metadata mismatch (Git commit, config
  hash, selected-patch manifest hash, seed, model class, or synthetic flag).
- Added SUCCESS sealing that requires all four sealed Zarr groups, five fixed evidence
  files, caller-required evidence paths inside the sandbox, no `.partial` files, and
  false formal-use attrs. It writes `SUCCESS` last with a deterministic combined SHA-256
  and UTC timestamp.

## TDD and verification evidence

- RED: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke/test_export.py -q` failed at collection with
  `ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.export'`.
- RED (checkpoint path guard): the new sentinel-sandbox test failed because checkpoint
  saving initially accepted an outside path.
- GREEN: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke/test_export.py -q` → `11 passed in 8.67s`.
- Regression: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke -q -m 'not npu'` → `60 passed in 22.26s`.
- Static checks: `python -m ruff check experiments/china_v1_fusion_smoke
  tests/isolated_smoke` and `python -m black --check
  experiments/china_v1_fusion_smoke/export.py tests/isolated_smoke/test_export.py` both
  passed. `git diff --check` also passed before commit.

## Files

- `experiments/china_v1_fusion_smoke/export.py`
- `tests/isolated_smoke/test_export.py`

## Commit and push

- Commit: `c066808 feat: add auditable smoke export and success sealing`
- Pushed: `origin/codex/china-v1-fusion-smoke`

## Self-review

- All writes performed by export and sealing use `validate_write_path`; checkpoint writes
  require a discovered sentinel sandbox before creating either the partial or sealed file.
- A final Zarr cannot overwrite a sealed output or an existing diagnostic partial. Seal
  rejects partial artifacts anywhere below the sandbox and never writes `SUCCESS` before
  evidence validation.
- Checkpoint metadata equality is type-exact so `True` cannot match seed `1`; state-dict
  loading is strict.
- No `/data` artifact was added to Git. The implementation is intentionally isolated from
  production `AEFModel` and does not alter its registry or training paths.
