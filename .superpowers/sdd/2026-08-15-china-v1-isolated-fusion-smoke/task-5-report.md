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

## Review fix round 1

### Changes

- Added lexical, non-resolving sandbox checks that reject target, partial, final, and
  ancestor symlinks before writes, reads, hashing, or sealing. The checks still invoke the
  Task 1 `validate_write_path` guard after the symlink-free validation.
- Changed checkpoint reads to `torch.load(..., weights_only=True)`, require a sentinel
  sandbox for reads as well as writes, and validate payload structure before use.
- Prevalidate exact state-dict keys, tensor types, shapes, and dtypes before
  `load_state_dict`, preventing malformed payloads from partially mutating a model.
- Require the four fixed JSON evidence paths to be non-symlink regular files containing
  JSON objects; require the checkpoint to pass safe structural validation.
- Strengthened Zarr verification to reject internal symlinks, uncompressed/non-Zstd
  arrays, unexpected arrays, invalid identifier dtypes/values, and mismatched identifier
  attributes.

### Adversarial TDD evidence

- RED: after adding the adversarial cases, focused export tests produced `13 failed,
  11 passed`. The failures covered broken/live checkpoint partial symlinks, unsafe legacy
  pickle execution, missing/unexpected/shape-mismatched state dicts, directory/malformed
  JSON/malformed checkpoint evidence, mandatory external Zarr symlinks, and
  uncompressed/extra-array/wrong-identifier Zarr changes.
- GREEN: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke/test_export.py -q` → `27 passed in 13.57s`.
- Regression: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke -q -m 'not npu'` → `76 passed in 25.15s`.
- Static checks: `python -m ruff check experiments/china_v1_fusion_smoke
  tests/isolated_smoke`, `python -m black --check experiments/china_v1_fusion_smoke/export.py
  tests/isolated_smoke/test_export.py`, and `git diff --check` all passed.

### Self-review and delivery

- The new tests include broken and live target/partial symlinks, both a direct mandatory
  Zarr symlink and a symlinked mandatory ancestor, a payload that would write a marker if
  legacy pickle executed, and byte-for-byte model-state preservation on malformed loads.
- All mutations remain confined to pytest temporary sandboxes; no `/data` artifact was
  staged.
- Code/test commit: `d73e51c fix: harden smoke export sealing boundaries`; pushed to
  `origin/codex/china-v1-fusion-smoke`.

## Review fix round 2

### Changes

- State-dict validation now requires CPU, dense `torch.strided` tensors before applying a
  checkpoint. Loading snapshots every current state tensor and copies the snapshot directly
  back on any application exception, so a module hook cannot leave prior parameters changed.
- SUCCESS now compares the complete `(patch_id, period)` axes across all four Zarr groups.
  When `manifests/patch_selection.json` provides `patch_ids` (or a list of patch objects),
  the common Zarr patch axis must also match that trusted selection.
- Any caller-supplied required directory is recursively checked for symlinks before it can
  be hashed; `_digest_path` repeats that guard defensively.

### Adversarial TDD evidence

- RED: the three new scenarios produced `4 failed, 27 passed`: a same-shape sparse COO
  tensor changed the earlier parameter before the later failure; a module that mutates then
  throws left its value changed; coordinated `patch_id` array+attribute edits across every
  group sealed; and an external symlink child of a supplied directory was hashed.
- GREEN: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke/test_export.py -q` → `31 passed in 15.15s`.
- Regression: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest
  tests/isolated_smoke -q -m 'not npu'` → `80 passed in 27.34s`.
- Static checks: `python -m ruff check experiments/china_v1_fusion_smoke
  tests/isolated_smoke`, `python -m black --check experiments/china_v1_fusion_smoke/export.py
  tests/isolated_smoke/test_export.py`, and `git diff --check` passed.

### Self-review and delivery

- Regression tests prove byte-for-byte state preservation for both sparse and deliberately
  mutating application failures; they also prove trusted patch selection rejects a
  coordinated self-attested cross-group axis mutation and that an external child symlink
  cannot enter the digest.
- Code/test commit: `16714bb fix: make smoke sealing and reload transactional`; pushed to
  `origin/codex/china-v1-fusion-smoke`.
