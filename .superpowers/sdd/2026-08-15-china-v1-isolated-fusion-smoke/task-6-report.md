# Task 6 Report: CPU Contract Runner and Real Four-Patch Preparation

## Scope and implementation status

- Added `experiments/china_v1_fusion_smoke/runner.py` with strict CLI dispatch for
  `inspect`, `prepare`, `cpu-contract`, and `npu-smoke`.
- Added `tests/isolated_smoke/test_runner.py` covering stage dispatch, sandbox-only writes,
  four-patch/two-year limits, missing sentinel, source snapshot mutation, strict synthetic
  metadata, path audit, and Task 7 launcher gating.
- `inspect`, `prepare`, and the small-fixture CPU contract are the only stages authorized for
  actual execution in Task 6. No NPU stage has been started.

## TDD evidence

### RED

Command:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD \
  python -m pytest tests/isolated_smoke/test_runner.py -q
```

Expected missing-feature result before production implementation:

```text
ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.runner'
1 error in 6.97s
```

### Initial GREEN

The same focused command after the minimum implementation returned:

```text
...........                                                              [100%]
11 passed in 7.37s
```

Ruff, Black check, and `git diff --check` also passed before the implementation commit.

## Real setup and inspect evidence

The required sandbox bootstrap completed without installation or writes outside the exact root.
The sandbox Python was invoked with `PYTHONNOUSERSITE=1` and
`PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD`.

`inspect` selected these stable IDs:

1. `parent_32643:310:3383`
2. `parent_32643:311:3390`
3. `parent_32643:312:3346`
4. `parent_32643:312:3406`

This is exactly 4 patches and 8 patch-years. Sixteen representative headers (one January raster
per sensor/year for each patch) were read directly from ZIPs. Every header is `128×128`, uses
`EPSG:32643`, and has the expected S1 two-band or S2 ten-band contract. `inspect` reported no
created paths.

## Real prepare evidence

`prepare` completed with the same four IDs and 8 patch-years. Evidence under the fixed sandbox:

- `manifests/patch_selection.json`: four patches, each with 48 exact ZIP member/CRC references;
- `manifests/source_snapshot_before.json` and `source_snapshot_after.json`: 48 fixed source ZIPs,
  size and nanosecond mtime equal before/after, and no source hashing;
- `synthetic/aef/patch_00_2020.pt` through `patch_07_2021.pt`: 8 caches, 4,212,494 bytes each;
- `synthetic/highres_2m/patch_00_2020.pt` through `patch_07_2021.pt`: 8 caches, 5,326,734 bytes
  each;
- `manifests/aef_registry.json` and `highres_2m_registry.json`: 8 ordered entries each;
- `manifests/prepare_manifest.json`: `patch_years=8`, `source_unchanged=true` and explicit
  no-accuracy-conclusion policy;
- `path_audit.json`: 26 created or modified sandbox paths and no `.partial` artifact.

Both registries require `synthetic=true`, `allowed_use=smoke_test_only`,
`formal_training_allowed=false`, and `formal_evaluation_allowed=false`. The high-resolution
registry additionally records `claimed_native_gsd_m=null` and
`contains_real_2m_information=false`.

The measured sandbox size after prepare is **100,427,959 bytes**, below the 5 GiB stop limit.
No accuracy, official AEF semantics, or real 2 m quality conclusion is made.

## CPU and concurrency notes

`runtime.num_workers` remains exactly zero, and the real patch-year TIFF loader is serial. During
selection/header inspection, Task 2 uses a bounded `ProcessPoolExecutor` over at most eight ZIP
metadata/header calls. The observed child processes matched that existing bounded archive pool,
not DataLoader workers, and all exited with the stage. This pool can briefly compete with an
external OCR workload for CPU and remains an explicit final-review consideration; Task 6 did not
change the Task 2 implementation.

## Version control

- Implementation commit: `43daedc feat: add isolated four-patch fusion smoke runner`
- Implementation push: successful to `origin/codex/china-v1-fusion-smoke`
- Base before Task 6: `25080bf6a21489c43f823a731312bdc5128fd15a`

The real sandbox artifacts are not tracked by Git.

