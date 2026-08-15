# Task 7 Report: Physical NPU 2 Launcher and Full-Size Smoke

## Result

Task 7 is complete. The fixed foreground launcher ran the isolated four-branch fusion smoke on
physical NPU 2, exposed as the only logical device `npu:0`. The final sandbox contains four
full-size Zarr groups, a checkpoint, metrics, reproducibility evidence, a finalized path audit,
and a last-written `SUCCESS` seal. A fresh read-only `verify_success` call passes.

This is a synthetic smoke result only. It does not authorize formal training, formal evaluation,
or any accuracy conclusion.

## Implementation

The Task 7 implementation added or changed:

- `scripts/smoke/run_china_v1_isolated_fusion_smoke.sh`: fixed worktree/sandbox foreground
  launcher, physical-device idle check, exact logical-device mapping, CANN environment, tee log,
  and post-tee final sealing;
- `tests/isolated_smoke/test_launcher.py`: static launcher policy and finalize-order coverage;
- `tests/isolated_smoke/test_npu_smoke.py`: opt-in one-visible-NPU Full forward/backward test;
- `experiments/china_v1_fusion_smoke/model.py`: numerically stable all-valid high-resolution mask
  downsampling on NPU;
- `experiments/china_v1_fusion_smoke/runner.py`: fail-closed `/proc/<pid>/fd` occupancy fallback
  when `fuser` is unavailable, plus two-phase `READY_TO_SEAL`/finalize dispatch;
- `experiments/china_v1_fusion_smoke/export.py`: final-log metadata in the authoritative audit and
  a read-only full-seal verifier;
- focused model, Runner, and Export regressions for all fixes above.

## TDD and diagnostic evidence

### Launcher RED/GREEN

The first focused launcher run failed in four tests because the launcher did not exist. After the
minimum foreground launcher was added, the static launcher suite passed.

### NPU preflight failures and fixes

The first real opt-in NPU preflight stopped at the first convolution with ACL error `500001` and
`ModuleNotFoundError: tbe`. The launcher had replaced `PYTHONPATH` with worktree paths and thereby
removed CANN's Python paths. A launcher regression was added before preserving CANN's paths after
the three explicit worktree entries.

The next preflight produced a nonzero AEF gradient but a zero high-resolution gradient. A
controlled CPU/NPU comparison isolated the issue to strict equality after NPU `avg_pool2d`: a
fully valid pooled mask was `1.0000001192092896`, so `.eq(1.0)` invalidated every cell. A regression
first captured full-valid and locally invalid masks, then the implementation was changed to
max-pool the invalid indicator and compare it with zero. The opt-in NPU test subsequently passed
(`1 passed` in 18.08 seconds).

### Missing `fuser` and final-seal fixes

The first full-launch attempt stopped before any NPU operation because this host has no `fuser`
binary. Regression tests then required an empty `/proc` scan to mean idle, matching character
device `st_rdev` to mean busy, and an unreadable process descriptor directory to fail closed. The
Runner and launcher now share that behavior instead of treating a missing command as idle.

Review of that aborted attempt also found that the former launcher could write `SUCCESS` before
`tee` appended the Runner's final JSON to the log. Regressions proved the post-seal mutation. The
launcher now uses two phases: `npu-smoke` writes `READY_TO_SEAL`, the foreground tee pipeline
finishes, and `--finalize-seal` audits the completed log before writing `SUCCESS`. The read-only
verifier rejects any later mutation. Focused Export/Runner/launcher verification returned
`71 passed`, and the combined Export/Runner/launcher/model/NPU selection returned
`81 passed, 1 skipped` before the full run.

## Fixed foreground execution

Immediately before execution, the independent device guard reported:

```text
device=/dev/davinci2 users=() idle=True
```

The only full-run command was:

```bash
bash scripts/smoke/run_china_v1_isolated_fusion_smoke.sh
```

It ran in the foreground from the fixed linked worktree, without `torchrun`, `nohup`, a background
job, or a process-kill command. The launcher set:

```text
ASCEND_RT_VISIBLE_DEVICES=2
XUANNV_SMOKE_DEVICE=npu:0
PYTHONNOUSERSITE=1
WANDB_MODE=disabled
```

and sourced `/usr/local/Ascend/cann-9.0.0/set_env.sh`. Its `PYTHONPATH` began with the fixed
worktree's `src`, `downstreams`, and root entries and retained the sourced CANN entries.

## Device and software evidence

- Physical device: `/dev/davinci2`, NPU ID 2, Chip Logic ID 2.
- Chip: `Ascend 910B4-1`; board product `IT21HMDC_Bin4_1`.
- Launcher-visible logical device: `npu:0` and exactly one visible NPU.
- CANN toolkit: 9.0.0 (`V100R001C10SPC001B250`).
- Driver: 26.0.rc1; firmware: 9.0.0.0.205.
- Python: 3.11.15; PyTorch: 2.6.0+cpu with torch-npu 2.6.0.post5.

`npu-smi` reports 65,536 MB HBM capacity. A fresh post-run idle query reported 5% HBM use and 0%
NPU/AICore utilization. **Peak HBM was not instrumented by this sealed run**, so there is no valid
peak-HBM value to report and the post-run 5% observation must not be interpreted as one.

## Full-size output audit

All four Zarr groups were reopened independently after sealing:

| Group | Embedding | Valid mask | Finite | vMF norm min / median / max | One-shot latency (s) |
| --- | --- | --- | --- | --- | ---: |
| `base` | float16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | yes | 0.9999996424 / 1.0 / 1.0000003576 | 5.464612 |
| `base_aef` | float16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | yes | 0.9999996424 / 1.0 / 1.0000003576 | 0.747578 |
| `base_highres` | float16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | yes | 0.9999996424 / 1.0 / 1.0000003576 | 0.761585 |
| `full` | float16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | yes | 0.9999996424 / 1.0 / 1.0000003576 | 0.131724 |

These are single smoke timings, not benchmark statistics. The first `base` measurement includes
NPU cold-start/compilation overhead and must not be used for branch-performance comparisons.

Every group has the same patch axis:

1. `parent_32643:310:3383`
2. `parent_32643:311:3390`
3. `parent_32643:312:3346`
4. `parent_32643:312:3406`

and the same period axis from `2020Q1` through `2021Q4`.

## Fusion, gradients, checkpoint, and reproducibility

- Full/Base zero-gate maximum absolute error: `0.0`.
- Gate-override gradient L1:
  - AEF adapter: `0.2702299631200731`;
  - high-resolution adapter: `0.4636795846745372`;
  - high-resolution stem: `0.1533270152285695`;
  - output projection: `25.318119764328003`.
- Checkpoint reload maximum absolute error: `0.0`.
- Deterministic rerun: `matches=true`, maximum absolute error `0.0`, fixed seed `20260815`.
- Checkpoint SHA-256:
  `287bb7fb9ffeafea7076e5c79156a9fee5da472873a77a9fa06e25e350fa6b13`.
- Independent checkpoint hashing exactly matches `reproducibility.json`.

## Isolation and final seal

- Source snapshots before and after are byte-identical; `run_manifest.json` records
  `source_unchanged=true`.
- No `.partial` path remains.
- Final sandbox size: **292,396,383 bytes**, below 5 GiB.
- Final path audit stage: `finalize-seal`; it declares the completed foreground log and
  `READY_TO_SEAL` with exact size, nanosecond mtime, and SHA-256.
- Foreground log: 40,705 bytes,
  SHA-256 `ad54984456aacba528815d81416fb4415339b10eeaaa95d640c591a4382c2726`.
- `READY_TO_SEAL`: 412 bytes,
  SHA-256 `29cf862ed768f91b00b0e67cb80d905f06d57f5974809338aebbebc905c8d6f2`.
- `SUCCESS` was sealed at `2026-08-15T11:30:13.745620Z`, is newer than every other sandbox
  entry, and records combined SHA-256
  `db42b6afb3c464ecc3fd2f9dcc1857c26a77ec6f318d6f1ff621a88a959fa440`.
- Fresh `verify_success` result: PASS.

The audit stage history is transparent about the earlier authorized CPU replay and stopped Task 7
attempts:
`["cpu-contract", "prepare", "cpu-contract", "prepare", "cpu-contract", "npu-smoke", "finalize-seal"]`.
Only the final `npu-smoke` reached NPU execution and `READY_TO_SEAL`; the finalizer alone wrote
`SUCCESS`.

The main worktree pre/post snapshots match exactly: commit
`4cf032a6215fa9c573c998a0ea6daa005b48fb96`, branch `v3-semantic-64d`, upstream ahead/behind
`+0/-0`, the same two modified tracked paths, and the same five untracked paths/directories. The
Task 7 linked worktree was clean and equal to its upstream after every pushed implementation step.

## Fresh completion verification

The final non-NPU regression suite was split only to avoid one large process retaining all test
resources; together the two commands cover every file in `tests/isolated_smoke`:

```text
archive/data-contract/registry/safety/synthetic: 42 passed in 19.06s
export/runner/launcher/model/NPU-test: 81 passed, 1 skipped in 23.13s
Ruff: All checks passed!
Black: 8 files would be left unchanged.
git diff --check: exit 0
fresh verify_success: PASS
SUCCESS newer than latest other entry: true
remaining .partial paths: 0
```

The skipped test is the deliberately opt-in NPU integration test. Its real opt-in invocation had
already passed on physical NPU 2 before the full smoke, as recorded above.

## Version control

All substantive implementation steps were committed and pushed immediately:

- `ccfb362 test: run isolated fusion smoke on physical NPU 2`
- `181ea2a fix: fail closed when fuser is unavailable`
- `f316836 fix: seal smoke only after foreground logging`

The sandbox artifacts remain outside Git. Task 8 has not been started.
