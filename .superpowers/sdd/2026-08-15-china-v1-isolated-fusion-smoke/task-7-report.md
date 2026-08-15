# Task 7 Report: Physical NPU 2 Launcher and Full-Size Smoke

## Result

Task 7 fix round 1 is implemented and awaiting review. The corrected fixed foreground launcher ran
the isolated four-branch fusion smoke on physical NPU 2, exposed as the only logical device
`npu:0`. The authoritative sandbox contains four full-size Zarr groups, a checkpoint, metrics,
runtime provenance, reproducibility evidence, a finalized path audit, and a last-written `SUCCESS`
seal. A fresh read-only `verify_success` call passes.

The first Task 7 seal was rejected because it did not prove tee completion, did not digest every
path-audit output, did not record per-group peak NPU memory or complete runtime provenance, and
reported pre-export FP32 norms as if they described the persisted FP16 Zarr arrays. Its small
evidence was archived recoverably under `attempts/rejected_20260815_first_seal/`; its 191,740,996
bytes of replaceable Zarr output were deliberately not duplicated. The corrected physical-NPU run
replaced the rejected current output only after the archive copy was verified byte-for-byte.

This remains a synthetic smoke result only. It does not authorize formal training, formal
evaluation, benchmarking, or any accuracy conclusion. Task 8 has not been started.

## Implementation

The completed implementation includes:

- `scripts/smoke/run_china_v1_isolated_fusion_smoke.sh`: fixed worktree/sandbox foreground
  launcher, physical-device idle check, exact logical-device mapping, CANN environment, tee log,
  explicit post-pipeline `TEE_COMPLETE`, and final sealing;
- `experiments/china_v1_fusion_smoke/runner.py`: fail-closed `/proc/<pid>/fd` occupancy fallback,
  strict `READY_TO_SEAL` and `TEE_COMPLETE` validation, per-group NPU peak-memory measurement,
  complete runtime provenance, CPU-contract fallback binding, and separately labeled FP32 versus
  reopened-FP16 norm summaries;
- `experiments/china_v1_fusion_smoke/export.py`: strict finalizer validation, mandatory evidence
  roots, complete union digest coverage for every `path_audit.created_or_modified` entry, schema
  enforcement for the new evidence, and read-only full-seal verification;
- `experiments/china_v1_fusion_smoke/model.py`: numerically stable all-valid high-resolution mask
  downsampling on NPU;
- `tests/isolated_smoke/test_launcher.py`, `test_runner.py`, `test_export.py`, `test_model.py`, and
  `test_npu_smoke.py`: regression coverage for launcher policy, recovery states, provenance,
  metrics, digest closure, tampering, model behavior, and the opt-in one-visible-NPU path.

## TDD and diagnostic evidence

### Initial launcher and NPU fixes

The first focused launcher run failed in four tests because the launcher did not exist. The first
real opt-in NPU preflight then stopped at the first convolution with ACL error `500001` and
`ModuleNotFoundError: tbe`: the launcher had discarded CANN's Python paths. A regression was added
before preserving those paths after the three explicit isolated-worktree entries.

The next preflight produced an AEF gradient but a zero high-resolution gradient. A controlled
CPU/NPU comparison isolated the cause to strict equality after NPU `avg_pool2d`: a fully valid
pooled mask became `1.0000001192092896`, invalidating every cell. A regression first captured
fully valid and locally invalid masks; the implementation now max-pools the invalid indicator.
The opt-in NPU test subsequently passed on physical NPU 2.

The host has no `fuser` binary, so the first full-launch attempt correctly stopped before NPU use.
Tests then required an empty `/proc` scan to mean idle, matching character-device `st_rdev` to mean
busy, and an unreadable process descriptor directory to fail closed.

### Fix round 1 RED/GREEN

The review findings were reproduced before implementation:

```text
tee-completion protocol: 3 failed, 29 passed
seal digest and tamper closure: 10 failed
peak HBM, provenance, and persisted norms: 3 failed
```

The former two-phase protocol could seal after the Runner returned but before tee itself was proven
complete. The launcher now has three states:

1. `npu-smoke` writes `READY_TO_SEAL` after compute evidence is complete;
2. only after the foreground tee pipeline exits successfully, `--mark-tee-complete` atomically
   writes `TEE_COMPLETE`, binding the exact READY hash and final log path, size, nanosecond mtime,
   and SHA-256;
3. `--finalize-seal` independently revalidates READY and TEE, completes the path audit, and writes
   `SUCCESS` last.

Recovery fails closed for READY without TEE and for TEE without READY. READY plus a valid TEE may
resume finalization without rerunning NPU compute. The finalizer rejects any later log or marker
mutation.

The `SUCCESS` combined digest now covers the strict union of every
`path_audit.created_or_modified` evidence path and mandatory `manifests`, `synthetic`, and `outputs`
roots, excluding only `SUCCESS` itself. Missing, symlinked, outside-root, duplicate, and unknown
entries fail closed. The corrected seal covers 323 digest roots resolving to 319 unique leaf files;
all 322 audit-created paths are covered, with no uncovered entry.

## Fixed foreground execution

Immediately before the corrected execution, the independent guard reported:

```text
device=/dev/davinci2 users=() idle=True
```

The only corrected full-run command was:

```bash
bash scripts/smoke/run_china_v1_isolated_fusion_smoke.sh
```

It ran once in the foreground from the fixed linked worktree, without `torchrun`, `nohup`, a
background job, or a process-kill command. The launcher set:

```text
ASCEND_RT_VISIBLE_DEVICES=2
XUANNV_SMOKE_DEVICE=npu:0
PYTHONNOUSERSITE=1
WANDB_MODE=disabled
```

and sourced `/usr/local/Ascend/cann-9.0.0/set_env.sh`. Its `PYTHONPATH` began with the fixed
worktree's `src`, `downstreams`, and root entries and retained the sourced CANN entries. A fresh
post-run `/proc` guard again reported `users=() idle=True`.

## Runtime and device provenance

- Git commit executed: `0ef6e6b513ef4811a14a89e50e63bd8ea050c8a4`.
- Physical device: `/dev/davinci2`, physical NPU 2; only logical device `npu:0` was visible.
- Device name: `Ascend910B4-1`; visible-device value `2`; logical device count `1`.
- Interpreter: the exact sandbox venv
  `/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815/env/bin/python`.
- Python 3.11.15; PyTorch 2.6.0+cpu; torch-npu 2.6.0.post5.
- CANN root/version: `/usr/local/Ascend/cann-9.0.0`, version 9.0.0, with its install-info path.
- Driver: 26.0.rc1, with its version-info path.
- Module paths record the exact worktree `model.py` and `runner.py`, and imported `torch` and
  `torch_npu` package files.
- CPU contract fallback:
  `manifests/cpu_contract.json`, SHA-256
  `2bfc300c7d3ce98393ef3f11e46221ce4a4364a27e4f0dcbf9218262323cae22`.

The finalizer requires these fields; their absence or mismatch prevents sealing.

## Full-size output and NPU memory audit

All four Zarr groups were independently reopened after sealing. Each group has embedding dtype
`float16`, shape `[4,8,64,128,128]`, boolean valid-mask shape `[4,8,1,128,128]`, and only finite
values.

| Group | Latency (s) | Peak allocated (bytes) | Allocated delta (bytes) | Peak reserved (bytes) | Reserved delta (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 5.260446 | 695,546,880 | 538,973,184 | 1,025,507,328 | 847,249,408 |
| `base_aef` | 0.653409 | 1,165,312,512 | 740,301,312 | 1,434,451,968 | 983,564,288 |
| `base_highres` | 0.666613 | 1,187,988,992 | 762,977,792 | 1,434,451,968 | 983,564,288 |
| `full` | 0.007950 | 1,255,098,880 | 830,087,680 | 1,434,451,968 | 983,564,288 |

Before each group, the Runner synchronizes the NPU, clears unused cache, synchronizes again,
resets peak statistics, and records allocated/reserved baselines. It synchronizes again before
reading peak allocated and reserved bytes. These are smoke-run observability values, not benchmark
statistics; in particular, one-shot latency is affected by warm-up and compilation.

The report now distinguishes the model's pre-export FP32 norm check from the independently reopened
persisted FP16 Zarr norm check:

| Evidence | Source dtype | Computation dtype | Min | Median | Max |
| --- | --- | --- | ---: | ---: | ---: |
| Pre-export model output (all groups) | float32 | float32 | 0.9999996424 | 1.0 | 1.0000003576 |
| Reopened persisted Zarr (all groups) | float16 | float32 | 0.9998238087 | 0.9999999404 | 1.0001806021 |

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
  - high-resolution stem: `0.15332698542624712`;
  - output projection: `25.318119764328003`.
- Checkpoint reload maximum absolute error: `0.0`.
- Deterministic rerun: `matches=true`, maximum absolute error `0.0`, seed `20260815`.
- Checkpoint SHA-256:
  `4960b82b90fb89645099c46649fd6a0ffc722aab8561688e80d7978d7b1a27f9`, independently matching
  `reproducibility.json`.

## Isolation and corrected final seal

- Source snapshots before and after are byte-identical; `run_manifest.json` records
  `source_unchanged=true`.
- Final path-audit stage: `finalize-seal`; its stage history is exactly
  `["npu-smoke", "finalize-seal"]`.
- `path_audit.created_or_modified`: 322 entries; no created evidence is outside the digest union.
- `path_audit.json` SHA-256:
  `208625713720235db0059396a3f8ad22cd1671e0995f032497000bce0e74b5d6`.
- Foreground log: 38,033 bytes, mtime_ns `1786796244056000000`, SHA-256
  `57f343981a3f857f34a0cdc6d0bce46ba3d974b332d948c9836bb845e88ef4ca`.
- `READY_TO_SEAL`: 412 bytes, SHA-256
  `f7c36cb690cd838a63839912cbcf19f6e15bd2734af6f3b22dc2e039dbe23aaf`.
- `TEE_COMPLETE`: 572 bytes; it independently binds that READY hash and all final-log metadata.
- `SUCCESS` was sealed at `2026-08-15T12:17:44.025102Z`, is newer than every other current
  sandbox entry, and records combined SHA-256
  `408993be3d507d4f604fd05a26a671026419c9975374ba403a7612b32860efab`.
- Fresh `verify_success`: PASS; remaining `.partial` paths: 0.
- Final sandbox size, including the small rejected-attempt archive: 292,698,261 bytes, below 5 GiB.

The main worktree pre/post snapshots match exactly: commit
`4cf032a6215fa9c573c998a0ea6daa005b48fb96`, branch `v3-semantic-64d`, upstream ahead/behind
`+0/-0`, the same two modified tracked paths, and the same five untracked paths/directories. The
Task 7 linked worktree was clean and equal to its upstream before this documentation update.

## Verification

The final non-NPU regression suite was split so that no single process retained all test resources:

```text
archive/data-contract/registry/safety/synthetic: 42 passed in 18.61s
runner/launcher/model/NPU-test: 43 passed, 1 skipped in 7.88s
Export main partition: 49 passed, 5 deselected in 27.09s
Export complementary partition: 5 passed, 49 deselected in 10.80s
Ruff: All checks passed!
Black: 5 files would be left unchanged.
launcher shell syntax: PASS
git diff --check: PASS
fresh verify_success: PASS
```

Together the Export partitions cover all 54 Export tests exactly once. The skipped test is the
deliberately opt-in NPU integration test; its real invocation and the corrected full smoke both
passed on physical NPU 2. Independent post-run checks also confirmed SUCCESS is the newest current
sandbox evidence, no `.partial` path remains, every source snapshot still matches, all 322
audit-created paths are digest-covered, and the main worktree snapshot is unchanged.

## Version control

All substantive implementation steps were committed and pushed immediately:

- `ccfb362 test: run isolated fusion smoke on physical NPU 2`
- `181ea2a fix: fail closed when fuser is unavailable`
- `f316836 fix: seal smoke only after foreground logging`
- `9467100 fix: bind complete NPU smoke evidence`
- `0ef6e6b fix: preserve smoke interpreter provenance`

The sandbox artifacts remain outside Git. The rejected seal is retained only as diagnostic evidence
and is not part of the corrected current seal. Task 8 has not been started.
