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

### Path-audit regression RED/GREEN

The first real CPU run exposed that a later stage replaced, rather than accumulated, the prior
prepare path list. A regression test ran `prepare` followed by `cpu-contract` and failed before
the fix with:

```text
KeyError: 'stages'
1 failed in 7.02s
```

The minimal fix validates and merges prior relative paths/stages before atomically rewriting the
audit. The regression then passed (`1 passed in 6.89s`), and the focused runner suite passed with
`12 passed in 7.66s`.

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
- `path_audit.json`: the first prepare observed 26 created or modified sandbox paths and no
  `.partial` artifact.

Both registries require `synthetic=true`, `allowed_use=smoke_test_only`,
`formal_training_allowed=false`, and `formal_evaluation_allowed=false`. The high-resolution
registry additionally records `claimed_native_gsd_m=null` and
`contains_real_2m_information=false`.

The measured sandbox size after the first prepare was **100,427,959 bytes**, below the 5 GiB stop
limit.
No accuracy, official AEF semantics, or real 2 m quality conclusion is made.

## Real CPU contract evidence

The sandbox Python ran the default small fixture, not `--full-shape`. All four required groups
completed:

| Group | Shape | Finite | Maximum vMF norm error |
| --- | --- | --- | ---: |
| `base` | `[2,4,64,16,16]` | true | `2.980232238769531e-07` |
| `base_aef` | `[2,4,64,16,16]` | true | `2.980232238769531e-07` |
| `base_highres` | `[2,4,64,16,16]` | true | `2.980232238769531e-07` |
| `full` | `[2,4,64,16,16]` | true | `2.980232238769531e-07` |

`full_zero_gate_matches_base=true` used exact tensor equality for both `pre_vmf` and normalized
embedding. The evidence is in `manifests/cpu_contract.json`; it explicitly forbids an accuracy
conclusion.

After the cumulative-audit fix, `prepare` and `cpu-contract` were replayed without NPU access so
the actual evidence uses the corrected schema. Current `path_audit.json` contains 27 cumulative
paths and transparently records the stage history
`["cpu-contract", "prepare", "cpu-contract"]`: the initial CPU run, preparation replay, and final
CPU replay. The replay again found all 48 source ZIP size/mtime records unchanged. Current sandbox
size is **100,429,169 bytes**, still below 5 GiB.

## Final test and static verification

Fresh completion commands used `PYTHONNOUSERSITE=1` and the explicit worktree `PYTHONPATH`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD \
  python -m pytest tests/isolated_smoke -q -m 'not npu'
python -m ruff check experiments/china_v1_fusion_smoke tests/isolated_smoke
python -m black --check experiments/china_v1_fusion_smoke/runner.py \
  tests/isolated_smoke/test_runner.py
git diff --check
```

Results:

```text
98 passed in 32.14s
All checks passed!
2 files would be left unchanged.
git diff --check: exit 0
```

## Concurrency note

`runtime.num_workers` remains exactly zero, and the real patch-year TIFF loader is serial. During
selection/header inspection, Task 2 uses a bounded `ProcessPoolExecutor` over at most eight ZIP
metadata/header calls. The observed child processes matched that existing bounded archive pool,
not DataLoader workers, and all exited with the stage. This pool can briefly compete with an
external OCR workload for CPU and remains an explicit final-review consideration; Task 6 did not
change the Task 2 implementation.

## Version control

- Implementation commit: `43daedc feat: add isolated four-patch fusion smoke runner`
- Implementation push: successful to `origin/codex/china-v1-fusion-smoke`
- Preparation evidence commit: `087b615 docs: record isolated smoke preparation evidence`
- Cumulative audit fix: `330e133 fix: preserve cumulative smoke path audit`
- Base before Task 6: `25080bf6a21489c43f823a731312bdc5128fd15a`

The real sandbox artifacts are not tracked by Git. Task 6 did not invoke `npu-smoke`, inspect an
NPU, export accuracy metrics, or write `SUCCESS`; those remain Task 7 responsibilities.

## Fix Round 1: isolation, launcher provenance, source-tree audit, and final seal

Independent review rejected the initial Task 6 result on four Important findings. Round 1
reproduced all four, but follow-up review showed the Task 7 ancestor fix was incomplete. The
corrected round 1 accounting is therefore **3 addressed, 1 open**; round 2 below closes that
remaining finding. No fix round ran NPU code.

### 1. Symlink-closed sandbox and sentinel

The original `ensure_sandbox` used `resolve()` and `is_file()`, so a root symlink to an outside
directory and a symlink sentinel were both accepted. The two real-filesystem regression tests
failed with `DID NOT RAISE SafetyError` before the fix.

`ensure_sandbox` now uses lexical absolute paths and lstat-aware symlink checks for the root,
ancestors, and sentinel. `validate_write_path` also rejects every existing symlink component and
performs a final resolved containment check. Because all Runner writes already pass through this
function, the stricter guard applies uniformly. Focused safety plus Runner verification returned
`22 passed in 7.63s`.

### 2. Independent physical NPU and Task 7 launcher identity

The initial guard trusted four environment variables. Three device/provenance cases were RED
before implementation: a manually forged environment without launcher ancestry, a missing
`/dev/davinci2`, and a busy device.

The round 1 Runner independently required:

- the exact isolated worktree as current directory;
- the exact sandbox `env/bin/python` interpreter;
- physical `/dev/davinci2` to exist as a character device;
- `fuser /dev/davinci2` to report no user;
- a real `/proc` parent chain containing bash/sh executing the fixed Task 7 launcher from the
  fixed worktree;
- the launcher itself to be a regular, non-symlink file.

A controlled `/proc` fixture verified the positive parent-chain parser. Manually setting the four
environment variables could not satisfy these checks. Runner focused verification returned
`16 passed in 7.62s`. However, this first version accepted the fixed launcher path in any argv
position, so it did not prove that the shell actually executed that file. Finding #2 therefore
remained open after round 1. The physical device helper was replaced only in unit tests; no real
NPU device was inspected or started in Task 6.

### 3. Bounded related-source directory snapshots

The initial snapshot covered only canonical ZIP size/mtime and did not notice a new sibling
`.forbidden.lock`. The regression inserted that file between prepare snapshots and failed because
the old Runner did not raise.

The snapshot now records:

- the 48 canonical ZIPs with type, size and nanosecond mtime;
- every canonical archive parent directory;
- the necessary year, sensor and source-root ancestors;
- the complete direct-entry name/type/size/mtime list for each watched directory.

It does not recurse unrelated trees and does not hash any large ZIP. Both changed-ZIP and
new-direct-entry stop-rule tests pass; the full Runner suite at that point returned `17 passed in
7.71s`.

### 4. Authoritative path audit includes the final SUCCESS seal

The initial NPU path audit was written before `SUCCESS` but omitted it, while the in-memory result
appended it later. Two RED tests proved the gap: Runner did not accept a planned final-seal
argument, and `seal_success` accepted an empty audit.

The NPU stage now writes the on-disk audit while `SUCCESS` is absent and includes:

```json
{
  "created_or_modified": ["...", "SUCCESS"],
  "final_seal": {
    "path": "SUCCESS",
    "status": "expected_last_write",
    "exists_when_audit_written": false
  }
}
```

`seal_success` validates the exact stage, created-path entry, path, status and pre-write existence
flag before hashing the audit and writing `SUCCESS` last. It never modifies the audit afterward.
Negative tests reject a missing declaration, an altered status, and an altered path. Runner plus
Export focused verification returned `56 passed in 19.94s`; the final Export suite returned
`40 passed in 20.26s`.

### Real replay after the fixes

The authorized real `inspect`, `prepare`, and small `cpu-contract` stages were replayed. No NPU
stage was invoked.

- Stable patch IDs remained unchanged.
- Source snapshots contain 48 archives, 55 bounded watched directories, and 253 direct-entry
  records.
- Before and after snapshots are exactly equal and state `hashing_performed=false`.
- Base, Base+AEF, Base+Highres and Full CPU groups remain finite with exact zero-gate identity.
- No `.partial` artifact or lingering Runner child process remains.
- Final sandbox size is **100,518,960 bytes**, below 5 GiB.

### Final fix-round verification

```text
isolated non-NPU: 109 passed in 33.45s
Ruff: All checks passed!
Black: 6 files would be left unchanged.
git diff --check: exit 0
```

Fix-round commits, all pushed to `origin/codex/china-v1-fusion-smoke`:

- `82147db fix: reject symlink smoke sandbox paths`
- `51b919d fix: verify physical NPU launcher provenance`
- `a7a1605 fix: audit related source directory entries`
- `31fa2b5 fix: seal authoritative smoke path audit`
- `82d4e77 test: reject tampered final smoke seal audit`

### Deferred Minor findings

- Task 2 archive metadata/header pools remain bounded at eight processes and are independent of
  data-loading `num_workers=0`; changing that reviewed Task 2 implementation is deferred.
- `patch_selection.json` keeps all 48 member paths and CRCs but only representative headers, not
  per-entry shape/CRS/transform for all 48 references; expanding that manifest is deferred.

## Fix Round 2: exact Task 7 shell script operand

Follow-up review supplied a concrete provenance decoy: `bash -c ... <fixed-launcher-path>` can
place the launcher path in `$0`/argv without executing it. The round 1 implementation iterated all
tokens and incorrectly accepted that form, `bash -s <fixed-launcher-path>`, and an unrelated
script followed by the fixed path.

### RED evidence

A real fake-`/proc` parameterized test supplied those three cmdlines while keeping the expected
worktree cwd. Before the fix, all three negative cases failed because
`_has_task7_launcher_ancestor` returned `true`:

```text
3 failed, 3 passed, 16 deselected in 6.86s
```

The positive cases independently covered both an absolute fixed launcher operand and the Task 7
planned invocation form:

```text
/usr/bin/bash scripts/smoke/run_china_v1_isolated_fusion_smoke.sh
```

The latter is resolved against the ancestor process cwd read from `/proc/<pid>/cwd`.

### Minimal fix and GREEN evidence

The ancestor check now accepts only a two-element shell argv: the `bash`/`sh` executable and the
fixed launcher as its actual script operand. It no longer scans arbitrary later tokens. This
structurally rejects `-c`, `-s`, and post-script decoys while preserving the fixed worktree,
interpreter, physical `/dev/davinci2`, `fuser` idle, and real parent-chain checks.

```text
launcher-focused: 6 passed, 16 deselected in 7.38s
full Runner: 22 passed in 7.76s
isolated non-NPU: 113 passed in 33.73s
Ruff: All checks passed!
Black: 2 files would be left unchanged.
git diff --check: exit 0
```

Round 2 commit: `0dd423d fix: require exact Task 7 script operand`.

Corrected review accounting: **round 1 = 3 addressed / 1 open; round 2 = 1 addressed / 0 open**.
No NPU was inspected or started.
