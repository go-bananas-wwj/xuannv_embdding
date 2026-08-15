# Task 8 Report: Evidence Report, Regression, and Handoff

## Result

Task 8 produced the planned repository Markdown report and CHANGELOG entry. The human-readable
verdict is **engineering PASS**, while accuracy, real AEF semantics, real 2 m quality, national
production readiness, formal training, and formal evaluation remain explicitly **not evaluated / not
authorized**.

The report consumes only the corrected current seal. The rejected first seal under
`attempts/rejected_20260815_first_seal/` is identified as a diagnostic archive and is not treated as
current evidence. No webpage was created because the requested delivery remains repository
Markdown, and no chart was added because four one-shot groups with a cold-start first group are not
latency-comparable.

## Evidence inventory

Fresh read-only inspection and `verify_success` established:

- current SUCCESS combined SHA-256
  `408993be3d507d4f604fd05a26a671026419c9975374ba403a7612b32860efab`;
- 323 digest roots, 319 unique leaf files, 322 unique path-audit entries, and zero uncovered
  entries;
- four independently reopened finite FP16 Zarr embeddings with shape `[4,8,64,128,128]` and bool
  valid masks with shape `[4,8,1,128,128]`;
- the fixed four patch IDs and `2020Q1` through `2021Q4` axes on every group;
- byte-identical source snapshots covering 48 archives, 55 watched directories, and 253 direct
  entries, with content hashing explicitly disabled;
- exact zero-gate, checkpoint-reload, and deterministic-rerun errors of `0.0`, plus finite nonzero
  gradients on both side branches, the high-resolution stem, and output projection;
- distinct pre-export FP32 and reopened persisted-FP16 vMF norm summaries;
- per-group one-shot latency and peak allocated/reserved NPU bytes, converted to MiB and GiB in the
  report without cross-group performance interpretation;
- complete Python, torch, torch-npu, CANN, driver, Git, and physical-NPU-2 to logical-`npu:0`
  provenance;
- sandbox size 292,698,261 bytes (279.14 MiB / 0.273 GiB), no `.partial` path, and SUCCESS as the
  final current-seal write.

The main worktree remained at commit `4cf032a6215fa9c573c998a0ea6daa005b48fb96` on
`v3-semantic-64d`, ahead/behind `0/0`, with its pre-existing two modified paths and five untracked
paths/directories unchanged.

## Repository deliverables

- `docs/reports/china_v1_isolated_fusion_smoke_20260815.md`: answer-first technical evidence
  report with scope, method, compact evidence tables, robustness/limitations, and next steps;
- `CHANGELOG.md`: 2026-08-15 entry summarizing the isolated smoke and its strict conclusion
  boundary.

The report contains no `/data` Markdown asset link. Absolute runtime paths appear only as quoted
provenance text where relevant.

## Verification

Task 8 did not set the NPU opt-in environment variable and did not trigger NPU execution. The
specified fresh regression returned:

```text
194 passed, 1 skipped in 46.36s
```

The single skip is the deliberately opt-in NPU integration test; the corrected Task 7 seal already
contains its physical NPU 2 execution evidence. Additional checks returned:

```text
Ruff: All checks passed!
git diff --check: PASS
report evidence assertion: PASS
fresh verify_success: PASS
```

## Version control and handoff

The primary deliverables were committed and immediately pushed:

- `33b6ea3 docs: report isolated China V1 fusion smoke evidence`

No sandbox artifact was added to Git, no NPU was touched, and Task 8 made no write to `/data`.
After this ledger update, the task stops before whole-branch review as requested.
