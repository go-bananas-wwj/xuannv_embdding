# SDD ledger — plan: docs/superpowers/plans/2026-08-15-china-v1-isolated-fusion-smoke.md

Merge base: 4cf032a
Execution start: d66d74c

Task 1: review approved implementation commit 893ca33; human ruled that all plan/test commands must replace `PYTHONPATH=$PWD` with `$PWD/src:$PWD/downstreams:$PWD` so imports resolve the isolated worktree. Fix round 1 started.
Task 1: fix round 1/5 (1 addressed, 0 open — explicit isolated-worktree PYTHONPATH; commits 893ca33..e4ee266)
Task 1: complete (commits d66d74c..e4ee266, review clean)
Task 2: minor (deferred): tests/isolated_smoke/test_archive.py uses broad pytest.raises(Exception) for malformed raster.
Task 2: minor (deferred): archive intersection creates multiple bounded process pools; acceptable for four-patch smoke but final review should triage.
Task 2: fix round 1 started for canonical member validation, exact four-patch/48-reference cardinality, full grid equality, and finite-pixel masks.
Task 2: fix round 1/5 (4 addressed, 1 open — affine equality uses default relative tolerance and can accept a 10 m northing shift; commits d232478..8416d59)
Task 2: fix round 2/5 (1 addressed, 0 open — affine equality now uses rtol=0 and 1e-6 absolute tolerance; commits 8416d59..2bf0f74)
Task 2: complete (commits e4ee266..2bf0f74, review clean; 2 deferred minors)
Task 3: fix round 1 started for NaN-safe masked averaging, unit-vector validity, recognized highres registry policy, and contradictory formal-use rejection.
Task 3: fix round 1/5 (4 addressed, 0 open — adversarial masks and registry policy hardened; commits 1badd26..fe05d15)
Task 3: complete (commits 2bf0f74..fe05d15, review clean)
Task 4: minor (deferred): explicit validation for exactly four quarters and exact 5H highres grid could improve fail-closed diagnostics.
Task 4: fix round 1 started for AMP/BF16 zero-gate identity and mask dtype promotion.
Task 4: fix round 1/5 (1 addressed, 0 open — AMP zero-gate identity restored; commits 44f06f9..e5f5803)
Task 4: complete (commits fe05d15..e5f5803, review clean; 1 deferred minor)
Task 5: fix round 1 started for symlink-safe writes/evidence, weights-only checkpoint load, transactional state reload, evidence validation, and complete Zarr contract verification.
Task 5: fix round 1/5 (4 addressed, 3 open — checkpoint apply can mutate on sparse tensor; cross-group patch identity self-attested; caller directory recursive symlink; commits 59f4041..4c90c1f)
Task 5: fix round 2/5 (2 addressed, 1 open — checkpoint application is transactional for sparse/application failures and caller directories reject recursive symlinks; cross-group axes compare and use patch selection when present, but trusted patch selection remains optional; commits 4c90c1f..01ab4ce)
Task 5: fix round 3/5 (1 addressed, 0 open — `manifests/patch_selection.json` is mandatory, strictly four unique patch IDs, matches all group axes, and is included in SUCCESS combined SHA-256; commits 01ab4ce..4aec329)
Task 5: complete (commits e5f5803..2cf836e, review clean)
Task 6: implementation RED/GREEN complete (runner import RED; 11 focused tests GREEN; commit 43daedc pushed).
Task 6: real setup/inspect/prepare complete (four fixed patches, 8 patch-years, 48 source ZIP snapshots unchanged, 100427959-byte sandbox, no NPU started).
Task 6: path-audit self-review fix complete (sequential-stage RED; cumulative audit GREEN; commit 330e133 pushed).
Task 6: complete (real small CPU four-group contract and exact zero-gate identity passed; 98 isolated non-NPU tests passed; sandbox 100429169 bytes; no NPU started).
Task 6: fix round 1 started for symlink-closed sandbox writes, independent NPU/launcher provenance, bounded related-source entry snapshots, and authoritative final SUCCESS audit.
Task 6: minor (deferred): Task 2 archive pools remain bounded at 8 and are independent of data-loading `num_workers=0`.
Task 6: minor (deferred): all 48 monthly refs retain member/CRC, but shape/CRS/transform are stored only for representative headers.
Task 6: fix round 1/5 (3 addressed, 1 open — root/sentinel symlinks fail closed; source audit covers 48 ZIPs plus 55 related directories/253 direct entries; seal validates planned final SUCCESS; Task 7 ancestor check remained open because any argv token could match the launcher path; commits 54f03f4..82d4e77).
Task 6: verification after fix round 1 (109 isolated non-NPU tests passed; real inspect/prepare/cpu replay passed; source unchanged; sandbox 100518960 bytes; no NPU started; finding #2 subsequently reopened).
Task 6: fix round 2/5 (1 addressed, 0 open — Task 7 ancestry now requires the fixed launcher as the exact shell script operand and rejects `bash -c`, `bash -s`, and trailing argv decoys; commit 0dd423d).
Task 6: complete after fix round 2 (22 focused Runner tests and 113 isolated non-NPU tests passed; Ruff/Black/diff-check passed; no NPU inspected or started).
Task 6: review clean (commits 25080bf..403febd; 2 deferred minors retained for final review).
Task 7: launcher/NPU preflight TDD complete; fixed sourced CANN Python path preservation and NPU-stable high-resolution valid-mask downsampling (commit ccfb362 pushed; opt-in NPU Full backward passed).
Task 7: fail-closed physical-device occupancy fallback complete after the host exposed missing `fuser` (commit 181ea2a pushed; `/proc/<pid>/fd` character-device matching covered).
Task 7: two-phase post-tee final sealing complete (commit f316836 pushed; final log metadata is audited, and read-only seal verification rejects later mutation).
Task 7: full foreground physical NPU 2 smoke complete (`base`, `base_aef`, `base_highres`, `full` all finite at `[4,8,64,128,128]`; nonzero AEF/highres gradients; exact zero-gate, checkpoint reload, and deterministic rerun; source unchanged; final `verify_success` passed; sandbox 292396383 bytes).
Task 7: observability note — NPU 2 maps to Ascend 910B4-1 logical `npu:0`; post-run idle HBM was 5% of 65536 MB, but peak HBM was not instrumented and no peak claim is made.
Task 7: review rejected the first seal with 3 important findings and 1 minor finding: tee completion was not independently proven; SUCCESS did not close over every audit-created path; peak HBM and runtime provenance were incomplete; pre-export FP32 norms were mislabeled as persisted FP16 evidence.
Task 7: fix round 1 RED complete (`3 failed, 29 passed` for tee recovery/completion; `10 failed` for seal digest/tamper closure; `3 failed` for HBM/provenance/persisted norms).
Task 7: fix round 1 implementation complete — atomic `TEE_COMPLETE` now binds READY and final-log metadata; recovery and finalization fail closed; the SUCCESS digest covers the mandatory roots plus every audit-created path; per-group allocated/reserved NPU peaks, full runtime provenance, CPU fallback binding, and separate reopened-FP16 norms are mandatory (commits `9467100`, `0ef6e6b`, pushed).
Task 7: rejected first-seal small evidence archived recoverably at `attempts/rejected_20260815_first_seal/`; copy integrity verified before removing 191740996 bytes of replaceable old Zarr output. The archive is diagnostic and excluded from current sealed evidence.
Task 7: fix round 1 corrected physical NPU 2 foreground rerun complete exactly once (`base`, `base_aef`, `base_highres`, `full` finite at `[4,8,64,128,128]`; nonzero gradients; exact zero-gate/checkpoint/determinism; source unchanged; physical device idle before and after).
Task 7: corrected seal independent audit PASS — combined SHA-256 `408993be3d507d4f604fd05a26a671026419c9975374ba403a7612b32860efab`; 323 digest roots / 319 unique leaf files; 322 audit-created entries; 0 uncovered; READY/TEE/log bindings match; SUCCESS newest; 0 partial paths; sandbox 292698261 bytes.
Task 7: corrected per-group NPU peak allocated bytes: base 695546880, base_aef 1165312512, base_highres 1187988992, full 1255098880; peak reserved bytes: base 1025507328, remaining groups 1434451968. Persisted FP16 Zarr norm range for every group: 0.9998238087..1.0001806021 (median 0.9999999404), separately from pre-export FP32 norms.
Task 7: fix round 1 implementation accounts for all 3 important findings and 1 minor finding (0 implementation findings open; pending reviewer confirmation). Task 8 not started.
Task 7: fix round 1 fresh completion verification — 42 archive/data/registry/safety/synthetic tests passed; 43 Runner/launcher/model tests passed with 1 deliberately opt-in NPU test skipped; all 54 Export tests passed across isolated 49+5 partitions; Ruff/Black/shell syntax/diff-check passed; fresh `verify_success` passed.
