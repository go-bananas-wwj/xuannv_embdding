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
