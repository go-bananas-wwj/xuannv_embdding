# RSE V5 Registered Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible, fail-closed training and downstream-evaluation protocol for the RSE paper using the reviewed v5 complete-2x2 spatial split.

**Architecture:** Preserve the old v4 protocol and its results as diagnostic-only artifacts. Introduce an explicitly versioned v5 protocol that binds every generated training manifest, encoder configuration, embedding export, shot schedule, result record, aggregation, and bootstrap calculation to the v5 split hash. The five v5 folds use compact, non-overlapping 40-patch test clusters, a 20-patch validation set outside the full test buffer, and at least 150 training patches.

**Tech Stack:** Python 3.11, PyTorch/torch_npu, JSON/YAML, pytest, Ruff, Bash launch scripts.

## Global Constraints

- Use only `configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json` for formal v5 evidence.
- Keep v4 code, manifests, outputs, and reports immutable and label them diagnostic-only.
- Do not use downstream task labels to train the embedding encoder.
- OSM building/road/water probes are recorded only as **OSM-assisted spatially held-out readouts**, never as ontology-independent transfer; independent-label claims require a frozen external or adjudicated label release.
- Each fold computes normalization statistics from its upstream training manifest only; its registry entry and source-manifest hash are immutable protocol inputs.
- All v5 configurations are self-contained and run on Ascend NPU.
- Every code/config/document step must pass independent review, be committed, and be pushed before the next step.

---

### Task 1: Add an explicit evidence-class gate

**Files:**
- Modify: `scripts/eval/run_registered_paper_downstream.py`
- Modify: `scripts/report/aggregate_registered_paper_results.py`
- Modify: `scripts/report/paired_spatial_bootstrap.py`
- Create: `scripts/report/admit_registered_results.py`
- Modify: `docs/paper/rse_2026/evidence_ledger.md`
- Test: `tests/test_paper_v5_protocol.py`

**Interfaces:**
- Every result records `protocol_id`, `evidence_class`, and `label_independence_status`.
- `v5_osm_assisted` results can be aggregated only into explicitly marked diagnostic/weak-supervision reports; an `independent_transfer` table requires a frozen independent-label release manifest.
- A release-admission anchor at `configs/eval/release_anchors/rse_v5_osm_assisted.json` records `registry_path`, `registry_sha256`, `zenodo_doi`, and `registry_url`; the URL bytes must hash to `registry_sha256` before any sealed result transitions from internal preliminary to `paper_eligible: true`.
- A release-admission record at `configs/eval/release_admissions/rse_v5_osm_assisted.json` records the anchor hash plus every admitted result/metric/artifact/report hash. Aggregation and manuscript rendering consume this record instead of mutating sealed result entries.

- [ ] Write failing tests that reject an OSM-overlapping task from an independent-transfer report, reject a record without a v5 protocol ID, and reject a mixed v4/v5 bootstrap matrix.
- [ ] Run `python -m pytest tests/test_paper_v5_protocol.py -q` and confirm the tests fail before the gate exists.
- [ ] Implement the evidence-class descriptor and propagate it through result, aggregate, and bootstrap provenance; add the manuscript-table gate; implement `scripts/report/admit_registered_results.py` and require aggregate/bootstrap/manuscript statistics to bind to its immutable release-admission record.
- [ ] Update the evidence ledger to distinguish admissible OSM-assisted results from the still-required independent annotation release; run focused tests and commit `fix: gate paper claims by label independence`.

### Task 2: Version the v5 training inputs and manifest provenance

**Files:**
- Modify: `scripts/experiments/build_paper_subset_registry.py`
- Modify: `scripts/experiments/build_clean_paper_configs.py`
- Modify: `scripts/data/build_paper_manifests.py`
- Modify: `scripts/data/compute_statistics.py`
- Modify: `scripts/train/train.py`
- Create: `configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json`
- Create: `configs/eval/haidian_paper_v5_normalization_statistics.json`
- Create: `configs/paper_registered_v5_20260726/*.yaml`
- Test: `tests/test_paper_v5_protocol.py`

**Interfaces:**
- Consumes the committed v5 split JSON and Haidian patch metadata.
- Produces hash-sealed 40/80/150 training subsets plus train/val/test/buffer manifests, fold-specific training-only normalization statistics, and five self-contained encoder YAML files per enabled experimental family.

- [ ] Write failing tests that reject a v5 subset registry whose source split hash differs from the committed v5 split; reject a manifest sidecar or fold-statistics registry with a mismatched source hash; and assert each generated encoder config has matching v5 split/subset/manifest/statistics hashes and no held-out geography in its manifests.
- [ ] Run `python -m pytest tests/test_paper_v5_protocol.py -q` and confirm the new tests fail before the v5 builders exist.
- [ ] Add explicit `--split`, `--output`, `--protocol-name`, and versioned defaults to the subset/config builders; materialize v5 fold manifests with a sidecar recording source-manifest/split/subset hashes; compute fold statistics from the full upstream training manifest only; record paths/checksums in `haidian_paper_v5_normalization_statistics.json`; require both sidecars in training validation; generate v5 configs without modifying v4 paths.
- [ ] Run the focused tests, Ruff, and config validation; commit `feat: add v5 paper training protocol`.

### Task 3: Bind export, downstream execution, aggregation, and bootstrap to v5

**Files:**
- Modify: `scripts/eval/run_registered_paper_downstream.py`
- Modify: `scripts/eval/prepare_registered_shot_manifests.py`
- Modify: `scripts/report/aggregate_registered_paper_results.py`
- Modify: `scripts/report/paired_spatial_bootstrap.py`
- Create: `scripts/eval/export_registered_v5_paper_encoders.sh`
- Create: `scripts/eval/launch_registered_v5_downstream.sh`
- Create: `configs/eval/registered_embedding_exports_v5_20260726.json`
- Create: `configs/eval/rse_v5_osm_assisted_matrix.json`
- Test: `tests/test_paper_v5_protocol.py`

**Interfaces:**
- Adds a protocol resolver that maps `v4_diagnostic` and `v5_osm_assisted` to immutable split path/hash, expected manifest provenance, and evidence class.
- Requires `--protocol v5_osm_assisted` for each v5 export, shot schedule, probe, aggregation, and bootstrap result.
- `rse_v5_osm_assisted_matrix.json` freezes family, encoder-fold count, comparator, head, task, shot, fold, seed, expected-job-count, and scientific-status fields. Queue, result writer, aggregator, and bootstrap reject cells outside this manifest.

- [ ] Write failing tests showing that a v5 run rejects a v4 split, mismatched split/manifest/statistics hashes, an unversioned embedding registry, a shot schedule from another protocol, a result cell outside the committed matrix manifest, or a bootstrap matrix with mismatched v5 test support.
- [ ] Run the focused tests and confirm they fail for the missing resolver/binding.
- [ ] Implement a protocol descriptor used by encoder provenance, split/statistics validation, export validation, schedules, result metadata, aggregation, and bootstrap; enforce the committed matrix manifest at every entry point; preserve v4 constants only as the v4 descriptor.
- [ ] Add a generic six-shard exporter accepting every family declared in the matrix manifest and emitting a Git-tracked v5 embedding registry only after each completed export's hashes are verified and committed.
- [ ] Add a six-NPU downstream launcher that invokes only `v5_osm_assisted`, writes attempt-specific directories, preserves failed-attempt logs, and admits exactly one canonical successful result ID per job.
- [ ] Run focused tests, existing registered-protocol/aggregation/bootstrap tests, Ruff, and dry-run export/launcher checks; commit `feat: add fail-closed v5 evidence protocol`.

### Task 4: Launch, monitor, and seal the declared v5 encoder matrix

**Files:**
- Create: `scripts/experiments/run_registered_v5_paper_queue.sh`
- Create: `docs/paper/rse_2026/v5_execution_log.md`
- Test: `tests/test_paper_v5_protocol.py`

**Interfaces:**
- Formal primary OSM-assisted matrix: `full_150` five encoder folds and 90 Conv3x3 jobs (building/road/water × 5/10 shot × 5 folds × 3 seeds). The AEF comparison uses the same v5 split, schedules, labels, probe, and 90-job matrix.
- Predeclared scaling/ablation matrix: `full_40` and `full_80` each use five encoders and 90 OSM-assisted jobs against `full_150`; `no_osm_150`, `coarse_osm_only_150`, `probe_nohardneg_150`, `no_highres_path_150`, and `no_masking_150` each use five encoders and 90 OSM-assisted jobs against `full_150`. No other family or probe cell can enter the RSE primary/ablation tables without a new committed protocol version.
- Each queue job runs in `/data/xuannv_embedding/outputs/paper_registered_v5_20260726/<experiment>/attempt_N/`; an immutable `attempt_manifest.json` records config, split, subset-registry, manifest-sidecar, parent-attempt, and checkpoint hashes. A verified `canonical_attempt.json` selects one successful attempt without altering any failed attempt.

- [ ] Write a failing test or shell dry-run assertion that checks every v5 job uses a v5 config/output namespace, cannot resume from v4, and uses only a declared matrix family.
- [ ] Implement the three two-NPU lane queue with restart handling, checkpoint verification, immutable status logs, `attempt_N` recovery paths, attempt manifests, and a verified canonical-attempt pointer; add failure/restart tests.
- [ ] Commit and push the queue and blank execution-log template before launch.
- [ ] Launch the five `full_150` encoders; independently audit completion and checkpoint-fold provenance; update, commit, and push the execution log.
- [ ] Export the five sealed v5 encoders; generate and commit the v5 embedding registry before launching any probe.

### Task 5: Add the matched AEF baseline adapter

**Files:**
- Create: `scripts/eval/export_aef_v5_embeddings.py`
- Create: `scripts/eval/launch_aef_v5_downstream.sh`
- Create: `configs/eval/registered_aef_embedding_exports_v5_20260726.json`
- Modify: `scripts/eval/run_registered_paper_downstream.py`
- Test: `tests/test_paper_v5_protocol.py`

**Interfaces:**
- Consumes the AEF embedding source and the identical committed 320-patch manifest; emits one 64-dimensional, 128x128-compatible embedding map per test protocol patch or a recorded deterministic spatial adapter when native AEF resolution differs.
- Pins the AEF source release URI/version/checksum, input date policy, source tile/grid mapping, and a deterministic non-learned spatial/channel adapter. The adapter may not consume task labels, probe labels, validation metrics, or test metrics.
- Emits a Git-tracked AEF export registry carrying source release/artifact hash, temporal/grid mapping, adapter source hash and parameters, split hash, manifest hash, per-map hashes, and `protocol_id: v5_osm_assisted`.
- Uses the same task labels, fold-specific train/test patch support, 5/10-shot schedules, Conv3x3 hyperparameters, threshold rule, result schema, and bootstrap pairing keys as Xuannv.

- [ ] Write failing tests rejecting AEF exports with a mismatched release/manifest/split/schedule, a learned or non-recorded resolution adapter, a missing source temporal/grid mapping, or a result whose task support differs from the paired Xuannv result.
- [ ] Re-export the cached annual-2025 AEF COGs onto every committed local reference grid using a deterministic non-learned reprojection/mosaic rule; the old locally window-resampled maps remain diagnostic-only and cannot enter V5 results.
- [ ] Seal one source manifest before probing: mirror and source URIs, temporal coverage, COG/index hashes, per-patch COG/mosaic choice, reference-grid CRS/transform/bounds/hash, resampling parameters, nodata fraction, per-map checksum, and a validity-mask checksum. The V5 adapter must reject incomplete coverage, a learned adapter, or nonidentical valid evaluation support.
- [ ] Implement the AEF source loader plus explicit spatial adapter and export provenance; generate one v5 registry only after the exact per-map and validity-mask checksums are verified, then commit and push it.
- [ ] Implement the matched six-NPU 90-job AEF launcher and run dry-run/provenance tests; commit `feat: add matched AEF v5 baseline protocol`.

**Interpretation boundary:** the resulting comparison is an annual global AEF baseline versus a monthly
city-scale representation under static OSM-assisted spatial readout. It must not be reported as a
same-date temporal comparison, a change-detection comparison, or evidence that one model has a
temporally superior representation.

### Task 6: Run v5 probes, admission, and statistical analysis

**Files:**
- Create: `docs/paper/rse_2026/v5_results_ledger.md`
- Create: `docs/paper/rse_2026/v5_downstream_results.md`
- Test: existing registered-protocol, aggregation, and paired-bootstrap tests

**Interfaces:**
- Consumes sealed v5 exports and v5 shared 5/10-shot schedules.
- Produces versioned per-job results, aggregate metrics, and formal paired spatial bootstrap intervals.

- [ ] Generate v5 schedules for building, road, and water over five folds × three seeds; verify protocol/split/label hashes.
- [ ] Run the 90-job Conv3x3 Xuannv matrix and the matched 90-job AEF matrix across six NPUs; keep every failed attempt immutable and use the canonical successful attempt only after provenance verification.
- [ ] Aggregate only exact v5 five-fold/three-seed matrices and invoke `paired_spatial_bootstrap.py` with `--protocol v5_osm_assisted`; produce separate OSM-assisted and independent-label report sections.
- [ ] Create a release bundle and `configs/eval/release_anchors/rse_v5_osm_assisted.json`; obtain a versioned Zenodo DOI and DOI-matched registry URL, verify fetched bytes equal the recorded registry hash, then run `scripts/report/admit_registered_results.py` to emit `configs/eval/release_admissions/rse_v5_osm_assisted.json`. Aggregation/bootstrap/manuscript rendering must reference this admission record before treating evidence as paper eligible. Until then, retain all v5 result reports as internal preliminary evidence.
- [ ] Audit provenance, counts, thresholds, confidence intervals, and visualizations; update, commit, and push the results ledger after each completed family.

### Task 7: Write the RSE manuscript from sealed evidence

**Files:**
- Modify: `docs/paper/rse_2026/manuscript_outline_en.md`
- Create: `docs/paper/rse_2026/manuscript_rse_en.md`
- Create: `docs/paper/rse_2026/figures/`

**Interfaces:**
- Consumes only the v5 results ledger, audited figures, and the existing RSE journal requirements.
- Produces an English RSE-ready manuscript with abstract, methods, results, statistical evidence, limitations, data/code statement, and references.

- [ ] Populate tables/figures only from release-admitted v5 results; label prior v4 numbers as excluded diagnostics and label OSM-assisted results as weak-supervision spatial readouts.
- [ ] Check every numerical claim against the results ledger and every visual against its provenance manifest.
- [ ] Obtain independent scientific-method and manuscript-format reviews; address all P0/P1/P2 findings, render PDF, and run a final evidence audit.
