# Harbin PU+Query Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether P10C's Harbin transfer weakness is caused primarily by its frozen Conv3x3 readout or by non-transferable embedding semantics, using a paired prototype-based PU+Query evaluation against AEF.

**Architecture:** Add a Harbin-specific, lock-driven evaluator that reuses the proven PU+Query scoring recipe while consuming the sealed 380-patch P10C/AEF exports and their one-to-one label-ID map. It will create shared polygon supports per spatial fold, evaluate single and max multi-prototype modes with Query disabled/enabled, and keep validation calibration distinct from test metrics.

**Tech Stack:** Python 3.11, NumPy, PyTorch, rasterio, scipy, scikit-learn, existing P10C/AEF embedding exports.

## Global Constraints

- Use exactly the 380-patch Harbin coverage lock and spatial five-fold split from `harbin_strict_conv3x3_matrix_20260728.json`.
- Compare frozen Haidian P10C, Harbin-scratch P10C, and AEF under identical supports, labels, folds, seeds, prototype modes, and metrics.
- Polygon count (`1/3/5/9`) is a distinct prompt budget; never mix it with the Conv3x3 `5/10` patch-shot table.
- Select every reported threshold on the corresponding validation split only; test labels must not affect supports, normalization, prototypes, background selection, thresholding, or Query gating.
- Preserve the output resolution at 128x128 and write large artifacts only under `/data/xuannv_embedding/`.
- Add tests first, run them before implementation, commit and push each completed code/document step.

---

### Task 1: Freeze paired prompt schedules and evaluator contract

**Files:**
- Create: `tests/test_harbin_pu_query_transfer.py`
- Create: `configs/eval/harbin_pu_query_transfer_20260729.json`
- Create: `scripts/eval/run_harbin_pu_query_transfer.py`

**Interfaces:**
- Consumes: `harbin_strict_conv3x3_matrix_20260728.json`, its coverage inventory, its spatial split, and sealed embedding roots.
- Produces: `prepare_harbin_pu_query(config_path, output_root) -> PreparedProtocol`, with fixed patch IDs, `harbin_patch_* -> patch_*` label mapping, and reusable polygon schedules.

- [ ] **Step 1: Write failing tests**

```python
def test_prepare_rejects_non_380_coverage(tmp_path):
    with pytest.raises(ValueError, match="380"):
        module.prepare_harbin_pu_query(bad_config, tmp_path)

def test_polygon_schedule_is_identical_for_all_families(tmp_path):
    prepared = module.prepare_harbin_pu_query(config_path, tmp_path)
    assert prepared.schedule_for("building", 0, 42, 5) == prepared.schedule_for("building", 0, 42, 5)

def test_label_resolution_uses_locked_source_patch_id(tmp_path):
    assert module.resolve_label_id("harbin_patch_000001", mapping) == "patch_000001"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_harbin_pu_query_transfer.py -q`

Expected: FAIL because the evaluator module and its contract functions do not exist.

- [ ] **Step 3: Implement the minimum lock and schedule layer**

```python
def resolve_label_id(patch_id: str, mapping: dict[str, str]) -> str:
    if patch_id not in mapping:
        raise KeyError(f"unlocked label mapping for {patch_id}")
    return mapping[patch_id]

def prepare_harbin_pu_query(config_path: Path, output_root: Path) -> PreparedProtocol:
    # Validate coverage, family seals, labels and folds before sampling polygons.
    ...
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_harbin_pu_query_transfer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add configs/eval/harbin_pu_query_transfer_20260729.json \
  scripts/eval/run_harbin_pu_query_transfer.py tests/test_harbin_pu_query_transfer.py
git commit -m "feat: add locked Harbin PU query transfer protocol"
git push origin HEAD:codex/p10c-harbin-transfer
```

### Task 2: Implement validation-calibrated PU+Query scoring

**Files:**
- Modify: `scripts/eval/run_harbin_pu_query_transfer.py`
- Modify: `tests/test_harbin_pu_query_transfer.py`

**Interfaces:**
- Consumes: `PreparedProtocol`, a sealed family embedding root, task/fold/seed/polygon budget, and prototype mode.
- Produces: `evaluate_cell(...) -> dict[str, float]` with F1, AP, AUC-ROC, selected validation threshold, and optional Query result.

- [ ] **Step 1: Write failing tests**

```python
def test_test_labels_do_not_change_validation_threshold():
    assert module.select_validation_threshold(scores, labels) == expected_threshold

def test_max_prototype_scores_best_matching_support_prototype():
    assert module.aggregate_foreground_similarity(pixels, prototypes, "max").tolist() == [1.0, 1.0]

def test_query_disabled_preserves_base_scores():
    scores, adapted = module.score_pu_query(feature, model, query_mode="disabled")
    assert not adapted
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_harbin_pu_query_transfer.py -q`

Expected: FAIL because validation-only thresholding and paired scoring are not yet implemented.

- [ ] **Step 3: Implement scoring and archives**

```python
def evaluate_cell(prepared, family, task, fold, seed, polygon_count, prototype_mode, query_mode):
    # Fit support-only normalization/prototypes/background, select threshold on validation,
    # then compute test F1/AP/AUC without reading test labels during inference.
    ...
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_harbin_pu_query_transfer.py -q`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/eval/run_harbin_pu_query_transfer.py tests/test_harbin_pu_query_transfer.py
git commit -m "feat: score Harbin PU query transfer with validation calibration"
git push origin HEAD:codex/p10c-harbin-transfer
```

### Task 3: Run, audit, and report the paired experiment

**Files:**
- Create: `scripts/eval/audit_harbin_pu_query_transfer.py`
- Create: `tests/test_harbin_pu_query_transfer_audit.py`
- Create: `docs/paper/rse_2026/harbin_pu_query_transfer_20260729.md`
- Create: `docs/production/assets/harbin_pu_query_transfer_20260729/harbin_pu_query_transfer_f1.png`

**Interfaces:**
- Consumes: one result archive per family/task/fold/seed/polygon-count/prototype/query cell.
- Produces: an auditable aggregate CSV, Markdown report, and F1 figure; separate base and Query-enabled rows.

- [ ] **Step 1: Write failing audit test**

```python
def test_audit_rejects_missing_paired_family_cell(tmp_path):
    with pytest.raises(ValueError, match="missing paired cell"):
        module.audit_results(partial_root, config_path, tmp_path)
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_harbin_pu_query_transfer_audit.py -q`

Expected: FAIL because the audit module does not exist.

- [ ] **Step 3: Implement audit, run the full matrix, and create report**

```bash
python scripts/eval/run_harbin_pu_query_transfer.py \
  --config configs/eval/harbin_pu_query_transfer_20260729.json \
  --output-root /data/xuannv_embedding/outputs/paper_p10c_haidian_harbin_20260728/harbin_pu_query_transfer
python scripts/eval/audit_harbin_pu_query_transfer.py ...
```

- [ ] **Step 4: Run audit test and verify GREEN**

Run: `python -m pytest tests/test_harbin_pu_query_transfer.py tests/test_harbin_pu_query_transfer_audit.py -q`

Expected: PASS, with a complete paired result grid and no use of test labels during inference.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/eval/audit_harbin_pu_query_transfer.py \
  tests/test_harbin_pu_query_transfer_audit.py \
  docs/paper/rse_2026/harbin_pu_query_transfer_20260729.md \
  docs/production/assets/harbin_pu_query_transfer_20260729
git commit -m "docs: report Harbin PU query transfer comparison"
git push origin HEAD:codex/p10c-harbin-transfer
```
