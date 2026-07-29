# Harbin Frozen-Transfer Multihead Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate the valid frozen Haidian P10C-to-Harbin transfer against AEF with matched downstream heads, and separately quantify P10C's modality-mismatch sensitivity.

**Architecture:** A lock-driven matrix reuses the existing 380-patch coverage inventory, spatial folds, label-ID map and deterministic 5/10-shot schedules. The primary table contains only frozen P10C and AEF; a P10C-only high-resolution-mask diagnostic is reported separately.

**Tech Stack:** Python 3.11, PyTorch, torch_npu, existing Harbin exports and downstream head factory.

## Global Constraints

- Primary families are exactly `p10c_haidian_frozen_harbin` and `aef_annual_2025`; exclude the Harbin scratch family.
- Reuse the locked 380 patches, five spatial folds, 5/10-shot schedules, three seeds, support-only standardizer, and validation-only threshold selection.
- Each reader uses identical architecture and hyperparameters for frozen P10C and AEF.
- Linear/MLP/Conv3x3 are representation probes; UNet/DeepLab-lite are high-capacity upper bounds.
- The AEF annual-2025 versus Xuannv monthly-2026-04 time-context difference remains explicit.
- The high-resolution masking diagnostic is P10C-only and never modifies AEF inputs.
- New data artifacts go under a new `/data/xuannv_embedding/outputs/` root.

---

### Task 1: Freeze matched-reader contract

**Files:**
- Create: `configs/eval/harbin_frozen_transfer_multihead_20260729.json`
- Create: `scripts/eval/run_harbin_frozen_transfer_multihead.py`
- Create: `tests/test_harbin_frozen_transfer_multihead.py`

**Interfaces:** `prepare_multihead_matrix(config_path, output_root) -> PreparedMultihead` validates family seals and shared schedules.

- [ ] Write failing tests:

```python
def test_primary_matrix_rejects_harbin_scratch_family(tmp_path):
    with pytest.raises(ValueError, match="scratch"):
        module.prepare_multihead_matrix(config_with_scratch, tmp_path)

def test_primary_matrix_has_matched_readers(tmp_path):
    prepared = module.prepare_multihead_matrix(config_path, tmp_path)
    assert prepared.heads == ("linear", "wide_mlp", "deep_wide_mlp", "conv3x3", "unet", "deeplab_lite")
```

- [ ] Run `python -m pytest tests/test_harbin_frozen_transfer_multihead.py -q`; expect RED.
- [ ] Implement the locked family/head/schedule contract and configuration.
- [ ] Run the same test; expect GREEN.
- [ ] Commit: `feat: lock Harbin frozen transfer multihead protocol` and push.

### Task 2: Implement and run matched readers

**Files:**
- Modify: `scripts/eval/run_harbin_frozen_transfer_multihead.py`
- Modify: `tests/test_harbin_frozen_transfer_multihead.py`

**Interfaces:** `run_reader_cell(prepared, family, task, fold, shot, seed, head) -> dict[str, float]` writes an atomic metrics/prediction bundle.

- [ ] Write failing tests:

```python
def test_linear_reader_has_no_spatial_kernel():
    assert module.build_reader("linear", 64).kernel_size == (1, 1)

def test_result_is_published_only_after_metrics_and_predictions_exist(tmp_path):
    module.write_cell_atomically(tmp_path, metrics, predictions)
    assert (tmp_path / "metrics.json").is_file()
```

- [ ] Run the test file; expect RED.
- [ ] Implement linear, `wide_mlp`, `deep_wide_mlp`, `conv3x3`, `unet`, and `deeplab_lite` reader cells using the existing head factory. Fit the standardizer and reader from the support split only; choose threshold only from validation.
- [ ] Run tests; expect GREEN.
- [ ] Launch six-NPU matrix: `2 families x 3 tasks x 5 folds x 2 shots x 3 seeds x 6 heads = 1,080` cells in a new root.
- [ ] Export a separate P10C-only high-resolution-masked diagnostic and commit/push the runner.

### Task 3: Audit and report corrected transfer evidence

**Files:**
- Create: `scripts/eval/audit_harbin_frozen_transfer_multihead.py`
- Create: `tests/test_harbin_frozen_transfer_multihead_audit.py`
- Create: `docs/paper/rse_2026/harbin_frozen_transfer_multihead_20260729.md`
- Create: `docs/production/assets/harbin_frozen_transfer_multihead_20260729/harbin_frozen_transfer_multihead_f1.png`

**Interfaces:** `audit_multihead_results(results_root, config_path)` rejects missing paired cells, scratch-family contamination, non-validation thresholds, and non-atomic outputs.

- [ ] Write a failing test that places one scratch-family result in a primary root and expects `ValueError`.
- [ ] Run `python -m pytest tests/test_harbin_frozen_transfer_multihead_audit.py -q`; expect RED.
- [ ] Implement audit, CSV aggregate, figure, and report. Include per-head F1/AP/AUC mean±std and a separate P10C modality diagnostic table.
- [ ] Run both multihead test files; expect GREEN.
- [ ] Commit: `docs: report corrected Harbin frozen transfer multihead results` and push.
