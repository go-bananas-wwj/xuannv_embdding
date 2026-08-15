# Task 4 report: reversible tiny quarterly fusion model

## Delivered

- Added `experiments/china_v1_fusion_smoke/model.py` with the isolated
  `IsolatedFusionSmokeModel` and `FusionOutput` contract.
- Added `tests/isolated_smoke/test_model.py` with four-way branch coverage, vMF norm checks,
  zero-gate identity, explicit branch skipping, gradient smoke, and the real 5× high-resolution
  downsampling contract.
- Kept the model under `experiments/`; no production model registration or `src/` model changes
  were made.

## TDD evidence

### RED

The branch-matrix/vMF test was created before the model module. The required focused command
failed for the expected missing-feature reason:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_model.py -q
```

```text
ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.model'
1 error in 11.98s
```

### GREEN

After the minimum implementation, the initial four branch cases passed. Zero-gate/skip tests
then brought the focused suite to 6 passing tests, and the gradient/high-resolution checks
brought it to 8 passing tests.

Fresh final regression verification:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_model.py tests/test_model.py -q
```

```text
55 passed in 14.23s
```

Fresh style/static verification:

```text
ruff check: All checks passed
black --check: 2 files would be left unchanged
git diff --check: exit 0
```

## Shape, norm, identity, and gradient evidence

The fixed-seed `B=2`, `H=W=16` evidence run produced:

```text
shape=(2, 4, 64, 16, 16)
max_norm_error=2.98023224e-07
max_identity_pre_vmf_diff=0
max_identity_embedding_diff=0
aef_adapter_grad_l1=0.0506814502
highres_stem_grad_l1=0.00731668575
highres_adapter_grad_l1=0.0860755993
output_projection_grad_l1=0.528922766
highres_kernel_stride=(5, 5)/(5, 5)
```

- All four `(use_aef, use_highres)` combinations preserve `[B,4,64,H,W]` and unit vectors.
- Independent scalar parameters `aef_gate` and `highres_gate` are zero-initialized and bounded by
  `tanh` during normal execution. With both gates at zero, Full and Base are exactly identical in
  both `pre_vmf` and normalized embeddings for the evidence input.
- `gate_override=0.1` constructs device/dtype-local scalar coefficients without mutating either
  gate parameter. With the sensor stems frozen, both side paths and the 1×1 output projection
  receive finite, nonzero gradients.
- High-resolution input `[2,3,80,80]` enters the first high-resolution convolution directly and
  exits it as `[2,32,16,16]`; the convolution uses `kernel_size=5`, `stride=5`.
- Disabled branches accept `None` for their tensors and masks and are not evaluated.

## Implementation notes

- S2 and S1 use separate native-PyTorch `Conv2d → GELU → Conv2d` stems followed by pixelwise
  valid-month weighted quarterly means.
- The concatenated sensor features pass through a 1×1 output projection.
- AEF uses `64→16`, depthwise 3×3 processing, then `16→64`.
- High-resolution data begins with the required 5×5/stride-5 `3→32` convolution and then uses a
  lightweight depthwise adapter to 64 channels.
- Annual deltas are expanded at runtime with `delta[:, None].expand(...)`; they are not copied or
  stored four times.
- No `einops` dependency was introduced.

## Files

- `experiments/china_v1_fusion_smoke/model.py`
- `tests/isolated_smoke/test_model.py`
- `.superpowers/sdd/2026-08-15-china-v1-isolated-fusion-smoke/task-4-report.md`

## Self-review

- Reviewed the final diff against every Task 4 interface and step.
- Mutation check: wrong branch enablement, nonzero initial gates, placeholder access, omitted vMF,
  missing side gradients, or pre-resizing before the 5× convolution each causes at least one
  focused test to fail.
- Input masks are applied before convolutions to prevent masked nonfinite values from contaminating
  features; annual deltas are masked again at their output grid.
- The change is isolated and does not modify `AEFModel`; all 47 existing production-model tests in
  `tests/test_model.py` remain included in the 55-test GREEN run.

## Version control

- Implementation commit: `4cb8a5f feat: add reversible tiny quarterly fusion smoke model`
- Implementation pushed to `origin/codex/china-v1-fusion-smoke`.
