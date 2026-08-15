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

## Fix Round 1: preserve zero-gate identity under AMP

### Finding and root cause

CPU BF16 autocast reproduced the NPU AMP risk. The base projection and both annual adapters
produced BF16 tensors, but the AEF and downsampled high-resolution masks were cast to their raw
input dtype, FP32. Multiplying BF16 adapter deltas by those FP32 masks promoted the deltas and
therefore Full `pre_vmf` to FP32. Base remained BF16. Although zero-gated `pre_vmf` values matched
after an explicit cast, vMF normalization ran at different precisions and changed embeddings.

The diagnostic run reported:

```text
component_dtypes {'base_projection': torch.bfloat16, 'aef_adapter': torch.bfloat16,
                  'highres_adapter': torch.bfloat16}
parameter_gate_dtype torch.float32
raw_mask_dtype torch.float32
base torch.bfloat16 torch.bfloat16
full torch.float32 torch.float32
max_pre_vmf_diff_after_cast 0.0
max_embedding_diff_after_cast 0.0020360350608825684
```

A separate scalar promotion check confirmed that the zero-dimensional FP32 gate does not promote
a BF16 tensor, while multiplying by an FP32 mask does. The minimal root-cause fix therefore casts
each annual mask to its computed branch delta's dtype and device after the adapter.

### RED

The BF16 autocast regression was added before the model fix:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_model.py::test_bfloat16_autocast_zero_gate_full_matches_base_dtype_and_values \
  -q
```

```text
F                                                                        [100%]
E       AssertionError: assert torch.float32 == torch.bfloat16
1 failed in 8.82s
```

The failure occurred at `assert full.pre_vmf.dtype == base.pre_vmf.dtype`, proving the regression
detects the reported AMP promotion rather than an unrelated error.

### GREEN

After the two post-adapter mask casts:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_model.py::test_bfloat16_autocast_zero_gate_full_matches_base_dtype_and_values \
  -q
```

```text
.                                                                        [100%]
1 passed in 10.52s
```

The regression asserts Base and Full `pre_vmf` dtypes match, Base remains BF16, embedding dtypes
match, and both `pre_vmf` and embedding values are exactly equal with `atol=0`, `rtol=0`.

Fresh focused plus production-model regression after formatting:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_model.py tests/test_model.py -q
```

```text
........................................................                 [100%]
56 passed in 22.23s
```

Fresh style/static verification:

```text
ruff check: All checks passed
black --check: 2 files would be left unchanged
git diff --check: exit 0
```

### Scope and version control

- Changed only `experiments/china_v1_fusion_smoke/model.py` and
  `tests/isolated_smoke/test_model.py` for the fix.
- The shape-validation Low finding remains deferred as directed; no unrelated validation or API
  changes were added.
- Fix commit: `09502f3 fix: preserve fusion dtype under autocast`
- Fix pushed to `origin/codex/china-v1-fusion-smoke`.
