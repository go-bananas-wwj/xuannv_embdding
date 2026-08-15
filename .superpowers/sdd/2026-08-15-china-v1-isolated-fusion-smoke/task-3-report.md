# Task 3 report: deterministic synthetic annual contexts and registry separation

## Delivered

- Added `experiments/china_v1_fusion_smoke/synthetic.py` with
  `SyntheticAnnualContext` and deterministic annual S2-derived AEF/high-resolution context generation.
- Added `experiments/china_v1_fusion_smoke/registry.py` with separate smoke and formal validators.
- Added focused contract tests in `tests/isolated_smoke/test_synthetic.py` and
  `tests/isolated_smoke/test_registry.py`.

## TDD evidence

1. RED was captured before production implementation:

   ```text
   ModuleNotFoundError: No module named 'experiments.china_v1_fusion_smoke.synthetic'
   ```

   Command: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke/test_synthetic.py -q`

2. Fresh GREEN verification after the final commit:

   ```text
   7 passed in 8.51s
   ```

   Command: `PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest tests/isolated_smoke/test_synthetic.py tests/isolated_smoke/test_registry.py -q`

3. Style/static checks before commit:

   ```text
   black --check: 4 files would be left unchanged
   ruff check: All checks passed
   git diff --check: exit 0
   ```

## Contract evidence

- Determinism is independent of Python's randomized `hash()`: a SHA-256 digest of
  `seed|patch_id|year` initializes a CPU `torch.Generator`, which drives both the 64×10
  projection and texture.
- The generator accepts only CPU, `float32` S2 `[4,3,10,128,128]` and boolean valid masks
  `[4,3,1,128,128]`. It averages only valid observations, leaves invalid annual pixels masked,
  and produces AEF `[64,128,128]` and highres `[3,640,640]`.
- The generated AEF is normalized over its 64-channel dimension; the test checks every valid
  vector's norm equals one within `1e-5` absolute/relative tolerance.
- Metadata explicitly labels AEF as `annual_s2_fixed_projection` and highres as
  `annual_s2_rgb_5x_deterministic_texture`. Both require `synthetic=true`,
  `allowed_use=smoke_test_only`, and prohibit formal training/evaluation. Highres additionally
  states `claimed_native_gsd_m=null`, `model_input_gsd_m=2`, and
  `contains_real_2m_information=false`.
- Registry comparisons verify both value and type, so values such as `1` and `0` do not satisfy
  boolean policy fields. The formal validator rejects `synthetic is True` with
  `synthetic data is forbidden in formal registries`.

## Self-review

- No production `src/` package files were changed.
- No runtime filesystem writes were added; the new generator is pure tensor computation.
- No `einops` dependency or import was introduced.
- Scope matches the Task 3 brief; no claim treats the generated side inputs as official AEF or
  real 2 m data.

## Version control

- Implementation commit: `371c86c feat: add synthetic annual side inputs with formal-use guards`
- Pushed to `origin/codex/china-v1-fusion-smoke`.
