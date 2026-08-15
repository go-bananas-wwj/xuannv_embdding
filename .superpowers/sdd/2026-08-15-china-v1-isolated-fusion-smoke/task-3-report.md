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

## Fix round 1: adversarial input and registry hardening

### Root cause and RED evidence

The review found four independent contract escapes in `371c86c`:

1. `observations * valid` retains `NaN` because IEEE multiplication does not turn `NaN * 0`
   into zero.
2. A zero annual S2 vector projects to an all-zero AEF vector, while the old AEF mask used only
   source-observation availability.
3. The smoke validator applied highres requirements only when `synthetic_kind` already exactly
   matched the highres kind, allowing missing and unknown kinds through.
4. The formal validator checked only `synthetic`, allowing an otherwise smoke-only registry to
   flip that field to `false`.

After adding regression tests, the exact command below produced 9 expected failures and 8 passes:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_synthetic.py tests/isolated_smoke/test_registry.py -q
```

```text
..FFF.....FF.FFFF
9 failed, 8 passed in 8.44s
```

The failing test coverage was:

- `test_invalid_nan_observations_do_not_contaminate_annual_context`
- `test_all_invalid_nan_observations_produce_finite_outputs_and_false_masks`
- `test_degenerate_zero_projection_is_not_marked_as_valid_aef`
- `test_smoke_registry_requires_a_recognized_kind_and_all_disclaimers` (missing kind, unknown
  kind, and missing highres native-GSD disclaimer)
- `test_formal_registry_rejects_smoke_only_declarations_with_false_synthetic_bit` (smoke use,
  training/evaluation disallowance, and synthetic kind)

### Fix and GREEN evidence

- Annual aggregation now combines the supplied mask with per-observation finite checks and uses
  `torch.where` to replace invalid values before summing. All-invalid and partially-invalid NaN
  inputs now yield finite AEF/highres tensors and correctly false masks.
- `aef_valid` now requires both annual source validity and projection norm `> 1e-6`; invalid or
  degenerate vectors are explicitly exported as zeros. Highres validity continues to represent
  annual source validity rather than AEF-vector validity.
- Smoke registries now require one recognized kind: `annual_s2_fixed_projection` or
  `annual_s2_rgb_5x_deterministic_texture`. The latter must have all exact 2 m disclaimers.
- Formal registries reject known synthetic kinds, `smoke_test_only`, and false formal-use flags
  even when `synthetic` is false; malformed formal-use flag types are rejected too.

Fresh focused verification after the fixes:

```bash
PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD python -m pytest \
  tests/isolated_smoke/test_synthetic.py tests/isolated_smoke/test_registry.py -q
```

```text
17 passed in 10.30s
```

Formatting/static verification:

```text
black: 4 files left unchanged
ruff check: All checks passed
git diff --check: exit 0
```
