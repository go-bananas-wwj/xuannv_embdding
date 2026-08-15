# China V1 Isolated Three-Branch Fusion Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立 worktree 和 `/data` 沙箱中，用 4 个真实 2020—2021 S1/S2 patch、模拟年度 AEF 和模拟年度 2 m 影像，完成单张 Ascend NPU 上的三路融合前向、反向、恢复和八季度导出冒烟测试。

**Architecture:** 新代码只存在于独立 `experiments/china_v1_fusion_smoke` 包，不注册、不修改现有玄女模型。真实月度 S1/S2 从 `/data2` ZIP 只读加载；年度 AEF 和 2 m 由确定性生成器写入专用沙箱；极小季度基座、两个低秩旁路和零初始化门控输出 `4×8×64×128×128`，并由路径守卫和 synthetic registry 双重阻止模拟数据进入正式流程。

**Tech Stack:** Python 3.11、PyTorch 2.6、torch_npu 2.6、CANN 9.0.0、Rasterio MemoryFile、Zarr v2、Numcodecs、PyYAML、pytest、Bash。

## Global Constraints

- 只在 `/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke` 和分支 `codex/china-v1-fusion-smoke` 修改代码。
- 所有运行产物只写 `/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815`。
- `/data2/china_xuannv_embedding/data` 永远只读；不得创建锁、索引、临时文件或完成标记。
- 只使用物理 NPU 2；启动器固定 `ASCEND_RT_VISIBLE_DEVICES=2`，进程内设备为逻辑 `npu:0`。
- 固定 `max_patches=4`、`years=[2020,2021]`、`max_optimizer_steps=2`。
- 模拟 AEF 和模拟 2 m 必须同时带 `synthetic=true`、`formal_training_allowed=false`、`formal_evaluation_allowed=false`。
- 正式 registry validator 必须拒绝 synthetic 数据；冒烟 validator 必须拒绝没有 synthetic 标记的旁路数据。
- 不修改现有 `src/xuannv_embedding/` 模型、Dataset、生产配置、checkpoint 或 Python 基础环境。
- 禁止 `einops`；所有 reshape/permute 使用 PyTorch 原生操作。
- 每个实际实验 YAML 自包含，不使用 `_base_`。
- 每个 Task 完成并验证后，单独 commit 并立即 push。

---

## File Map

| 文件 | 单一职责 |
|---|---|
| `configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml` | 固定输入、沙箱、设备、规模和 seed 的自包含配置 |
| `experiments/china_v1_fusion_smoke/config.py` | 加载和严格校验 smoke YAML |
| `experiments/china_v1_fusion_smoke/safety.py` | 路径、sentinel、规模和设备安全守卫 |
| `experiments/china_v1_fusion_smoke/archive.py` | 月度 ZIP 索引、稳定选择 4 patch、只读 TIFF 解码 |
| `experiments/china_v1_fusion_smoke/data.py` | S2/S1 缩放、季度组织和 batch 数据类 |
| `experiments/china_v1_fusion_smoke/synthetic.py` | 确定性生成模拟 AEF 与模拟 2 m |
| `experiments/china_v1_fusion_smoke/registry.py` | synthetic 冒烟 registry 与正式拒绝规则 |
| `experiments/china_v1_fusion_smoke/model.py` | 极小季度基座、AEF/2 m adapter、门控和 vMF |
| `experiments/china_v1_fusion_smoke/export.py` | Zarr、checkpoint、JSON manifest 和 SUCCESS 封口 |
| `experiments/china_v1_fusion_smoke/runner.py` | inspect、prepare、CPU、NPU 四阶段编排 |
| `scripts/smoke/setup_china_v1_isolated_fusion_env.sh` | 创建轻量 venv 和沙箱 |
| `scripts/smoke/run_china_v1_isolated_fusion_smoke.sh` | CANN、物理 NPU 2、路径和 provenance 启动守卫 |
| `tests/isolated_smoke/` | 与上述单元边界一一对应的测试 |
| `docs/reports/china_v1_isolated_fusion_smoke_20260815.md` | 实际运行证据和结论 |

---

### Task 1: Sandbox Safety, Config, and Environment Bootstrap

**Files:**

- Create: `configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml`
- Create: `experiments/__init__.py`
- Create: `experiments/china_v1_fusion_smoke/__init__.py`
- Create: `experiments/china_v1_fusion_smoke/config.py`
- Create: `experiments/china_v1_fusion_smoke/safety.py`
- Create: `scripts/smoke/setup_china_v1_isolated_fusion_env.sh`
- Test: `tests/isolated_smoke/test_safety_config.py`

**Interfaces:**

- Produces: `SmokeConfig`, `load_smoke_config(path: Path) -> SmokeConfig`.
- Produces: `ensure_sandbox(root: Path) -> Path`, `validate_write_path(path: Path, sandbox_root: Path) -> Path`.
- Produces: `validate_limits(config: SmokeConfig) -> None` and sentinel `.xuannv_isolated_smoke`.

- [ ] **Step 1: Write the failing config and path-guard tests**

```python
from pathlib import Path

import pytest

from experiments.china_v1_fusion_smoke.config import load_smoke_config
from experiments.china_v1_fusion_smoke.safety import SafetyError, validate_write_path


def test_checked_config_is_fixed_to_four_patches_two_years_and_npu2():
    cfg = load_smoke_config(
        Path("configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml")
    )
    assert cfg.max_patches == 4
    assert cfg.years == (2020, 2021)
    assert cfg.physical_npu == 2
    assert cfg.max_optimizer_steps == 2


@pytest.mark.parametrize(
    "unsafe",
    [
        "/root/workspace/xuannv",
        "/data/xuannv_embedding/processed",
        "/data/xuannv_embedding/outputs",
        "/data/xuannv_embedding/embeddings",
        "/data2/china_xuannv_embedding/data",
    ],
)
def test_write_guard_rejects_production_roots(unsafe: str, tmp_path: Path):
    root = tmp_path / "china_v1_fusion_smoke_20260815"
    root.mkdir()
    (root / ".xuannv_isolated_smoke").touch()
    with pytest.raises(SafetyError):
        validate_write_path(Path(unsafe), root)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_safety_config.py -q`

Expected: FAIL during import because the isolated smoke package does not exist.

- [ ] **Step 3: Add the exact self-contained YAML**

```yaml
experiment:
  name: china_v1_isolated_fusion_smoke_20260815
  seed: 20260815
  use_wandb: false
sandbox:
  root: /data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
  source_root: /data2/china_xuannv_embedding/data
  sentinel: .xuannv_isolated_smoke
runtime:
  physical_npu: 2
  logical_device: npu:0
  max_patches: 4
  years: [2020, 2021]
  max_optimizer_steps: 2
  num_workers: 0
data:
  s2_scale: 0.0001
  s1_epsilon: 1.0e-6
  s1_source_order: [vh, vv]
  s1_model_order: [vv, vh]
model:
  embed_dim: 64
  aef_hidden_dim: 16
  highres_hidden_dim: 32
  gate_init: 0.0
  gradient_smoke_gate: 0.1
export:
  dtype: float16
  zarr_version: 2
  periods: [2020Q1, 2020Q2, 2020Q3, 2020Q4, 2021Q1, 2021Q2, 2021Q3, 2021Q4]
```

- [ ] **Step 4: Implement strict parsing and path guards**

Use frozen dataclasses. Reject `_base_`, unknown/missing sections, device other than 2, wrong years, larger scale, and paths outside the exact source/sandbox roots.

```python
class SafetyError(RuntimeError):
    pass


def validate_write_path(path: Path, sandbox_root: Path) -> Path:
    root = sandbox_root.resolve(strict=True)
    if not (root / ".xuannv_isolated_smoke").is_file():
        raise SafetyError("isolated smoke sentinel is missing")
    resolved = path.resolve(strict=False)
    if resolved == root or resolved.is_relative_to(root):
        return resolved
    raise SafetyError(f"write path escapes sandbox: {resolved}")
```

- [ ] **Step 5: Implement the environment bootstrap**

The script must create the exact sandbox, use `python -m venv --system-site-packages`, never call `pip install`, create all subdirectories, and print created paths.

```bash
#!/usr/bin/env bash
set -euo pipefail
SANDBOX=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
mkdir -p "${SANDBOX}"
touch "${SANDBOX}/.xuannv_isolated_smoke"
python -m venv --system-site-packages "${SANDBOX}/env"
mkdir -p "${SANDBOX}"/{cache,synthetic/aef,synthetic/highres_2m,outputs,checkpoints,logs,manifests,tmp}
```

- [ ] **Step 6: Verify GREEN and the existing smoke baseline**

Run:

```bash
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_safety_config.py tests/test_smoke.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit and push**

```bash
git add configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
  experiments scripts/smoke/setup_china_v1_isolated_fusion_env.sh \
  tests/isolated_smoke/test_safety_config.py
git commit -m "feat: add isolated China V1 smoke sandbox contract"
git push
```

---

### Task 2: Read-Only ZIP Selection and Quarterly Tensor Contract

**Files:**

- Create: `experiments/china_v1_fusion_smoke/archive.py`
- Create: `experiments/china_v1_fusion_smoke/data.py`
- Test: `tests/isolated_smoke/test_archive.py`
- Test: `tests/isolated_smoke/test_data_contract.py`

**Interfaces:**

- Produces: `MonthRef(sensor, year, month, archive, member, crc)`.
- Produces: `PatchSelection(patch_id, months)` with 48 S2/S1 month references.
- Produces: `select_complete_patches(source_root: Path, count: int = 4) -> tuple[PatchSelection, ...]`.
- Produces: `inspect_selection_headers(selection: PatchSelection) -> tuple[RasterHeader, ...]`.
- Produces: `PatchYearBatch` with S2 `[B,4,3,10,128,128]` and S1 `[B,4,3,2,128,128]`, where `B=8` for 4 patches × 2 years.

- [ ] **Step 1: Write failing ZIP fixture tests**

```python
def test_selector_returns_first_four_complete_patch_ids(fake_archive_root: Path):
    selected = select_complete_patches(fake_archive_root, count=4)
    assert [item.patch_id for item in selected] == [
        "parent_32643:310:3383",
        "parent_32643:311:3390",
        "parent_32643:312:3346",
        "parent_32643:313:3352",
    ]
    assert all(len(item.months) == 48 for item in selected)
```

Fixtures must copy the real member convention `pc-s2/2020/01/{patch_id}.tif`. Also test missing-month rejection, CRC capture and lexical stability.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_archive.py -q`

Expected: FAIL because `select_complete_patches` does not exist.

- [ ] **Step 3: Implement ZIP intersection without extraction**

Use exact paths:

```python
archive = source_root / sensor / str(year) / f"{month:02d}" / (
    f"{sensor}_{year}_{month:02d}.zip"
)
```

Intersect member stems across `pc-s2` and `pc-s1`, years 2020/2021 and months 01—12. Sort before taking four. Read only `ZipInfo`; never call `extract()` or `extractall()`.

- [ ] **Step 4: Write failing quarterly tensor tests**

```python
def test_patch_year_batch_has_quarter_major_shapes(fake_complete_selection):
    batch = load_patch_year_batch(fake_complete_selection, years=(2020, 2021))
    assert batch.s2.shape == (8, 4, 3, 10, 128, 128)
    assert batch.s1.shape == (8, 4, 3, 2, 128, 128)
    assert batch.valid_s2.shape == (8, 4, 3, 1, 128, 128)
    assert batch.valid_s1.shape == (8, 4, 3, 1, 128, 128)
```

- [ ] **Step 5: Verify RED, then implement decoding and preprocessing**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_data_contract.py -q`

Expected: FAIL because `load_patch_year_batch` does not exist.

Use `ZipFile.open` plus `rasterio.io.MemoryFile`. Validate shape, CRS and transform. Apply:

```python
s2 = raw_s2.astype("float32") * 0.0001
s1_vv_vh = raw_s1.astype("float32")[[1, 0]]
s1_db = 10.0 * np.log10(np.maximum(s1_vv_vh, 1.0e-6))
quarter_index = (month - 1) // 3
```

- [ ] **Step 6: Run tests and real-header inspection**

Run:

```bash
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_archive.py tests/isolated_smoke/test_data_contract.py -q
PYTHONPATH=$PWD python - <<'PY'
from pathlib import Path
from experiments.china_v1_fusion_smoke.archive import (
    inspect_selection_headers,
    select_complete_patches,
)

root = Path("/data2/china_xuannv_embedding/data")
for selection in select_complete_patches(root, count=4):
    headers = inspect_selection_headers(selection)
    print(selection.patch_id, len(selection.months), headers[0])
PY
```

Expected: tests PASS; inspect prints four stable patch IDs and representative headers without source writes.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/china_v1_fusion_smoke/archive.py \
  experiments/china_v1_fusion_smoke/data.py \
  tests/isolated_smoke/test_archive.py tests/isolated_smoke/test_data_contract.py
git commit -m "feat: add read-only national patch-year smoke loader"
git push
```

### Task 3: Deterministic Synthetic AEF and 2 m with Registry Separation

**Files:**

- Create: `experiments/china_v1_fusion_smoke/synthetic.py`
- Create: `experiments/china_v1_fusion_smoke/registry.py`
- Test: `tests/isolated_smoke/test_synthetic.py`
- Test: `tests/isolated_smoke/test_registry.py`

**Interfaces:**

- Consumes: one patch-year S2 `[4,3,10,128,128]`, valid mask, patch ID, year and seed.
- Produces: `SyntheticAnnualContext(aef, aef_valid, highres, highres_valid, metadata)`.
- Produces: `generate_synthetic_context(s2_year: Tensor, valid_s2_year: Tensor, patch_id: str, year: int, seed: int) -> SyntheticAnnualContext`.
- Produces: `validate_smoke_registry(raw) -> None` and `validate_formal_registry(raw) -> None`.

- [ ] **Step 1: Write failing deterministic-generation tests**

```python
def test_synthetic_context_is_deterministic_and_aligned():
    first = generate_synthetic_context(s2, valid_s2, "patch-a", 2020, seed=20260815)
    second = generate_synthetic_context(s2, valid_s2, "patch-a", 2020, seed=20260815)
    torch.testing.assert_close(first.aef, second.aef)
    torch.testing.assert_close(first.highres, second.highres)
    assert first.aef.shape == (64, 128, 128)
    assert first.highres.shape == (3, 640, 640)
    norms = torch.linalg.vector_norm(first.aef[:, first.aef_valid[0]], dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_synthetic.py -q`

Expected: FAIL because the generator is undefined.

- [ ] **Step 3: Implement deterministic annual contexts**

Average valid S2 over 12 months. Create a fixed `[64,10]` projection with a CPU generator. Derive the per-patch seed from SHA-256 of `20260815|patch_id|year`, never Python `hash()`.

```python
projection = torch.randn((64, 10), generator=generator)
aef = torch.einsum("oc,chw->ohw", projection, annual_s2)
aef = torch.nn.functional.normalize(aef, p=2, dim=0, eps=1e-6)

rgb = annual_s2[[2, 1, 0]].unsqueeze(0)
up = torch.nn.functional.interpolate(
    rgb, size=(640, 640), mode="bicubic", align_corners=False
)
blur = torch.nn.functional.avg_pool2d(up, kernel_size=5, stride=1, padding=2)
texture = torch.randn(up.shape, generator=generator) * 0.002
highres = (up + 0.15 * (up - blur) + texture).clamp(0.0, 1.5).squeeze(0)
```

- [ ] **Step 4: Write failing registry-policy tests**

```python
def test_formal_registry_rejects_synthetic_context():
    raw = {
        "synthetic": True,
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    validate_smoke_registry(raw)
    with pytest.raises(RegistryError, match="synthetic data is forbidden"):
        validate_formal_registry(raw)
```

Also require the highres registry to contain `contains_real_2m_information=false` and `claimed_native_gsd_m=null`.

- [ ] **Step 5: Verify RED, then implement strict validators**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_registry.py -q`

Expected: FAIL because validators are undefined. Implement exact equality checks; permissive truthiness is not allowed.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_synthetic.py tests/isolated_smoke/test_registry.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/china_v1_fusion_smoke/synthetic.py \
  experiments/china_v1_fusion_smoke/registry.py \
  tests/isolated_smoke/test_synthetic.py tests/isolated_smoke/test_registry.py
git commit -m "feat: add synthetic annual side inputs with formal-use guards"
git push
```

---

### Task 4: Reversible Tiny Quarterly Fusion Model

**Files:**

- Create: `experiments/china_v1_fusion_smoke/model.py`
- Test: `tests/isolated_smoke/test_model.py`

**Interfaces:**

- Consumes: S2 `[B,4,3,10,H,W]`, S1 `[B,4,3,2,H,W]`, AEF `[B,64,H,W]`, highres `[B,3,5H,5W]` and masks.
- Produces: `FusionOutput(embedding, pre_vmf, gates)`.
- Produces: `IsolatedFusionSmokeModel.forward(s2: Tensor, s1: Tensor, valid_s2: Tensor, valid_s1: Tensor, aef: Tensor | None, aef_valid: Tensor | None, highres: Tensor | None, highres_valid: Tensor | None, use_aef: bool, use_highres: bool, gate_override: float | None = None) -> FusionOutput`.
- Output: embedding `[B,4,64,H,W]` with unit vectors on valid pixels.

- [ ] **Step 1: Write failing branch-matrix and vMF tests**

```python
@pytest.mark.parametrize(
    ("use_aef", "use_highres"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_all_branches_keep_the_output_contract(use_aef, use_highres):
    output = run_small_model(use_aef=use_aef, use_highres=use_highres)
    assert output.embedding.shape == (2, 4, 64, 16, 16)
    norms = torch.linalg.vector_norm(output.embedding, dim=2)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_model.py -q`

Expected: FAIL because the model is absent.

- [ ] **Step 3: Implement the minimum native-PyTorch model**

Use separate S2/S1 `Conv2d → GELU → Conv2d` stems, valid-month weighted mean and a `1×1` base projection. AEF is `64→16→64`; highres begins with `Conv2d(3,32,kernel_size=5,stride=5)`. Expand annual deltas with `delta[:, None].expand(-1,4,-1,-1,-1)`.

```python
z = base
if use_aef:
    z = z + torch.tanh(aef_gate) * aef_delta[:, None]
if use_highres:
    z = z + torch.tanh(highres_gate) * highres_delta[:, None]
embedding = torch.nn.functional.normalize(z, p=2, dim=2, eps=1e-6)
```

If `gate_override` is provided, use a scalar tensor on `z.device` without mutating gate parameters.

- [ ] **Step 4: Add zero-gate identity and explicit-skip tests**

```python
def test_zero_initialized_full_matches_base():
    model = IsolatedFusionSmokeModel(embed_dim=64).eval()
    base = run_model(model, use_aef=False, use_highres=False).embedding
    full = run_model(model, use_aef=True, use_highres=True).embedding
    torch.testing.assert_close(full, base, atol=1e-6, rtol=1e-6)


def test_disabled_branches_need_no_placeholder_tensors():
    output = model(
        s2=s2, s1=s1, valid_s2=valid_s2, valid_s1=valid_s1,
        aef=None, aef_valid=None, highres=None, highres_valid=None,
        use_aef=False, use_highres=False,
    )
    assert output.embedding.isfinite().all()
```

- [ ] **Step 5: Add gradient-smoke tests**

Run Full with `gate_override=0.1`. Backpropagate `output.pre_vmf.square().mean()` and require finite nonzero gradients on AEF adapter, highres adapter and output projection. The base is frozen by setting `requires_grad_(False)` on base parameters in the test.

- [ ] **Step 6: Verify GREEN and production-model regression**

Run:

```bash
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_model.py tests/test_model.py -q
```

Expected: all tests PASS and existing `AEFModel` behavior is unchanged.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/china_v1_fusion_smoke/model.py tests/isolated_smoke/test_model.py
git commit -m "feat: add reversible tiny quarterly fusion smoke model"
git push
```

### Task 5: Auditable Zarr Export, Checkpoint Reload, and SUCCESS Sealing

**Files:**

- Create: `experiments/china_v1_fusion_smoke/export.py`
- Test: `tests/isolated_smoke/test_export.py`

**Interfaces:**

- Produces: `export_group_zarr(group, embedding, valid, patch_ids, periods, output, sandbox_root) -> Path`.
- Produces: `save_smoke_checkpoint(model, path, metadata) -> str` returning SHA-256.
- Produces: `load_smoke_checkpoint(model, path, expected_metadata) -> None`.
- Produces: `seal_success(sandbox_root, required_files) -> Path`.

- [ ] **Step 1: Write failing export and checkpoint tests**

```python
def test_export_contract_and_checkpoint_reload(tmp_sandbox: Path):
    output = tmp_sandbox / "outputs" / "full" / "embedding.zarr"
    export_group_zarr("full", embedding, valid, patch_ids, periods, output, tmp_sandbox)
    group = zarr.open_group(str(output), mode="r")
    assert group["embedding"].shape == (4, 8, 64, 128, 128)
    assert group["embedding"].dtype == np.dtype("float16")
    assert group.attrs["formal_training_allowed"] is False
```

Test that `seal_success` rejects a missing manifest, partial Zarr, an output outside sandbox, and formal-use true.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_export.py -q`

Expected: FAIL because export functions do not exist.

- [ ] **Step 3: Implement atomic Zarr v2 writing**

Write to `embedding.zarr.partial`, verify arrays and attrs, then rename to `embedding.zarr`. Use chunks `(1,1,64,128,128)` and Blosc Zstd through Numcodecs. Never create one file per quarter.

- [ ] **Step 4: Implement exact checkpoint metadata checks**

Metadata contains Git commit, config SHA-256, selected-patch manifest SHA-256, seed, model class and `synthetic=true`. Refuse load on any mismatch.

- [ ] **Step 5: Implement final sealing**

Require all four Zarr groups, `run_manifest.json`, `metrics.json`, `path_audit.json`, `reproducibility.json` and checkpoint. Write `SUCCESS` last with combined SHA-256 and UTC time.

- [ ] **Step 6: Verify GREEN**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_export.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/china_v1_fusion_smoke/export.py tests/isolated_smoke/test_export.py
git commit -m "feat: add auditable smoke export and success sealing"
git push
```

---

### Task 6: CPU Contract Runner and Real Four-Patch Preparation

**Files:**

- Create: `experiments/china_v1_fusion_smoke/runner.py`
- Test: `tests/isolated_smoke/test_runner.py`

**Interfaces:**

- Consumes all Task 1—5 interfaces.
- Produces CLI stages: `inspect`, `prepare`, `cpu-contract`, `npu-smoke`.
- Produces: `manifests/patch_selection.json`, synthetic AEF/2 m cache and preflight manifests.

- [ ] **Step 1: Write failing CLI-stage and path-audit tests**

```python
def test_cpu_contract_runs_all_groups_inside_sandbox(tmp_config: Path):
    result = run_stage(tmp_config, stage="cpu-contract")
    assert set(result.groups) == {"base", "base_aef", "base_highres", "full"}
    assert result.full_zero_gate_matches_base
    assert all(path.is_relative_to(result.sandbox_root) for path in result.created_paths)
```

Also test invalid stages, missing sentinel, more than four patches, wrong years, changed source snapshot and missing synthetic flags.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_runner.py -q`

Expected: FAIL because the runner is absent.

- [ ] **Step 3: Implement exact CLI orchestration**

```text
python -m experiments.china_v1_fusion_smoke.runner
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml
  --stage inspect|prepare|cpu-contract|npu-smoke
```

- `inspect`: stable selection and representative headers only.
- `prepare`: source size/mtime snapshot, real 4 patch load, synthetic contexts, registries, unchanged-source verification.
- `cpu-contract`: small fixture tensors by default; `--full-shape` enables full-size CPU contract.
- `npu-smoke`: full prepared data; Task 7 launcher is the only supported entry.

- [ ] **Step 4: Implement provenance and path audit**

Walk only the sandbox before/after. Record all created or modified sandbox paths in `path_audit.json`. Record source ZIP size and mtime before/after without hashing hundreds of GB.

- [ ] **Step 5: Verify GREEN**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_runner.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Bootstrap the actual sandbox and run inspect/prepare**

```bash
bash scripts/smoke/setup_china_v1_isolated_fusion_env.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD
SMOKE_PY=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815/env/bin/python
${SMOKE_PY} -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml --stage inspect
${SMOKE_PY} -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml --stage prepare
```

Expected: four fixed patch IDs, 8 patch-years, unchanged source, all writes in sandbox, total below 5 GB.

- [ ] **Step 7: Run the CPU contract and all isolated CPU tests**

```bash
${SMOKE_PY} -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml --stage cpu-contract
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke -q -m "not npu"
```

Expected: all tests PASS; base/full zero-gate identity PASS.

- [ ] **Step 8: Commit and push**

Do not add `/data` artifacts.

```bash
git add experiments/china_v1_fusion_smoke/runner.py tests/isolated_smoke/test_runner.py
git commit -m "feat: add isolated four-patch fusion smoke runner"
git push
```

### Task 7: Physical NPU 2 Launcher and Full-Size Smoke

**Files:**

- Create: `scripts/smoke/run_china_v1_isolated_fusion_smoke.sh`
- Create: `tests/isolated_smoke/test_launcher.py`
- Create: `tests/isolated_smoke/test_npu_smoke.py`

**Interfaces:**

- Consumes: Task 6 `--stage npu-smoke`.
- Produces: four full-size Zarr groups, checkpoint, metrics, reproducibility audit and `SUCCESS`.

- [ ] **Step 1: Write failing static launcher tests**

```python
def test_launcher_is_single_card_and_sandbox_only():
    source = Path("scripts/smoke/run_china_v1_isolated_fusion_smoke.sh").read_text()
    assert "ASCEND_RT_VISIBLE_DEVICES=2" in source
    assert "--stage npu-smoke" in source
    assert "torchrun" not in source
    assert "nohup" not in source
    assert "/data/xuannv_embedding/outputs" not in source
```

Also assert exact worktree, CANN path, `PYTHONNOUSERSITE=1`, `PYTHONPATH=$WORKTREE`, logical `npu:0`, and `WANDB_MODE=disabled`.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_launcher.py -q`

Expected: FAIL because the launcher is absent.

- [ ] **Step 3: Implement foreground launcher and idle guard**

```bash
#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke
SANDBOX=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
if [[ ! -e /dev/davinci2 ]]; then
  echo "NPU 2 device is absent" >&2
  exit 19
fi
if [[ -n "$(fuser /dev/davinci2 2>/dev/null || true)" ]]; then
  echo "NPU 2 is busy; refusing smoke run" >&2
  exit 20
fi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH="${WORKTREE}"
export ASCEND_RT_VISIBLE_DEVICES=2
export XUANNV_SMOKE_DEVICE=npu:0
export WANDB_MODE=disabled
cd "${WORKTREE}"
"${SANDBOX}/env/bin/python" -m experiments.china_v1_fusion_smoke.runner \
  --config configs/smoke/china_v1_isolated_fusion_4patch_20260815.yaml \
  --stage npu-smoke 2>&1 | tee "${SANDBOX}/logs/npu_smoke.log"
```

Do not use `exec` before `tee`, because the pipeline exit status is guarded by `set -o pipefail`.

- [ ] **Step 4: Add the opt-in NPU integration test**

Skip unless `RUN_XUANNV_NPU_SMOKE=1`. With one visible logical device, load prepared data, run a small Full forward/backward on `npu:0`, synchronize and assert finite results.

```python
@pytest.mark.npu
def test_one_visible_npu_runs_full_branch_backward():
    if os.environ.get("RUN_XUANNV_NPU_SMOKE") != "1":
        pytest.skip("explicit NPU smoke opt-in required")
    assert torch.npu.device_count() == 1
    result = run_small_npu_full_branch(device="npu:0")
    assert result.embedding.isfinite().all().item()
    assert result.aef_grad_norm > 0
    assert result.highres_grad_norm > 0
```

- [ ] **Step 5: Run static and NPU preflight tests**

```bash
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_launcher.py -q
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export RUN_XUANNV_NPU_SMOKE=1
PYTHONPATH=$PWD python -m pytest tests/isolated_smoke/test_npu_smoke.py -q
```

Expected: static tests PASS and NPU preflight PASS on one logical NPU.

- [ ] **Step 6: Run the full smoke in the foreground**

First confirm `fuser /dev/davinci2` prints nothing, then run:

```bash
bash scripts/smoke/run_china_v1_isolated_fusion_smoke.sh
```

Expected:

- four groups have `[4,8,64,128,128]`;
- no NaN/Inf;
- zero-gate Full/Base identity passes;
- gate override 0.1 gives finite nonzero AEF/highres gradients;
- checkpoint reload and deterministic rerun pass;
- source snapshot remains unchanged;
- `SUCCESS` is written last.

- [ ] **Step 7: Run post-smoke isolation checks**

```bash
test -f /data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815/SUCCESS
du -sh /data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
git -C /root/workspace/xuannv status --short
git status --short
```

Expected: sandbox below 5 GB; main worktree status matches its pre-smoke snapshot.

- [ ] **Step 8: Commit and push**

```bash
git add scripts/smoke/run_china_v1_isolated_fusion_smoke.sh \
  tests/isolated_smoke/test_launcher.py tests/isolated_smoke/test_npu_smoke.py
git commit -m "test: run isolated three-branch fusion smoke on NPU 2"
git push
```

---

### Task 8: Evidence Report, Regression, and Handoff

**Files:**

- Create: `docs/reports/china_v1_isolated_fusion_smoke_20260815.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: actual `run_manifest.json`, `metrics.json`, `path_audit.json`, `reproducibility.json`, disk usage and Git/NPU provenance.
- Produces: a human-readable `PASS`、`PARTIAL` or `FAIL` verdict.

- [ ] **Step 1: Build the report from actual evidence**

Record the selected patch IDs; source archive count and unchanged check; Python/torch/torch_npu/CANN/driver/Git; physical/logical device mapping; shapes and bytes; per-group latency and peak HBM; gradient norms; zero-gate maximum error; vMF norm min/median/max; checkpoint/reproducibility; sandbox size; created paths; and an explicit synthetic-data warning.

- [ ] **Step 2: Write the report and changelog entry**

Allow `PASS` only when `SUCCESS` and every required evidence file exist. Do not link `/data` paths as GitHub assets. State that no accuracy or 2 m quality conclusion is valid.

- [ ] **Step 3: Run the full regression**

```bash
export PYTHONPATH=$PWD/src:$PWD/downstreams:$PWD
python -m pytest tests/isolated_smoke tests/test_smoke.py tests/test_model.py -q
python -m ruff check experiments/china_v1_fusion_smoke tests/isolated_smoke
git diff --check
```

Expected: all tests PASS, Ruff clean, diff check clean.

- [ ] **Step 4: Verify repository scope**

```bash
git status --short
git diff --name-only 707071d..HEAD
git -C /root/workspace/xuannv status --short
```

Expected: only isolated-smoke files plus `CHANGELOG.md` changed on this branch; main worktree retains its original unrelated changes.

- [ ] **Step 5: Commit and push**

```bash
git add docs/reports/china_v1_isolated_fusion_smoke_20260815.md CHANGELOG.md
git commit -m "docs: report isolated China V1 fusion smoke evidence"
git push
```

- [ ] **Step 6: Run fresh completion verification**

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse @{u})"
test -f /data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815/SUCCESS
python -m pytest tests/isolated_smoke tests/test_smoke.py tests/test_model.py -q
```

Expected: clean worktree; local/remote commits equal; SUCCESS exists; all tests PASS.

---

## Execution Order and Stop Rules

Execute Tasks 1—8 strictly in order. Stop and report rather than guessing if:

- real archives differ from `pc-s2/{year}/{month}/pc-s2_{year}_{month}.zip` or the S1 equivalent;
- four complete patch IDs cannot be selected;
- any source archive size/mtime changes during prepare or smoke;
- NPU 2 is busy;
- the venv cannot import torch_npu after CANN setup;
- Zarr/Numcodecs are unavailable from inherited site-packages;
- sandbox exceeds 5 GB before final export;
- formal validator accepts synthetic data;
- zero-gate identity, finite-gradient, checkpoint or reproducibility checks fail.

No stop condition authorizes killing another process, changing NPU assignment, installing into the base environment, deleting existing data or falling back to a production Xuannv directory.
