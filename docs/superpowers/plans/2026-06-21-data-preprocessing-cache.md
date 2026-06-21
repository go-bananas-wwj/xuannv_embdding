# Data Preprocessing Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent per-patch disk cache for `MonthlyEmbeddingDataset` so training epochs after the first skip TIFF decoding/normalization/binning.

**Architecture:** Wrap `__getitem__` with a lazy `.pt` cache keyed by `(region, patch_id)`; store cache metadata to invalidate when stats/sources change; add a standalone pre-generation script.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, YAML config, pytest.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/xuannv_embedding/config.py` | Add `cache_dir` to `DataConfig` and load it from YAML |
| `src/xuannv_embedding/data/dataset.py` | Cache read/write logic, cache metadata, atomic writes |
| `src/xuannv_embedding/data/builder.py` | Pass `cache_dir` from config into dataset |
| `scripts/train/train.py` | Pass `cache_dir` from config into dataset |
| `scripts/data/preprocess_cache.py` | Standalone script to pre-warm cache |
| `configs/v1.1_distill_long_stable_50ep.yaml` | Enable `cache_dir` |
| `tests/test_dataset.py` | Unit test for lazy cache creation and invalidation |

---

### Task 1: Add `cache_dir` to `DataConfig`

**Files:**
- Modify: `src/xuannv_embedding/config.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Add field to `DataConfig`**

  In `src/xuannv_embedding/config.py`, change the `DataConfig` dataclass:

  ```python
  @dataclass
  class DataConfig:
      """数据相关配置。"""

      root: Path
      region: str
      manifest_path: Path
      num_samples: int | None = None
      statistics_dir: Path | None = None
      max_patches: int | None = None
      batch_size: int = 4
      num_workers: int = 8
      patch_size: int = 256
      num_months: int = 17
      months: list[str] = field(default_factory=lambda: [])
      sources: list[str] = field(default_factory=lambda: ["s2", "s1", "landsat"])
      teacher_embedding_root: Path | None = None
      cache_dir: Path | None = None  # NEW
  ```

- [ ] **Step 2: Load `cache_dir` from YAML in `Config.from_yaml`**

  In the `DataConfig(...)` constructor call inside `from_yaml`, add:

  ```python
  cache_dir=(
      Path(data_cfg["cache_dir"])
      if data_cfg.get("cache_dir")
      else None
  ),
  ```

- [ ] **Step 3: Write a small config-load test**

  Append to `tests/test_dataset.py`:

  ```python
  def test_config_loads_cache_dir(tmp_path: Path) -> None:
      from xuannv_embedding.config import Config

      yaml_path = tmp_path / "cfg.yaml"
      yaml_path.write_text(
          """
experiment:
  name: cache_test
  use_wandb: false
training:
  epochs: 1
  lr: 1e-4
  weight_decay: 0.01
  warmup_epochs: 0
  gradient_accumulation_steps: 1
  save_every: 1
  eval_every: 1
model:
  embed_dim: 8
  sensor_channels:
    s2: 12
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: 12
      weight: 1.0
data:
  root: /tmp
  region: test
  manifest_path: /tmp/manifest.json
  cache_dir: /tmp/cache
""",
          encoding="utf-8",
      )
      cfg = Config.from_yaml(yaml_path)
      assert cfg.data.cache_dir == Path("/tmp/cache")
  ```

- [ ] **Step 4: Run the new test**

  ```bash
  cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
  pytest tests/test_dataset.py::test_config_loads_cache_dir -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/xuannv_embedding/config.py tests/test_dataset.py
  git commit -m "feat(config): add cache_dir to DataConfig"
  ```

---

### Task 2: Implement cache read/write in `MonthlyEmbeddingDataset`

**Files:**
- Modify: `src/xuannv_embedding/data/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Add `cache_dir` parameter and validation**

  Update the constructor signature and body:

  ```python
  def __init__(
      self,
      manifest_path: Path,
      statistics_dir: Path,
      sources: list[str],
      patch_size: int = 256,
      max_patches: int | None = None,
      num_months: int = 17,
      ref_year: int = 2025,
      ref_month: int = 1,
      teacher_embedding_root: Path | None = None,
      region: str | None = None,
      cache_dir: Path | None = None,  # NEW
  ) -> None:
      ...
      self.cache_dir = Path(cache_dir) if cache_dir else None
      ...
      self._load_statistics()
      self._ensure_cache_valid()
      ...
  ```

- [ ] **Step 2: Add cache helper methods**

  Insert these private methods into `MonthlyEmbeddingDataset`:

  ```python
  def _cache_dir_for_patch(self, entry: dict[str, Any]) -> Path:
      assert self.cache_dir is not None
      region = entry.get("region", self.region or "")
      return self.cache_dir / f"preprocessed_{self.patch_size}" / str(region)

  def _cache_file_for(self, entry: dict[str, Any]) -> Path:
      return self._cache_dir_for_patch(entry) / f"{entry['patch_id']}.pt"

  def _compute_stats_hash(self) -> str:
      import hashlib

      h = hashlib.sha256()
      for key in sorted(self.statistics.keys()):
          stats = self.statistics[key]
          h.update(json.dumps(stats, sort_keys=True).encode("utf-8"))
      return h.hexdigest()[:16]

  def _cache_meta(self) -> dict[str, Any]:
      return {
          "patch_size": self.patch_size,
          "num_months": self.num_months,
          "sources": sorted(self.sources),
          "statistics_hash": self._compute_stats_hash(),
          "teacher_embedding_root": (
              str(self.teacher_embedding_root) if self.teacher_embedding_root else None
          ),
          "version": "1.0",
      }

  def _ensure_cache_valid(self) -> None:
      if self.cache_dir is None:
          return

      preproc_dir = self.cache_dir / f"preprocessed_{self.patch_size}"
      meta_path = preproc_dir / "cache_meta.json"
      current_meta = self._cache_meta()

      if meta_path.exists():
          try:
              old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
              if old_meta == current_meta:
                  return
              logger.info("Cache meta changed, clearing %s", preproc_dir)
          except Exception as exc:
              logger.warning("Failed to read cache meta %s: %s", meta_path, exc)

      if preproc_dir.exists():
          import shutil

          shutil.rmtree(preproc_dir)
      preproc_dir.mkdir(parents=True, exist_ok=True)
      meta_path.write_text(
          json.dumps(current_meta, indent=2, sort_keys=True),
          encoding="utf-8",
      )
  ```

- [ ] **Step 3: Rename existing `__getitem__` body to `_load_sample`**

  Copy the entire current `__getitem__(self, idx)` implementation into a new method:

  ```python
  def _load_sample(self, idx: int) -> dict[str, Any]:
      """从磁盘加载并预处理一个样本（不含缓存）。"""
      # existing __getitem__ body here
  ```

- [ ] **Step 4: Make `__getitem__` cache-aware**

  Replace the original `__getitem__` with:

  ```python
  def __getitem__(self, idx: int) -> dict[str, Any]:
      entry = self.manifest[idx]

      if self.cache_dir is not None:
          cache_file = self._cache_file_for(entry)
          if cache_file.exists():
              try:
                  return torch.load(cache_file, map_location="cpu", weights_only=False)
              except Exception as exc:
                  logger.warning("Failed to load cache %s: %s", cache_file, exc)
                  cache_file.unlink(missing_ok=True)

      sample = self._load_sample(idx)

      if self.cache_dir is not None:
          cache_file = self._cache_file_for(entry)
          cache_file.parent.mkdir(parents=True, exist_ok=True)
          tmp = cache_file.with_suffix(".tmp")
          try:
              torch.save(sample, tmp)
              tmp.replace(cache_file)
          except Exception as exc:
              logger.warning("Failed to save cache %s: %s", cache_file, exc)
              tmp.unlink(missing_ok=True)

      return sample
  ```

- [ ] **Step 5: Add cache unit test**

  Append to `tests/test_dataset.py`:

  ```python
  def test_dataset_caches_samples(tmp_path: Path) -> None:
      root = tmp_path / "processed" / "test"
      root.mkdir(parents=True)
      stats_dir = tmp_path / "statistics" / "test"
      stats_dir.mkdir(parents=True)

      s2_dir = root / "patches" / "s2"
      _write_tiff(
          s2_dir / "s2_20250102_patch_000000.tif",
          np.full((1, 4, 4), 3.0, dtype=np.float32),
      )

      manifest = [
          {
              "patch_id": "patch_000000",
              "s2": ["patches/s2/s2_20250102_patch_000000.tif"],
          }
      ]
      manifest_path = root / "manifest.json"
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

      cache_dir = tmp_path / "cache"
      dataset = MonthlyEmbeddingDataset(
          manifest_path=manifest_path,
          statistics_dir=stats_dir,
          sources=["s2"],
          num_months=1,
          patch_size=4,
          cache_dir=cache_dir,
      )

      sample1 = dataset[0]
      cache_file = cache_dir / "preprocessed_4" / "test" / "patch_000000.pt"
      assert cache_file.exists()

      sample2 = dataset[0]
      assert sample1["patch_id"] == sample2["patch_id"]
      assert torch.equal(sample1["source_frames"]["s2"], sample2["source_frames"]["s2"])
  ```

- [ ] **Step 6: Run dataset tests**

  ```bash
  pytest tests/test_dataset.py -v
  ```
  Expected: all PASS

- [ ] **Step 7: Commit**

  ```bash
  git add src/xuannv_embedding/data/dataset.py tests/test_dataset.py
  git commit -m "feat(dataset): add persistent per-patch preprocessing cache"
  ```

---

### Task 3: Wire `cache_dir` through data builder and training entry

**Files:**
- Modify: `src/xuannv_embedding/data/builder.py`
- Modify: `scripts/train/train.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Pass `cache_dir` in `build_dataloader`**

  In `src/xuannv_embedding/data/builder.py`, add `cache_dir=cfg.data.cache_dir` to the `MonthlyEmbeddingDataset(...)` call.

- [ ] **Step 2: Pass `cache_dir` in training script `_build_loader`**

  In `scripts/train/train.py`, add `cache_dir=cfg.data.cache_dir` to both `MonthlyEmbeddingDataset(...)` calls inside `_build_loader`.

- [ ] **Step 3: Test builder passes cache_dir**

  Append to `tests/test_dataset.py`:

  ```python
  def test_build_dataloader_with_cache_dir(tmp_path: Path) -> None:
      from xuannv_embedding.config import DataConfig

      cfg = DataConfig(
          root=MANIFEST_PATH.parent,
          region="harbin",
          manifest_path=MANIFEST_PATH,
          statistics_dir=STATISTICS_DIR,
          max_patches=1,
          batch_size=1,
          num_workers=0,
          patch_size=128,
          sources=["s2"],
          cache_dir=tmp_path / "cache",
      )
      loader = build_dataloader(cfg, split="train")
      batch = next(iter(loader))
      assert batch["source_frames"]["s2"].shape == (1, 17, 12, 128, 128)
      assert (tmp_path / "cache" / "preprocessed_128" / "harbin").exists()
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/test_dataset.py::test_build_dataloader_with_cache_dir -v
  ```
  Expected: PASS

- [ ] **Step 5: Commit**

  ```bash
  git add src/xuannv_embedding/data/builder.py scripts/train/train.py tests/test_dataset.py
  git commit -m "chore: wire cache_dir through builder and train entry"
  ```

---

### Task 4: Add pre-generation script

**Files:**
- Create: `scripts/data/preprocess_cache.py`
- Test: run it manually

- [ ] **Step 1: Create the script**

  ```python
  #!/usr/bin/env python3
  from __future__ import annotations

  import argparse
  import logging
  from pathlib import Path

  from tqdm import tqdm

  from xuannv_embedding.config import Config
  from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset

  logger = logging.getLogger(__name__)


  def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(description="预生成训练数据缓存")
      parser.add_argument("--config", type=str, required=True, help="YAML 配置文件")
      parser.add_argument(
          "--max-patches",
          type=int,
          default=None,
          help="最多预处理多少个 patch（用于测试）",
      )
      return parser.parse_args()


  def main() -> None:
      args = parse_args()
      cfg = Config.from_yaml(args.config)
      if not cfg.data.cache_dir:
          raise ValueError("配置中未设置 data.cache_dir")

      dataset = MonthlyEmbeddingDataset(
          manifest_path=cfg.data.manifest_path,
          statistics_dir=cfg.data.statistics_dir,
          sources=cfg.data.sources,
          patch_size=cfg.data.patch_size,
          max_patches=args.max_patches,
          num_months=cfg.model.num_months,
          teacher_embedding_root=cfg.data.teacher_embedding_root,
          region=cfg.data.region,
          cache_dir=cfg.data.cache_dir,
      )

      logger.info("预生成 %d 个 patch 的缓存到 %s", len(dataset), cfg.data.cache_dir)
      for i in tqdm(range(len(dataset))):
          _ = dataset[i]
      logger.info("完成")


  if __name__ == "__main__":
      logging.basicConfig(
          level=logging.INFO,
          format="%(asctime)s %(levelname)s %(message)s",
      )
      main()
  ```

- [ ] **Step 2: Make it executable**

  ```bash
  chmod +x scripts/data/preprocess_cache.py
  ```

- [ ] **Step 3: Smoke-test with a small subset**

  ```bash
  python scripts/data/preprocess_cache.py \
      --config configs/v1.1_distill_long_stable_50ep.yaml \
      --max-patches 5
  ```

  Expected: creates files under `/data/xuannv_embedding/cache/preprocessed_128/{harbin,haidian}/...`

- [ ] **Step 4: Commit**

  ```bash
  git add scripts/data/preprocess_cache.py
  git commit -m "feat(data): add preprocessing cache pre-generation script"
  ```

---

### Task 5: Enable cache in training config

**Files:**
- Modify: `configs/v1.1_distill_long_stable_50ep.yaml`

- [ ] **Step 1: Add `cache_dir` to config**

  ```yaml
  data:
    region: combined
    root: /data/xuannv_embedding/processed
    manifest_path: /data/xuannv_embedding/processed/manifest_all_744.json
    statistics_dir: /data/xuannv_embedding/statistics
    patch_size: 128
    num_workers: 0
    batch_size: 4
    cache_dir: /data/xuannv_embedding/cache/preprocessed_128  # NEW
    sources:
      - s2
      - s1
      - landsat
      - worldcover
      - highres_optical_harbin
      - highres_optical
      - highres_sar
    teacher_embedding_root: /data/xuannv_embedding/embeddings/aef_teacher_2025_annual
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add configs/v1.1_distill_long_stable_50ep.yaml
  git commit -m "config(v1.1): enable preprocessing cache"
  ```

---

### Task 6: Integration smoke test

**Files:**
- None (manual verification)

- [ ] **Step 1: Pre-generate a few cache files and run 2 epochs**

  ```bash
  python scripts/data/preprocess_cache.py \
      --config configs/v1.1_distill_long_stable_50ep.yaml \
      --max-patches 20
  ```

- [ ] **Step 2: Run a short single-card training smoke with caching**

  ```bash
  python scripts/train/train.py --config configs/v1.1_distill_long_stable_50ep.yaml --device npu:0
  ```

  Let it run for 2 epochs. Verify:
  - First epoch logs show cache being created (if not pre-generated) or read
  - Second epoch wall time is clearly lower than the first

- [ ] **Step 3: Verify DDP 6-card starts without crash**

  ```bash
  export WANDB_API_KEY=...
  bash scripts/train/launch_6card.sh configs/v1.1_distill_long_stable_50ep.yaml
  ```

  Let it run for 2-3 epochs. Expected: no DataLoader worker / shared memory crash.

- [ ] **Step 4: Commit any small fixes from smoke test**

---

## Self-Review Checklist

- [ ] **Spec coverage:**
  - Lazy cache in `__getitem__` → Task 2
  - Persistent disk storage → Task 2
  - Cache invalidation via metadata → Task 2
  - DDP-safe atomic writes → Task 2
  - Config field → Task 1
  - Pre-generation script → Task 4
  - Training config enabled → Task 5
  - Tests → Tasks 1-3

- [ ] **Placeholder scan:** No TBD/TODO; every step has code or command.
- [ ] **Type consistency:** `cache_dir` is `Path | None` everywhere; helper names match across tasks.
