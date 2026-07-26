# Pujiang Slides 03/05/08 Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Pujiang presentation pages 3, 5 and 8 with a plain-language training
explanation and a fair three-polygon PU+Query comparison across Xuannv, AEF and traditional
multi-source features.

**Architecture:** Extend the existing strict PU+Query evaluator so feature loading is separated
from the sparse-label algorithm. All feature sources share support polygons, split, standardization,
background mining, Query refinement and threshold selection. Page builders consume only the
generated JSON and figures, then the reviewed previews are merged into the ten-slide deck.

**Tech Stack:** Python 3.11, NumPy, PyTorch, rasterio, scikit-learn, matplotlib, Pillow,
python-pptx, pytest.

## Global Constraints

- Use exactly three support polygons per task for the presentation comparison.
- Tasks are building, road and water on Haidian fold 0.
- Never use validation or test labels to select support polygons, thresholds or examples.
- Xuannv is P10C epoch800 2026-04 64D; AEF is official 2025 annual 64D.
- Traditional input is the existing 2026-04 42-channel multi-source feature stack.
- Apply the same PU+Query algorithm and parameters to all three feature sources.
- Report F1, AUC and AP; do not pre-commit to a winning model.
- Keep page 4 unchanged.
- Commit and push after each independently reviewed deliverable.

---

### Task 1: Generalize the strict PU+Query evaluator

**Files:**
- Modify: `scripts/eval/run_pu_query_sparse_eval.py`
- Create: `tests/test_pu_query_sparse_comparison.py`

**Interfaces:**
- Consumes: existing `PolygonSupport`, `train_pu_query`, `score_pu_query` behavior.
- Produces: `FeatureSpec`, `load_feature(spec, patch_id)`, and comparison `results.json`.

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
assert module.feature_specs(args)["xuannv"].channels == 64
assert module.feature_specs(args)["aef"].month == "202512"
assert module.feature_specs(args)["traditional"].channels == 42
assert module.pick_supports(candidates, 3, seed) == module.pick_supports(candidates, 3, seed)
```

Add a synthetic comparison test proving all feature sources receive identical support IDs and
test IDs.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_pu_query_sparse_comparison.py -q
```

Expected: fail because `FeatureSpec` and multi-feature comparison do not exist.

- [ ] **Step 3: Implement feature-source separation**

Add a frozen `FeatureSpec` containing `name`, `kind`, `root`, `month` and `channels`.
Support:

```text
xuannv      embedding_map.pt, 202604, 64 channels
aef         embedding_map.pt, 202512, 64 channels
traditional fixed_highres_feature_map, 202604, 42 channels
```

Reuse `fixed_highres_feature_map` and manifest loading from
`scripts/eval/run_strong_downstream_benchmark.py`; do not duplicate raster normalization rules.

- [ ] **Step 4: Write comparison output**

For each task, select three supports once, then loop over the three `FeatureSpec` objects. Save:

```json
{
  "protocol": {
    "polygon_count": 3,
    "fold": 0,
    "shared_supports": true,
    "test_patch_count": 64
  },
  "rows": [
    {
      "task": "building",
      "feature": "xuannv",
      "support_patch_ids": ["..."],
      "metrics": {"f1": 0.0, "auc": 0.0, "ap": 0.0}
    }
  ]
}
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_pu_query_sparse_comparison.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit and push**

```bash
git add scripts/eval/run_pu_query_sparse_eval.py tests/test_pu_query_sparse_comparison.py
git commit -m "feat: compare sparse PU query feature sources"
git push
```

### Task 2: Run the fair three-polygon experiment

**Files:**
- Create generated data under:
  `/data/xuannv_embedding/experiments/production/haidian_pu_query_3polygon_compare_20260726/`
- Create report assets under:
  `docs/production/assets/haidian_pu_query_3polygon_compare_20260726/`
- Create: `docs/production/haidian_pu_query_3polygon_comparison_20260726.md`

**Interfaces:**
- Consumes: comparison mode from Task 1.
- Produces: `results.json`, one comparison summary figure and three task-row figures.

- [ ] **Step 1: Run all three tasks**

Run the evaluator with `polygon_count=3`, `fold=0`, `seed=42`, all three features and all three
tasks. Record elapsed time and source paths.

- [ ] **Step 2: Validate result integrity**

Check every task has exactly three rows, all rows share support patch IDs, all have 64 test
patches, and every F1/AUC/AP is finite and in `[0, 1]`.

- [ ] **Step 3: Generate figures**

Create:

```text
building_3polygon_compare.png
road_3polygon_compare.png
water_3polygon_compare.png
pu_query_3polygon_metrics.png
```

Each task row must show the three support polygons, Xuannv prediction, AEF prediction,
traditional-feature prediction and GT. Mark the displayed test case as a representative example;
it must not affect aggregate metrics.

- [ ] **Step 4: Write the evidence report**

Document protocol, paths, F1/AUC/AP table, result interpretation and water-task limitations.

- [ ] **Step 5: Commit and push report assets**

```bash
git add docs/production/haidian_pu_query_3polygon_comparison_20260726.md \
  docs/production/assets/haidian_pu_query_3polygon_compare_20260726
git commit -m "docs: report three-polygon PU query comparison"
git push
```

### Task 3: Rebuild page 3

**Files:**
- Modify: `scripts/report/build_pujiang_slide03.py`
- Modify: `tests/test_pujiang_slide03.py`
- Regenerate: `docs/presentations/assets/pujiang_202607/slide03/`

**Interfaces:**
- Consumes: P10C configuration and masking implementation.
- Produces: reviewed page 3 preview and individual PPTX.

- [ ] **Step 1: Update tests for plain-language facts**

Assert the page states:

```text
65%的训练样本启动月份遮挡
随机遮住1至4个月
不是每个月遮挡65%
65%的训练样本启动空间遮挡
启动后约遮住32%的空间块
13类OSM粗粒度提示
0.14是辅助损失权重，不是准确率
训练完成后丢弃临时探针，只保留64维嵌入
```

- [ ] **Step 2: Rebuild the page**

Use three large explanatory cards: “遮住什么”“为什么能学会”“OSM怎样帮助语义”。
Remove incomparable loss-length bars.

- [ ] **Step 3: Run tests and independent review**

Run `tests/test_pujiang_slide03.py`, inspect the 1600×900 preview and request an independent
subagent review. Revise until PASS.

- [ ] **Step 4: Commit and push**

Commit the builder, test, metadata and preview immediately after review passes.

### Task 4: Rebuild page 5

**Files:**
- Modify: `scripts/report/build_pujiang_slides04_10.py`
- Modify: `tests/test_pujiang_slides04_10.py`
- Regenerate: `docs/presentations/assets/pujiang_202607/slide05/`

**Interfaces:**
- Consumes: three task-row figures from Task 2.
- Produces: reviewed page 5 preview and individual PPTX.

- [ ] **Step 1: Update tests**

Assert page 5 includes “3个目标区域”, “PU自动寻找可信背景”, “Query局部适配”,
“建筑物”, “道路”, “水体”, “玄女”, “AEF”, “传统多源特征” and “独立测试GT”.

- [ ] **Step 2: Rebuild the page**

Keep one short explanatory paragraph. Use three task rows and avoid a separate dense protocol
panel. Label water as candidate discovery if supported by metrics.

- [ ] **Step 3: Run tests and independent review**

Run focused tests, inspect preview and revise until an independent subagent returns PASS.

- [ ] **Step 4: Commit and push**

Commit page 5 immediately after review passes.

### Task 5: Rebuild page 8

**Files:**
- Modify: `scripts/report/build_pujiang_slides04_10.py`
- Modify: `tests/test_pujiang_slides04_10.py`
- Regenerate: `docs/presentations/assets/pujiang_202607/slide08/`

**Interfaces:**
- Consumes: aggregate comparison metrics from Task 2.
- Produces: reviewed page 8 preview and individual PPTX.

- [ ] **Step 1: Update tests**

Assert page 8 names the three feature sources, three polygons, shared PU+Query, F1, AUC, AP,
64 independent test patches and monthly-versus-annual time-product difference.

- [ ] **Step 2: Rebuild the page**

Use grouped F1 and AUC charts plus a compact AP table. The conclusion is generated from actual
metrics and cannot claim a win that the JSON does not support.

- [ ] **Step 3: Run tests and independent review**

Run focused tests, inspect preview and revise until an independent subagent returns PASS.

- [ ] **Step 4: Commit and push**

Commit page 8 immediately after review passes.

### Task 6: Merge and verify the ten-slide deck

**Files:**
- Regenerate:
  `docs/presentations/pujiang_202607/玄女月度地理嵌入_浦江交流_10页完整版_20260726.pptx`

**Interfaces:**
- Consumes: reviewed previews for pages 1 through 10.
- Produces: final ten-slide presentation.

- [ ] **Step 1: Merge reviewed previews**

Run:

```bash
python scripts/report/build_pujiang_slides04_10.py --merge
```

- [ ] **Step 2: Run all presentation tests**

Run:

```bash
python -m pytest tests/test_pujiang_slide01_api_assets.py \
  tests/test_pujiang_slide02.py tests/test_pujiang_slide03.py \
  tests/test_pujiang_slides04_10.py -q
```

- [ ] **Step 3: Request independent whole-deck review**

Check narrative order, model and time-product labels, sparse-label protocol, result boundaries,
visual consistency and final closure. Revise until PASS.

- [ ] **Step 4: Verify deliverable**

Confirm the complete PPTX contains exactly ten slides and report its absolute path and size.
