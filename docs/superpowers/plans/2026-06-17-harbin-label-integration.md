# Harbin Label Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the newly downloaded Harbin quadrant labels, integrate the `01/02/03/04` quadrant annotations into 128×128 construction masks, generate Harbin embeddings, retrain the construction-site downstream head, and produce two-row patch visualizations.

**Architecture:** Labels are delivered as per-patch LabelMe JSONs split into four quadrants and two dates. We copy them to `Data-raw`, rasterize each quadrant into its correct sub-region of a per-date 128×128 mask, generate stage-2 embeddings for the Harbin region, then feed masks + embeddings into the existing 5-fold downstream segmentation pipeline.

**Tech Stack:** Python, rasterio, shapely, PyTorch, ModelScope cache.

---

### Task 1: Persist Harbin Labels to `Data-raw`

**Files:**
- Create: `/data/xuannv_embedding/Data-raw/haerbin_label_2026/`

- [ ] **Step 1: Copy labels from ModelScope cache**

Run:

```bash
CACHE=/tmp/modelscope_cache/WeijieWu/xuannv_train_data/haerbin_label/2026
DEST=/data/xuannv_embedding/Data-raw/haerbin_label_2026
mkdir -p "$DEST"
cp -r "$CACHE"/patch_* "$DEST/"
find "$DEST" -name '*.json' | wc -l
```

Expected: 284 JSON files copied.

- [ ] **Step 2: Commit / record provenance**

Write a small provenance file (no git commit unless asked):

```bash
cat > /data/xuannv_embedding/Data-raw/haerbin_label_2026/README.txt << 'EOF'
Source: https://modelscope.cn/datasets/WeijieWu/xuannv_train_data/tree/master/haerbin_label
Downloaded via ModelScope CLI to /tmp/modelscope_cache/.../haerbin_label/2026
Copied to Data-raw on $(date -Iseconds).
Structure: patch_000xxx/{01,02,03,04}_{20251201,20260501}.json
EOF
```

---

### Task 2: Update `scripts/process_harbin_patches.py` for Quadrant Labels

**Files:**
- Modify: `/root/workspace/xuannv/scripts/process_harbin_patches.py`

- [ ] **Step 1: Add quadrant/date parsing helpers**

Insert after `_group_by_patch_id`:

```python
_QUADRANT_ORDER = {
    "01": (0, 0),
    "02": (0, 1),
    "03": (1, 0),
    "04": (1, 1),
}


def _parse_label_stem(stem: str) -> tuple[str, str, str] | None:
    """Parse '01_20260501' -> ('01', '20260501', '202605')."""
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    quadrant, date = parts[0], parts[1]
    return quadrant, date, date[:6]


def _group_by_patch_and_date(
    paths: list[Path], root: Path
) -> dict[str, dict[str, list[Path]]]:
    """{patch_id: {date: [label_paths]}}."""
    groups: dict[str, dict[str, list[Path]]] = {}
    for p in paths:
        rel = p.relative_to(root)
        m = re.match(r"(patch_\d+)", rel.parts[0])
        if not m:
            continue
        pid = m.group(1)
        parsed = _parse_label_stem(p.stem)
        if parsed is None:
            continue
        quadrant, date, month = parsed
        if quadrant not in _QUADRANT_ORDER:
            continue
        groups.setdefault(pid, {}).setdefault(month, []).append(p)
    return groups
```

- [ ] **Step 2: Add full-patch size inference and per-quadrant rasterization**

Replace `_rasterize_labelme` with the following two functions:

```python
def _infer_full_size(quadrant_dims: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Infer full image (W, H) from present quadrant dimensions.

    Assumes 2x2 layout: 01 top-left, 02 top-right, 03 bottom-left, 04 bottom-right.
    """
    left_w = max(
        [quadrant_dims[q][0] for q in ("01", "03") if q in quadrant_dims] or [0]
    )
    right_w = max(
        [quadrant_dims[q][0] for q in ("02", "04") if q in quadrant_dims] or [0]
    )
    top_h = max(
        [quadrant_dims[q][1] for q in ("01", "02") if q in quadrant_dims] or [0]
    )
    bottom_h = max(
        [quadrant_dims[q][1] for q in ("03", "04") if q in quadrant_dims] or [0]
    )

    if left_w == 0 and right_w == 0:
        raise ValueError("no quadrant widths available")
    if top_h == 0 and bottom_h == 0:
        raise ValueError("no quadrant heights available")

    # If one side is missing, mirror the known side.
    full_w = (left_w if left_w else right_w) + (right_w if right_w else left_w)
    full_h = (top_h if top_h else bottom_h) + (bottom_h if bottom_h else top_h)
    return full_w, full_h


def _rasterize_quadrant_labelme(
    label_path: Path,
    quadrant: str,
    full_size: tuple[int, int],
    out_shape: tuple[int, int],
    class_map: dict[str, int] | None = None,
) -> np.ndarray:
    if class_map is None:
        class_map = {
            "jiazhudongdi": 1,
            "gongdi": 1,
            "construction site": 1,
        }
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_h = data.get("imageHeight", 1)
    img_w = data.get("imageWidth", 1)

    q_row, q_col = _QUADRANT_ORDER[quadrant]
    full_w, full_h = full_size

    # Determine this quadrant's bounding box in full-pixel coordinates.
    if q_col == 0:
        x0 = 0
        x1 = max(img_w, full_w // 2)
    else:
        x1 = full_w
        x0 = full_w - max(img_w, full_w - (full_w // 2))
    if q_row == 0:
        y0 = 0
        y1 = max(img_h, full_h // 2)
    else:
        y1 = full_h
        y0 = full_h - max(img_h, full_h - (full_h // 2))

    q_w = x1 - x0
    q_h = y1 - y0

    # Map quadrant-local pixels -> output-mask sub-region.
    scale_x = out_shape[1] * (q_w / full_w) / max(img_w, 1)
    scale_y = out_shape[0] * (q_h / full_h) / max(img_h, 1)
    offset_x = out_shape[1] * (x0 / full_w)
    offset_y = out_shape[0] * (y0 / full_h)

    geometries = []
    for s in data.get("shapes", []):
        label = s.get("label", "").strip().lower()
        if label not in class_map:
            continue
        pts = [(x * scale_x + offset_x, y * scale_y + offset_y) for x, y in s["points"]]
        geom = Polygon(pts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid and not geom.is_empty:
            geometries.append((geom, class_map[label]))

    mask = np.zeros(out_shape, dtype=np.uint8)
    if geometries:
        mask = rasterize(
            geometries,
            out_shape=out_shape,
            fill=0,
            default_value=0,
            dtype=np.uint8,
            all_touched=True,
        )
    return mask
```

- [ ] **Step 3: Update `process_harbin` to iterate by patch/date and skip image copy when images are absent**

Replace the body of `process_harbin` with:

```python
def process_harbin(
    raw_root: Path,
    output_root: Path,
    target_mask_size: tuple[int, int] = (128, 128),
    class_map: dict[str, int] | None = None,
) -> None:
    labelme_dir = output_root / "labels" / "construction" / "labelme_raw"
    mask_dir = output_root / "labels" / "construction" / "masks"
    img_out_dir = output_root / "patches" / "highres_optical"

    label_paths = _find_files(raw_root, "*.json")
    image_paths = (
        _find_files(raw_root, "*.tif")
        + _find_files(raw_root, "*.png")
        + _find_files(raw_root, "*.jpg")
    )
    label_groups = _group_by_patch_and_date(label_paths, raw_root)
    image_groups = _group_by_patch_id(image_paths, raw_root)

    stats = []
    for pid in sorted(label_groups):
        month_groups = label_groups[pid]
        images = image_groups.get(pid, [])
        ref_img = _reference_image(images)

        for month in sorted(month_groups):
            q_labels = month_groups[month]
            patch_label_dir = labelme_dir / pid / month
            patch_label_dir.mkdir(parents=True, exist_ok=True)
            for lp in q_labels:
                shutil.copy2(lp, patch_label_dir / lp.name)

            # Gather dimensions per quadrant.
            quadrant_dims: dict[str, tuple[int, int]] = {}
            for lp in q_labels:
                parsed = _parse_label_stem(lp.stem)
                if parsed is None:
                    continue
                q, _, _ = parsed
                with open(lp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                quadrant_dims[q] = (data.get("imageWidth", 1), data.get("imageHeight", 1))

            full_size = _infer_full_size(quadrant_dims)
            mask = np.zeros(target_mask_size, dtype=np.uint8)
            for lp in q_labels:
                parsed = _parse_label_stem(lp.stem)
                if parsed is None:
                    continue
                q, _, _ = parsed
                m = _rasterize_quadrant_labelme(
                    lp, q, full_size, target_mask_size, class_map
                )
                mask = np.maximum(mask, m)

            mask_path = mask_dir / f"{pid}_{month}.tif"
            _save_mask(mask, mask_path)

            # Copy reference image only if present; downstream pipeline already
            # has processed highres optical patches, so this is optional.
            if ref_img is not None:
                patch_img_dir = img_out_dir / pid
                patch_img_dir.mkdir(parents=True, exist_ok=True)
                dst_img = patch_img_dir / ref_img.name
                if not dst_img.exists():
                    shutil.copy2(ref_img, dst_img)

            ratio = float((mask > 0).sum() / mask.size)
            stats.append({
                "patch_id": pid,
                "month": month,
                "label_count": len(q_labels),
                "image": str(dst_img.relative_to(output_root)) if ref_img else None,
                "mask": str(mask_path.relative_to(output_root)),
                "positive_ratio": ratio,
            })
            logger.info("%s %s 处理完成: 正样本比例 %.3f", pid, month, ratio)

    summary_path = output_root / "harbin_patches_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("共处理 %d 个 patch/date，汇总保存至 %s", len(stats), summary_path)
```

- [ ] **Step 4: Run unit tests / smoke test**

Run existing tests to ensure no regression:

```bash
cd /root/workspace/xuannv
pytest downstreams/tests/test_label_loaders.py -v
```

Then run the updated script on a small subset to verify it produces masks:

```bash
python scripts/process_harbin_patches.py \
  --raw-root /data/xuannv_embedding/Data-raw/haerbin_label_2026 \
  --output-root /data/xuannv_embedding/processed/harbin
```

Expected: `processed/harbin/labels/construction/masks/` contains `{patch_id}_202605.tif` and `{patch_id}_202512.tif` files; summary reports ~191 patch/month entries.

---

### Task 3: Make `EmbeddingDataset` Load Month-Specific Masks

**Files:**
- Modify: `/root/workspace/xuannv/downstreams/downstreams/data/embedding_dataset.py`

- [ ] **Step 1: Update mask path resolution**

Change the mask loading in `__getitem__` from:

```python
mask_path = self.mask_dir / f"{patch_id}.tif"
```

to:

```python
mask_path = self.mask_dir / f"{patch_id}.tif"
month_mask_path = self.mask_dir / f"{patch_id}_{self.month}.tif"
if month_mask_path.exists():
    mask_path = month_mask_path
```

- [ ] **Step 2: Run dataset tests**

```bash
pytest downstreams/tests/test_embedding_dataset.py -v
```

---

### Task 4: Generate Harbin Embeddings

**Files:**
- Use: `/root/workspace/xuannv/downstreams/scripts/precompute_embeddings.py`
- Use checkpoint: `/data/xuannv_embedding/outputs/harbin_128_stage2_v1_20260620_1628/best.pt`
- Use config: `/root/workspace/xuannv/configs/harbin_128_stage2.yaml`

- [ ] **Step 1: Run precompute_embeddings.py in the background**

```bash
cd /root/workspace/xuannv
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1_20260620_1628/best.pt \
  --regions harbin \
  --output-root /data/xuannv_embedding/embeddings
```

This may take 30–90 minutes. Run it as a background task and continue once complete.

- [ ] **Step 2: Verify output months**

After it finishes, list one patch directory:

```bash
ls /data/xuannv_embedding/embeddings/*harbin*/harbin/patch_000000/
```

Expected: `{YYYYMM}_embedding_map.pt` files. Note the month strings available for 202512 and 202605.

---

### Task 5: Build 5-Fold Splits for Harbin Construction Labels

**Files:**
- Use: `/data/xuannv_embedding/processed/harbin/labels/construction/split_5fold.json` (auto-created by `train_task.py`)

- [ ] **Step 1: Trigger split creation manually so it can be inspected before training**

```bash
cd /root/workspace/xuannv
python - << 'PY'
from pathlib import Path
from downstreams.data.split import create_stratified_folds
mask_dir = Path("/data/xuannv_embedding/processed/harbin/labels/construction/masks")
split = create_stratified_folds(mask_dir, n_folds=5, val_ratio=0.2, seed=42)
out = mask_dir.parent / "split_5fold.json"
import json
with open(out, "w", encoding="utf-8") as f:
    json.dump(split, f, ensure_ascii=False, indent=2)
print("wrote", out, "folds", len(split["folds"]))
PY
```

- [ ] **Step 2: Inspect positive ratios**

```bash
python - << 'PY'
import json
split = json.load(open("/data/xuannv_embedding/processed/harbin/labels/construction/split_5fold.json"))
for fold in split["folds"]:
    print(f"fold {fold['fold']}: train={len(fold['train'])} val={len(fold['val'])} test={len(fold['test'])}")
PY
```

---

### Task 6: Train Construction Downstream Head on Harbin

**Files:**
- Create: `/root/workspace/xuannv/downstreams/configs/construction_segmentation_harbin.yaml`
- Use: `/root/workspace/xuannv/downstreams/scripts/train_task.py`

- [ ] **Step 1: Create Harbin-specific config**

```yaml
_base_: construction_segmentation.yaml

experiment:
  name: construction_site_harbin_v1

training:
  month: 202605
  pos_weight: 50.0
  pos_prior: 0.02
  lr: 3.0e-5
  early_stop_metric: f1_best
```

(Adjust `month` to a month that exists in the generated embeddings.)

- [ ] **Step 2: Run a single-fold smoke test**

```bash
cd /root/workspace/xuannv
python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_segmentation_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/<RUN_DIR> \
  --label-root /data/xuannv_embedding/processed/harbin/labels/construction \
  --output-root /data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0 \
  --fold 0
```

Use the actual `<RUN_DIR>` produced by Task 4.

- [ ] **Step 3: Run full 5-fold if smoke passes**

Remove `--fold 0` and rerun to produce all five folds and `summary_5fold.json`.

- [ ] **Step 4: Run fractional-label experiments (optional)**

For `fraction` in `0.1 0.25 0.5`:

```bash
python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_segmentation_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/<RUN_DIR> \
  --label-root /data/xuannv_embedding/processed/harbin/labels/construction \
  --output-root /data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac<F> \
  --fraction <F>
```

---

### Task 7: Evaluate and Summarize

**Files:**
- Read: `/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/summary_5fold.json`

- [ ] **Step 1: Aggregate metrics**

```bash
python - << 'PY'
import json, glob, numpy as np
for exp in sorted(glob.glob("/data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_*")):
    summary = Path(exp) / "summary_5fold.json"
    if not summary.exists():
        continue
    data = json.load(open(summary))
    print(exp, data.get("mean", {}))
PY
```

- [ ] **Step 2: Compare with Haidian baseline**

Copy the mean metrics into the existing comparison report or create `/data/xuannv_embedding/outputs/downstream/harbin_construction_v1_summary.md`.

---

### Task 8: Produce Two-Row Patch Visualizations

**Files:**
- Use: `/root/workspace/xuannv/downstreams/scripts/visualize_patches.py`

- [ ] **Step 1: Generate visualizations for all labeled Harbin patches**

```bash
cd /root/workspace/xuannv
python downstreams/scripts/visualize_patches.py \
  --emb-root /data/xuannv_embedding/embeddings/<RUN_DIR>/harbin \
  --data-root /data/xuannv_embedding/processed/harbin \
  --mask-dir /data/xuannv_embedding/processed/harbin/labels/construction/masks \
  --pred-dir /data/xuannv_embedding/outputs/downstream/stage2_harbin_construction_v1_frac1.0/fold_0/predictions \
  --out-dir /data/xuannv_embedding/outputs/downstream/visualizations/harbin_stage2_v1 \
  --all-labeled \
  --month 202605
```

- [ ] **Step 2: Verify outputs**

```bash
ls /data/xuannv_embedding/outputs/downstream/visualizations/harbin_stage2_v1 | head -n 20
```

Expected: PNG files for each patch that has a mask and prediction.

---

## Self-Review

- **Spec coverage:** Copy labels (Task 1), integrate quadrants (Task 2), month-specific masks (Task 3), embeddings (Task 4), splits (Task 5), train/eval (Task 6-7), visualize (Task 8) are all covered.
- **Placeholder scan:** No `TBD`/`TODO` in code steps; month placeholder `<RUN_DIR>` is intentionally filled after Task 4 completes.
- **Type consistency:** `_group_by_patch_and_date` returns same key types as the original `_group_by_patch_id`; mask path template matches the new dataset loader logic.
