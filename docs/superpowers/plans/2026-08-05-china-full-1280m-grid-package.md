# China Full 1,280 m Grid Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the canonical nationwide 1,280 m parent grid, split it into sampled and unsampled patch sets, and deliver all three layers with documentation and reproducible audits in one archive.

**Architecture:** Enumerate exact patch cells from the frozen 12.8 km macrocell inventory one UTM zone at a time. Stream each zone in bounded batches to partitioned GeoParquet and per-zone Shapefiles, route records to sampled or unsampled outputs using the existing 62,000-key registry, then build a manifest, membership audit, Chinese DOCX, previews, checksums, and ZIP.

**Tech Stack:** Python 3.11, GeoPandas 1.1, Shapely 2.1, PyProj 3.7, Pyogrio 0.12, PyArrow 24, pytest, python-docx, Matplotlib.

## Global Constraints

- Patch side is exactly 1,280 m on a `(0 m, 0 m)`-anchored integer grid.
- UTM owner zones are WGS84 / UTM north EPSG:32643 through EPSG:32653.
- A parent patch is included when its center is covered by the frozen ADM0 boundary and its longitude belongs to that half-open UTM owner zone.
- WGS84 delivery geometry is EPSG:4326 with X=longitude and Y=latitude.
- Existing sampled membership is keyed by `grid_epsg + grid_col + grid_row`; all 62,000 keys must match exactly once.
- `sampled ∪ unsampled = all`, `sampled ∩ unsampled = ∅`.
- GeoParquet is canonical; Shapefile fields are intentionally short and each component must remain below 2 GB.
- Generation is streaming by UTM zone and batch; do not hold the nationwide polygon atlas in memory.
- Output root is `/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805/`.
- Existing unrelated worktree changes must not be reverted or included in task commits.

---

### Task 1: Freeze the Grid Contract and Exact Enumerator

**Files:**
- Create: `configs/national/china_full_1280m_grid_v1_20260805.json`
- Create: `scripts/data/china_full_grid.py`
- Create: `tests/test_china_full_grid.py`

**Interfaces:**
- Consumes: frozen ADM0 GeoJSON and `national_macrocell_inventory.jsonl`.
- Produces: `GridSpec`, `parent_key()`, `enumerate_macro_patch_records()`, and `validate_patch_record()`.

- [ ] **Step 1: Write failing tests for exact integer bounds and owner-zone filtering**

```python
def test_enumerator_emits_exact_1280m_parent_cells():
    spec = MODULE.GridSpec(side_m=1280, macro_side_patches=10, boundary_version="test")
    macro = {"grid_epsg": 32650, "grid_id": "utm50n", "macro_col": 34, "macro_row": 344}
    boundary = box(116.28, 39.83, 116.31, 39.86)
    records = list(MODULE.enumerate_macro_patch_records(macro, boundary, spec))
    assert records
    for record in records:
        minx, miny, maxx, maxy = record["utm_bounds"]
        assert maxx - minx == 1280
        assert maxy - miny == 1280
        assert minx == record["grid_col"] * 1280
        assert miny == record["grid_row"] * 1280
        assert record["parent_key"] == (
            f'{record["grid_epsg"]}:{record["grid_col"]}:{record["grid_row"]}'
        )
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run: `pytest tests/test_china_full_grid.py::test_enumerator_emits_exact_1280m_parent_cells -v`

Expected: FAIL because `scripts/data/china_full_grid.py` does not exist.

- [ ] **Step 3: Implement the immutable grid contract and iterator**

```python
@dataclass(frozen=True)
class GridSpec:
    side_m: int = 1280
    macro_side_patches: int = 10
    boundary_version: str = "geoBoundaries-CHN-ADM0-frozen-20260726"
    atlas_version: str = "china-full-1280m-v1-20260805"

def parent_key(grid_epsg: int, grid_col: int, grid_row: int) -> str:
    return f"{grid_epsg}:{grid_col}:{grid_row}"

def enumerate_macro_patch_records(macro, boundary_wgs84, spec):
    to_wgs84 = Transformer.from_crs(macro["grid_epsg"], 4326, always_xy=True)
    for local_col in range(spec.macro_side_patches):
        for local_row in range(spec.macro_side_patches):
            grid_col = macro["macro_col"] * 10 + local_col
            grid_row = macro["macro_row"] * 10 + local_row
            minx, miny = grid_col * spec.side_m, grid_row * spec.side_m
            center_lon, center_lat = to_wgs84.transform(minx + 640, miny + 640)
            if utm_owner_epsg(center_lon, center_lat) != macro["grid_epsg"]:
                continue
            if not boundary_wgs84.covers(Point(center_lon, center_lat)):
                continue
            yield build_patch_record(macro, grid_col, grid_row, center_lon, center_lat, spec)
```

- [ ] **Step 4: Add tests for deterministic IDs, unique macro-local keys, EPSG ownership, and WGS84 axis order**

Run: `pytest tests/test_china_full_grid.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add configs/national/china_full_1280m_grid_v1_20260805.json scripts/data/china_full_grid.py tests/test_china_full_grid.py
git commit -m "feat: define canonical China 1280m parent grid"
```

### Task 2: Add Streaming GeoParquet and Shapefile Writers

**Files:**
- Modify: `scripts/data/china_full_grid.py`
- Create: `scripts/data/build_china_full_grid.py`
- Modify: `tests/test_china_full_grid.py`

**Interfaces:**
- Consumes: iterators of canonical patch records and the 62,000 sampled parent-key map.
- Produces: `write_zone_batch()`, `ZoneWriteSummary`, and CLI output partitions for `all`, `sampled`, and `unsampled`.

- [ ] **Step 1: Write a failing writer test using a two-macro synthetic boundary**

```python
def test_writer_partitions_all_sampled_and_unsampled(tmp_path):
    records = synthetic_parent_records(count=12)
    sampled = {records[1]["parent_key"], records[7]["parent_key"]}
    summary = MODULE.write_zone_records(records, sampled, tmp_path, batch_size=5)
    assert summary.all_count == 12
    assert summary.sampled_count == 2
    assert summary.unsampled_count == 10
    assert summary.sampled_count + summary.unsampled_count == summary.all_count
```

- [ ] **Step 2: Verify the writer test fails**

Run: `pytest tests/test_china_full_grid.py::test_writer_partitions_all_sampled_and_unsampled -v`

Expected: FAIL because `write_zone_records` is undefined.

- [ ] **Step 3: Implement bounded batch writing**

```python
@dataclass
class ZoneWriteSummary:
    grid_id: str
    all_count: int = 0
    sampled_count: int = 0
    unsampled_count: int = 0
    parquet_parts: list[str] = field(default_factory=list)

def write_zone_records(records, sampled_keys, output_root, batch_size=100_000):
    iterator = iter(records)
    first = next(iterator)
    summary = ZoneWriteSummary(grid_id=first["grid_id"])
    batch = [first]
    for record in iterator:
        batch.append(record)
        if len(batch) == batch_size:
            flush_zone_batch(batch, sampled_keys, output_root, summary)
            batch.clear()
    if batch:
        flush_zone_batch(batch, sampled_keys, output_root, summary)
    if summary.all_count != summary.sampled_count + summary.unsampled_count:
        raise ValueError(f"partition mismatch for {summary.grid_id}: {summary}")
    return summary
```

Use Shapefile fields `PATCH_ID`, `UTM_EPSG`, `GRID_COL`, `GRID_ROW`, `MACRO_ID`, and
`SAMPLED`. Store complete hashes, bounds, versions, and WGS84 polygon geometry in GeoParquet.

- [ ] **Step 4: Add read-back assertions for CRS, geometry, fields, and partition algebra**

Run: `pytest tests/test_china_full_grid.py -v`

Expected: PASS, including `all=sampled+unsampled` and zero shared parent keys.

- [ ] **Step 5: Commit the writers**

```bash
git add scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py
git commit -m "feat: stream national grid partitions"
```

### Task 3: Build Membership and Geometry Audits

**Files:**
- Modify: `scripts/data/china_full_grid.py`
- Modify: `scripts/data/build_china_full_grid.py`
- Modify: `tests/test_china_full_grid.py`

**Interfaces:**
- Consumes: sampled registry JSONL and generated GeoParquet metadata.
- Produces: `audit_sample_membership()` and `china_full_grid_membership_audit.json`.

- [ ] **Step 1: Write failing tests for missing, duplicated, and altered sampled cells**

```python
def test_membership_audit_requires_every_sample_exactly_once():
    atlas = ["32650:1:1", "32650:1:2", "32650:1:3"]
    sampled = ["32650:1:1", "32650:1:3"]
    audit = MODULE.audit_sample_membership(atlas, sampled)
    assert audit["matched"] == 2
    assert audit["missing"] == []
    assert audit["duplicate_atlas_keys"] == []
```

- [ ] **Step 2: Verify failure, then implement set-based and footprint audits**

The production audit must report exact counts for all, sampled, unsampled, missing sampled,
duplicate parent keys, hash mismatches, and maximum footprint coordinate difference.

- [ ] **Step 3: Add projected geometry validation**

For every zone, read generated polygons in batches, project to `grid_epsg`, and assert:

```python
abs(width_m - 1280) <= 0.001
abs(height_m - 1280) <= 0.001
abs(area_m2 - 1_638_400) <= 1.0
```

Also verify same-zone positive-area overlap is zero and cross-zone overlap does not exceed 1%.

- [ ] **Step 4: Run the complete test module**

Run: `pytest tests/test_china_full_grid.py -v`

Expected: PASS.

- [ ] **Step 5: Commit audits**

```bash
git add scripts/data/china_full_grid.py scripts/data/build_china_full_grid.py tests/test_china_full_grid.py
git commit -m "test: audit national parent grid membership"
```

### Task 4: Generate the Index, Previews, and Combined Documentation

**Files:**
- Create: `scripts/docs/build_china_full_grid_package_docx.py`
- Create: `docs/data/china_full_1280m_grid_package_readme_20260805.md`
- Modify: `scripts/data/build_china_full_grid.py`
- Modify: `tests/test_china_full_grid.py`

**Interfaces:**
- Consumes: zone summaries, macrocell counts, membership audit, and prior sampling explanation.
- Produces: macrocell GeoPackage, PNG previews, `先读我.docx`, package README, and manifest metadata.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_package_docs_explain_parent_sampled_and_unsampled(tmp_path):
    docx = build_test_docx(tmp_path)
    document = Document(docx)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "全国完整父网格" in text
    assert "已采样 patch" in text
    assert "未采样 patch" in text
    assert "EPSG:32643" in text
    assert "X=经度、Y=纬度" in text
```

- [ ] **Step 2: Implement the macrocell GeoPackage**

Create `macrocells` and `utm_zones` layers with exact all/sample/unsampled counts. Confirm the
sum of macrocell counts equals the GeoParquet all-patch count.

- [ ] **Step 3: Generate three visual checks**

1. Nationwide patch-density preview by UTM zone.
2. Local 1,280 m grid view showing sampled and unsampled cells together.
3. UTM seam view showing owner zones and the accepted overlap rule.

- [ ] **Step 4: Build the combined Chinese DOCX**

Reuse the approved formatting: Chinese SimSun 12 pt, headings 14 pt, title 16 pt, English and
numbers Times New Roman, all text black. Merge the previous sampling explanation with sections
for the parent atlas, sampled/unsampled relationship, CRS, file inventory, and audit conclusions.

- [ ] **Step 5: Run documentation and artifact tests**

Run: `pytest tests/test_china_full_grid.py -v`

Expected: PASS.

- [ ] **Step 6: Commit documentation generators**

```bash
git add scripts/docs/build_china_full_grid_package_docx.py docs/data/china_full_1280m_grid_package_readme_20260805.md scripts/data/build_china_full_grid.py tests/test_china_full_grid.py
git commit -m "docs: add national grid package guide"
```

### Task 5: Run a UTM 50N Pilot and Freeze Storage Estimates

**Files:**
- Create: `docs/reviews/china_full_grid_utm50n_pilot_20260805.md`
- Modify: `configs/national/china_full_1280m_grid_v1_20260805.json`

**Interfaces:**
- Consumes: production CLI with `--zones 50`.
- Produces: measured throughput, peak RSS, bytes per patch, and projected nationwide size/time.

- [ ] **Step 1: Run the pilot**

```bash
python scripts/data/build_china_full_grid.py \
  --config configs/national/china_full_1280m_grid_v1_20260805.json \
  --zones 50 \
  --output /data/xuannv_embedding/outputs/china_full_1280m_grid_pilot_utm50_20260805
```

- [ ] **Step 2: Validate the pilot artifacts**

Run the CLI `--verify-only` mode and independently read every Shapefile and Parquet part with
GeoPandas/Pyogrio. Confirm counts, EPSG:4326, exact UTM dimensions, and sampled membership.

- [ ] **Step 3: Record measured storage and runtime**

The review must include exact patch count, elapsed time, peak RSS, all/sample/unsampled bytes,
ZIP compression ratio, and nationwide extrapolation by exact macrocell counts.

- [ ] **Step 4: Update only measured configuration limits**

Set batch size and Shapefile split threshold from pilot evidence; retain the 1.8 GB safety cap.

- [ ] **Step 5: Commit pilot evidence**

```bash
git add docs/reviews/china_full_grid_utm50n_pilot_20260805.md configs/national/china_full_1280m_grid_v1_20260805.json
git commit -m "docs: record national grid pilot sizing"
```

### Task 6: Materialize, Verify, and Package the Nationwide Atlas

**Files:**
- Generated: `/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805/`
- Generated: `/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805.zip`
- Create: `docs/reviews/china_full_grid_final_qa_20260805.md`

**Interfaces:**
- Consumes: all completed code, frozen configuration, boundary, macro inventory, and sampled registry.
- Produces: final nationwide package and QA report.

- [ ] **Step 1: Run all unit tests before the expensive job**

Run: `pytest tests/test_national_macrocell_inventory.py tests/test_china_full_grid.py -v`

Expected: PASS.

- [ ] **Step 2: Materialize all 11 UTM zones**

```bash
python scripts/data/build_china_full_grid.py \
  --config configs/national/china_full_1280m_grid_v1_20260805.json \
  --zones 43-53 \
  --output /data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805
```

- [ ] **Step 3: Run final verification**

```bash
python scripts/data/build_china_full_grid.py \
  --config configs/national/china_full_1280m_grid_v1_20260805.json \
  --output /data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805 \
  --verify-only
```

Expected gates:

```text
sampled_matched = 62000
sampled_missing = 0
duplicate_parent_keys = 0
all_count = sampled_count + unsampled_count
sampled_unsampled_intersection = 0
invalid_geometry = 0
owner_zone_mismatch = 0
```

- [ ] **Step 4: Build checksums and archive**

Hash every package file except `SHA256SUMS.json`, then create the ZIP with all GeoParquet
partitions, 11-zone all/sample/unsampled Shapefile sets, macrocell GPKG, previews, README,
DOCX, configuration, audit, and source provenance.

- [ ] **Step 5: Independently reopen the ZIP and validate every member**

Use `zipfile.testzip()`, compare member names with the manifest, recompute hashes after extraction,
and read one Shapefile plus every GeoParquet partition from each UTM zone.

- [ ] **Step 6: Write and commit the final QA report**

```bash
git add docs/reviews/china_full_grid_final_qa_20260805.md
git commit -m "docs: verify national 1280m grid package"
```
