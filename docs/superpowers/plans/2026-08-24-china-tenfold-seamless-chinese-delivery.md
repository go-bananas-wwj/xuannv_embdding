# China Tenfold Seamless Chinese Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the national 1280 m grid into a seamless, China-clipped, Chinese-named ten-region local delivery package.

**Architecture:** A new grid builder will create deterministic 1280 m cells in each UTM zone, clip each cell to the unique `China ∩ UTM-zone` ownership mask, and export WGS84 geometry. A spatial partitioner will assign all rebuilt cells to ten non-overlapping regions, then generate Chinese-named Shapefile and GeoPackage delivery files plus QA evidence.

**Tech Stack:** Python 3.11, PyArrow, Shapely 2, GeoPandas, Pyogrio, PyProj, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-china-tenfold-seamless-chinese-delivery-design.md`

## Global Constraints

- Keep 1280 m UTM grid anchoring at `(0, 0)` for EPSG:32643 through EPSG:32653.
- Use `geoBoundaries-CHN-ADM0-frozen-20260726` as the only China boundary source.
- Clip every grid cell to the mutually exclusive `China ∩ UTM-zone` ownership geometry before export.
- Export all geometry in EPSG:4326; edge cells may be clipped, inner cells remain 1280 m × 1280 m.
- Keep data under `/data/xuannv_embedding/outputs/`; do not overwrite V4 or upload ModelScope.
- User-facing output folders, filenames, README and QA documents must use Chinese.
- Every `.shp` must retain same-stem `.shx`, `.dbf`, `.prj` and `.cpg` companions.
- Commit and push each completed code/documentation step.

---

### Task 1: Build and test seamless UTM ownership masks

**Files:**
- Create: `scripts/data/rebuild_china_full_grid_seamless.py`
- Create: `tests/test_rebuild_china_full_grid_seamless.py`

**Interfaces:**
- Consumes: China ADM0 GeoJSON and UTM zone number `43..53`.
- Produces: `build_utm_owner_masks(china_geometry, zones) -> dict[int, BaseGeometry]`.

- [ ] **Step 1: Write the failing test**

```python
def test_owner_masks_cover_china_without_positive_area_overlap():
    masks = build_utm_owner_masks(china, [48, 49])
    assert masks[48].intersection(masks[49]).area == 0
    assert masks[48].union(masks[49]).equals(china)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py::test_owner_masks_cover_china_without_positive_area_overlap -v`

Expected: FAIL because the module and `build_utm_owner_masks` do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_utm_owner_masks(china_geometry, zones):
    return {
        zone: china_geometry.intersection(zone_longitude_polygon(zone))
        for zone in zones
    }
```

`zone_longitude_polygon(zone)` spans `[zone * 6 - 186, zone * 6 - 180]` longitude and a latitude range that contains the China boundary.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/rebuild_china_full_grid_seamless.py tests/test_rebuild_china_full_grid_seamless.py
git commit -m "feat: build seamless UTM ownership masks"
git push
```

### Task 2: Generate clipped 1280 m cells and verify seam behavior

**Files:**
- Modify: `scripts/data/rebuild_china_full_grid_seamless.py`
- Modify: `tests/test_rebuild_china_full_grid_seamless.py`

**Interfaces:**
- Consumes: a UTM owner mask, EPSG code, `cell_size_m=1280`.
- Produces: `build_clipped_grid_cells(owner_mask_wgs84, epsg, cell_size_m) -> GeoDataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
def test_clipped_cells_cover_owner_mask_without_outside_area():
    cells = build_clipped_grid_cells(owner_mask, epsg=32648, cell_size_m=1280)
    assert owner_mask.difference(cells.geometry.union_all()).area < 1e-12
    assert cells.geometry.union_all().difference(owner_mask).area < 1e-12

def test_utm_boundary_cells_have_no_positive_area_overlap_or_gap():
    west = build_clipped_grid_cells(west_owner_mask, epsg=32648, cell_size_m=1280)
    east = build_clipped_grid_cells(east_owner_mask, epsg=32649, cell_size_m=1280)
    assert west.geometry.union_all().intersection(east.geometry.union_all()).area == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py -v`

Expected: FAIL because `build_clipped_grid_cells` does not exist.

- [ ] **Step 3: Write minimal implementation**

Generate integer row/column ranges from the projected owner-mask bounds, construct UTM 1280 m boxes, retain boxes intersecting the mask, clip retained boxes to the mask, then transform polygons to EPSG:4326. Persist `parent_key`, `grid_id`, `grid_epsg`, `grid_col`, `grid_row`, `utm_bounds`, `longitude`, and `latitude`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/rebuild_china_full_grid_seamless.py tests/test_rebuild_china_full_grid_seamless.py
git commit -m "feat: generate China-clipped seamless UTM cells"
git push
```

### Task 3: Add spatial tenfold assignment and Chinese export contract

**Files:**
- Modify: `scripts/data/rebuild_china_full_grid_seamless.py`
- Modify: `tests/test_rebuild_china_full_grid_seamless.py`

**Interfaces:**
- Consumes: rebuilt cell table with `parent_key` and centroids.
- Produces: `assign_tenfold_regions(cells) -> GeoDataFrame` and `chinese_delivery_paths(root, shard_id) -> dict[str, Path]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_tenfold_assignment_is_disjoint_and_balanced():
    assigned = assign_tenfold_regions(cells)
    assert assigned.parent_key.is_unique
    assert assigned.shard_id.between(1, 10).all()
    assert assigned.groupby('shard_id').size().max() - assigned.groupby('shard_id').size().min() <= 1

def test_chinese_delivery_paths_use_requested_names(tmp_path):
    paths = chinese_delivery_paths(tmp_path, 1)
    assert paths['grid_shp'].name == '第01区_1280米网格.shp'
    assert paths['boundary_shp'].name == '第01区边界.shp'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py -v`

Expected: FAIL because assignment and Chinese-path functions do not exist.

- [ ] **Step 3: Write minimal implementation**

Reuse the confirmed V4 spatial traversal order, assign equal-size consecutive contiguous slices, and remap labels so the former spatial position 8 becomes `第01区`. Define Chinese root names `全国十个区域边界` and `全国网格覆盖范围` and per-zone paths under `第01区` through `第10区`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/rebuild_china_full_grid_seamless.py tests/test_rebuild_china_full_grid_seamless.py
git commit -m "feat: add seamless tenfold Chinese export layout"
git push
```

### Task 4: Materialize V5 and run spatial QA

**Files:**
- Modify: `scripts/data/rebuild_china_full_grid_seamless.py`
- Create: `docs/production/china_tenfold_seamless_v5_qa.md`

**Interfaces:**
- Consumes: China ADM0 GeoJSON and Task 1–3 functions.
- Produces: `/data/xuannv_embedding/outputs/china_full_grid_tenfold_delivery_v5_20260824/` and a ZIP archive.

- [ ] **Step 1: Write the failing test**

```python
def test_delivery_manifest_marks_clipped_seamless_grid(tmp_path):
    manifest = build_delivery_manifest(tmp_path, total_cells=20, shard_counts=[2] * 10)
    assert manifest['grid_method'] == 'utm_owner_mask_clipped_1280m_v1'
    assert manifest['language'] == 'zh-CN'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rebuild_china_full_grid_seamless.py::test_delivery_manifest_marks_clipped_seamless_grid -v`

Expected: FAIL because `build_delivery_manifest` does not exist.

- [ ] **Step 3: Write minimal implementation and generate data**

Write Chinese Shapefile sidecars and GeoPackage layers for every region grid and boundary, root ten-region and national boundaries, Chinese README/QA, manifest, overview image and ZIP. Generate into the V5 path only.

- [ ] **Step 4: Run spatial and archive verification**

Run a reproducible check that asserts:

```python
assert total_cells == sum(shard_counts)
assert all_geometries_valid
assert no_cell_area_outside_china
assert china_minus_grid_union_m2 <= 1.0
assert cross_zone_positive_overlap_m2 <= 1.0
assert seam_gap_m2 <= 1.0
assert zipfile.ZipFile(archive).testzip() is None
```

- [ ] **Step 5: Commit documentation and push**

```bash
git add docs/production/china_tenfold_seamless_v5_qa.md
git commit -m "data: document seamless Chinese tenfold delivery"
git push
```
