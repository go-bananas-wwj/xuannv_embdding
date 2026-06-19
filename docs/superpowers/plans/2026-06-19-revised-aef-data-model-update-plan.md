# AEF 数据与模型更新：修订实施计划（训练前修复优先）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在启动任何训练之前，先修复海淀区错误 PC 数据、重写并加速 patch 提取流程、补齐多源 patch 与 manifest，然后再进入模型架构改造与最小规模训练。

**Architecture:** 数据层先干净；模型层按设计稿改为逐月 embedding、无 L2 bottleneck、原生分辨率 highres 独立模态；训练层改为逐月 masked loss + 多 NPU DDP/AMP/checkpointing。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, xarray, rasterio, rioxarray, pytest, DDP.

---

## 阶段 A：数据修复与预处理加速（必须先完成）

### Task A0: 修复 `download_pc.py` 的 S1/Landsat 下载 bug

**Files:**
- Modify: `scripts/data/download_pc.py`
- Test: 对单个月份的 S1/Landsat 做端到端下载测试，验证非零像素比例

- [ ] **Step 1: 修改 GDAL 扩展名白名单**

```python
# 原配置会拦截 .tiff 后缀的 S1 RTC asset
# os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.jp2")
```

- [ ] **Step 2: 关闭 `errors_as_nodata` 静默填值，fill_value 改为 NaN**

```python
import numpy as np

ds = stackstac.stack(
    items,
    bounds_latlon=bbox,
    resolution=res,
    epsg=epsg,
    dtype="float32",
    fill_value=np.float32(np.nan),
    rescale=False,
    assets=assets,
    errors_as_nodata=(),
)
```

- [ ] **Step 3: 固定 EPSG 为 AOI 中心点所在 UTM 带**

```python
def _get_aoi_crs(aoi: gpd.GeoDataFrame) -> int:
    """返回 AOI 中心点所在 UTM 带的 EPSG 代码。"""
    centroid = aoi.to_crs("EPSG:4326").geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg

# 在 download_source 中替换
# epsg = _extract_epsg(items[0])
epsg = _get_aoi_crs(aoi)
```

- [ ] **Step 4: Landsat 增加 `qa_pixel` 波段**

```python
"landsat": [
    "blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"
]
```

- [ ] **Step 5: 缩短下载窗口并重新 sign items**

- S1/Landsat 建议按月或 1–2 个月切分；
- 在 `stackstac.stack` 前再次调用 `planetary_computer.sign_inplace(items)` 刷新 SAS token。

- [ ] **Step 6: 增加下载后覆盖率校验**

```python
def _validate_coverage(ds, min_valid_ratio=0.05):
    var = list(ds.data_vars)[0]
    arr = ds[var]
    for t in range(arr.sizes["time"]):
        band0 = arr.isel(time=t, band=0)
        valid = np.isfinite(band0.values).mean()
        if valid < min_valid_ratio:
            raise RuntimeError(f"time {t} coverage {valid:.2%} < {min_valid_ratio:.2%}")
```

- [ ] **Step 7: Commit & push**

```bash
git add scripts/data/download_pc.py
git commit -m "fix: PC download script S1/Landsat silent zero-fill and UTM zone mismatch"
git push origin feat/data-and-model-rework
```

---

### Task A1: 删除错误的海淀区 PC 数据并重新下载

**Files:**
- Modify: `scripts/data/download_pc_haidian.sh`（新建）
- Modify: `scripts/data/download_pc_parallel.sh`（可选：加入海淀任务）
- Test: 抽检 `raw/haidian/s2/*.nc` 非零像素比例

- [ ] **Step 1: 确认需要删除的文件**

```bash
ls -lh /data/xuannv_embedding/raw/haidian/s2/*.nc
ls -lh /data/xuannv_embedding/raw/haidian/s1/*.nc
ls -lh /data/xuannv_embedding/raw/haidian/landsat/landsat_20250101_20251231.nc
```

- [ ] **Step 2: 备份文件名后删除**

```bash
mkdir -p /data/xuannv_embedding/raw/_trash_2026-06-19
mv /data/xuannv_embedding/raw/haidian/s2/s2_20250101_20251231.nc \
   /data/xuannv_embedding/raw/haidian/s1/s1_20250101_20251231.nc \
   /data/xuannv_embedding/raw/haidian/s1/s1_20260101_20260531.nc \
   /data/xuannv_embedding/raw/haidian/landsat/landsat_20250101_20251231.nc \
   /data/xuannv_embedding/raw/_trash_2026-06-19/
# 注意：s2_20260101_20260531.nc 也异常，一并移走
mv /data/xuannv_embedding/raw/haidian/s2/s2_20260101_20260531.nc \
   /data/xuannv_embedding/raw/_trash_2026-06-19/
```

- [ ] **Step 3: 创建海淀下载脚本**

`scripts/data/download_pc_haidian.sh`：

```bash
#!/usr/bin/env bash
set -e
LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

download() {
    local source=$1 start=$2 end=$3 log=$4
    echo "[$(date)] START haidian $source $start..$end"
    python "$SCRIPT_DIR/download_pc.py" \
        --region haidian \
        --source "$source" \
        --start "$start" \
        --end "$end" \
        --region-file "$SCRIPT_DIR/../../configs/regions/haidian.geojson" \
        --output-root /data/xuannv_embedding/raw \
        --workers 12 \
        > "$log" 2>&1
    echo "[$(date)] DONE haidian $source $start..$end"
}

# S2 2025 按季度，2026 单独
download s2 2025-01-01 2025-03-31 "$LOG_DIR/download_pc_haidian_s2_2025q1.log" &
download s2 2025-04-01 2025-06-30 "$LOG_DIR/download_pc_haidian_s2_2025q2.log" &
download s2 2025-07-01 2025-09-30 "$LOG_DIR/download_pc_haidian_s2_2025q3.log" &
download s2 2025-10-01 2025-12-31 "$LOG_DIR/download_pc_haidian_s2_2025q4.log" &
download s2 2026-01-01 2026-05-31 "$LOG_DIR/download_pc_haidian_s2_2026.log" &

# S1 按月下载，避免 token 过期
download s1 2025-01-01 2025-01-31 "$LOG_DIR/download_pc_haidian_s1_202501.log" &
download s1 2025-02-01 2025-02-28 "$LOG_DIR/download_pc_haidian_s1_202502.log" &
# ... 继续到 2026-05（可循环生成）

# Landsat 按月下载
download landsat 2025-01-01 2025-01-31 "$LOG_DIR/download_pc_haidian_landsat_202501.log" &
# ... 继续到 2026-05

wait

echo "[$(date)] 海淀区 PC 下载完成"
```

- [ ] **Step 4: 执行下载**

```bash
chmod +x scripts/data/download_pc_haidian.sh
nohup scripts/data/download_pc_haidian.sh > /data/xuannv_embedding/logs/download_pc_haidian_all.log 2>&1 &
```

- [ ] **Step 5: 抽检验证**

```bash
python3 - <<'PY'
import xarray as xr, glob, numpy as np
for source, pattern in [
    ('s2', '/data/xuannv_embedding/raw/haidian/s2/*.nc'),
    ('s1', '/data/xuannv_embedding/raw/haidian/s1/*.nc'),
    ('landsat', '/data/xuannv_embedding/raw/haidian/landsat/*.nc'),
]:
    for f in sorted(glob.glob(pattern))[:2]:
        ds = xr.open_dataset(f, chunks={'time':1})
        v = list(ds.data_vars)[0]
        nz = (ds[v].isel(band=0) != 0).sum(dim=['y','x']).compute()
        print(source, f, 'nonzero min/median/max', nz.values.min(), int(np.median(nz.values)), nz.values.max())
PY
```

Expected: S2 中位数非零像素 >> 0（例如 > 100 万）。

- [ ] **Step 6: Commit & push 脚本**

```bash
git add scripts/data/download_pc_haidian.sh
git commit -m "data: add Haidian PC re-download script for corrupted 2025/2026 data"
git push origin feat/data-and-model-rework
```

---

### Task A2: 重写并加速 patch 提取流程

**Files:**
- Rewrite: `scripts/data/preprocess.py`
- Modify: `scripts/data/preprocess_harbin.json`, `scripts/data/preprocess_haidian.json`
- Test: 小批量 benchmark 对比旧/新流程

- [ ] **Step 1: 设计新的 `preprocess.py` 主流程**

核心改变：
1. 读取 AOI GeoJSON，按 `patch_size * resolution` 生成 patch 网格。
2. 对每个 `(nc_file, time_idx)`：
   - 一次性加载整时相到内存；
   - 按整数像素窗口批量切片；
   - 用 `np.pad` 处理边缘；
   - 写出为 COG GeoTIFF 或普通 GeoTIFF。
3. 跳过有效像素比例低于 `min_valid_ratio` 的 patch。
4. 写出 per-pixel valid mask（可选，`_mask.tif`）。

- [ ] **Step 2: 编写核心函数**

```python
def generate_patch_grid(aoi_path, patch_size, resolution, dst_crs):
    """基于 AOI 生成 patch 中心点/边界的 GeoDataFrame。"""
    import geopandas as gpd
    from shapely.geometry import box

    aoi = gpd.read_file(aoi_path).to_crs(dst_crs)
    bounds = aoi.total_bounds
    xmin, ymin, xmax, ymax = bounds
    size = patch_size * resolution

    patches = []
    ids = []
    x0 = xmin
    y0 = ymin
    for iy, y in enumerate(np.arange(ymax, ymin, -size)):
        for ix, x in enumerate(np.arange(xmin, xmax, size)):
            patch_id = f"{region}_{iy:04d}_{ix:04d}"
            patches.append(box(x, y - size, x + size, y))
            ids.append(patch_id)

    gdf = gpd.GeoDataFrame({'patch_id': ids}, geometry=patches, crs=dst_crs)
    return gdf[gdf.intersects(aoi.unary_union)]


def process_one_time(ds, data_var, time_idx, patch_gdf, dst_dir, nodata=0.0, min_valid_ratio=0.1):
    """对一个时间步做批量切片并写出。"""
    import rasterio
    from rasterio.transform import from_bounds

    arr = ds[data_var].isel(time=time_idx).values.astype(np.float32)  # (C, H, W)
    timestamp = ds['time'].values[time_idx]
    transform = ds.rio.transform() if hasattr(ds, 'rio') else None
    crs = ds.rio.crs if hasattr(ds, 'rio') else patch_gdf.crs

    for _, row in patch_gdf.iterrows():
        bounds = row.geometry.bounds
        # 计算像素窗口（假设 arr 已对齐到同一网格）
        window = rasterio.windows.from_bounds(*bounds, transform=transform)
        col_off, row_off, w, h = map(int, [window.col_off, window.row_off, window.width, window.height])

        # 边界填充
        pad_left = max(0, -col_off)
        pad_top = max(0, -row_off)
        pad_right = max(0, col_off + w - arr.shape[-1])
        pad_bottom = max(0, row_off + h - arr.shape[-2])

        if pad_left or pad_top or pad_right or pad_bottom:
            slice_arr = arr[:, max(0, row_off):min(row_off+h, arr.shape[-2]),
                              max(0, col_off):min(col_off+w, arr.shape[-1])]
            slice_arr = np.pad(slice_arr, ((0,0),(pad_top, pad_bottom),(pad_left, pad_right)),
                               mode='constant', constant_values=nodata)
        else:
            slice_arr = arr[:, row_off:row_off+h, col_off:col_off+w]

        valid_ratio = float(np.mean(slice_arr[0] != nodata))
        if valid_ratio < min_valid_ratio:
            continue

        out_path = dst_dir / f"{row['patch_id']}_{_timestamp_str(timestamp)}.tif"
        profile = {
            'driver': 'GTiff',
            'height': slice_arr.shape[-2],
            'width': slice_arr.shape[-1],
            'count': slice_arr.shape[-3],
            'dtype': slice_arr.dtype,
            'crs': crs,
            'transform': from_bounds(*bounds, slice_arr.shape[-1], slice_arr.shape[-2]),
            'nodata': nodata,
            'compress': 'deflate',
            'tiled': True,
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(slice_arr)
```

- [ ] **Step 3: 加入多进程**

```python
from concurrent.futures import ProcessPoolExecutor

def process_file(nc_path, config):
    ds = xr.open_dataset(nc_path, chunks={'time': 1})
    data_var = list(ds.data_vars)[0]
    for t in range(ds.sizes['time']):
        process_one_time(ds, data_var, t, patch_gdf, dst_dir)
    ds.close()

# 主函数中
with ProcessPoolExecutor(max_workers=config.get('workers', 8)) as executor:
    list(executor.map(lambda p: process_file(p, config), nc_files))
```

- [ ] **Step 4: 更新预处理配置**

`scripts/data/preprocess_harbin.json`：

```json
{
  "region": "harbin",
  "patch_size": 128,
  "resolution": 10,
  "aoi": "configs/regions/harbin.geojson",
  "min_valid_ratio": 0.05,
  "workers": 8,
  "sources": {
    "s2": {"raw_glob": "raw/harbin/s2/*.nc", "bands": 12},
    "s1": {"raw_glob": "raw/harbin/s1/*.nc", "bands": 2},
    "landsat": {"raw_glob": "raw/harbin/landsat/*.nc", "bands": 6}
  }
}
```

- [ ] **Step 5: 小批量 benchmark**

```bash
# 旧流程（单个文件 1 个时间步，100 patches）
time python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source s2 --max-times 1 --max-patches 100

# 新流程（同上）
time python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source s2 --max-times 1 --max-patches 100
```

Expected: 新流程耗时 < 旧流程的 1/50。

- [ ] **Step 6: Commit & push**

```bash
git add scripts/data/preprocess.py scripts/data/preprocess_harbin.json scripts/data/preprocess_haidian.json
git commit -m "refactor: speed up patch extraction with per-time-slice slicing + multiprocessing"
git push origin feat/data-and-model-rework
```

---

### Task A3: 生成 S1 / Landsat patch

**Files:**
- Modify: `scripts/data/preprocess.py`（已完成通用支持）
- Modify: `scripts/data/preprocess_all.sh`
- Test: 检查 `processed/*/patches/s1/`、`processed/*/patches/landsat/`

- [ ] **Step 1: 运行 S1 / Landsat 预处理**

```bash
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source s1
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source landsat
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --source s1
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --source landsat
```

- [ ] **Step 2: 统计有效 patch 数**

```bash
find processed/harbin/patches/s1 -name '*.tif' | wc -l
find processed/harbin/patches/landsat -name '*.tif' | wc -l
find processed/haidian/patches/s1 -name '*.tif' | wc -l
find processed/haidian/patches/landsat -name '*.tif' | wc -l
```

- [ ] **Step 3: Commit & push**

```bash
git add scripts/data/preprocess_all.sh
git commit -m "data: generate s1/landsat patches for harbin and haidian"
git push origin feat/data-and-model-rework
```

---

### Task A4: 高分辨率光学 patch 生成

**Files:**
- Create: `scripts/data/preprocess_highres.py`
- Create: `scripts/data/preprocess_highres_harbin.json`
- Create: `scripts/data/preprocess_highres_haidian.json`
- Test: 检查 `processed/*/patches/highres_optical/`

- [ ] **Step 1: 哈尔滨 DOM 预处理**

```bash
# 重投影 CGCS2000 -> EPSG:32652
gdalwarp -s_srs EPSG:4490 -t_srs EPSG:32652 -tr 0.5 0.5 -r bilinear \
  raw/harbin/highres_optical/dom_202512.tif \
  raw/harbin/highres_optical/_mosaic/dom_202512_32652.tif

python scripts/data/preprocess_highres.py --config scripts/data/preprocess_highres_harbin.json
```

- [ ] **Step 2: 海淀 PlanetScene 镶嵌 + 裁剪**

```bash
# 按月份镶嵌（示例 2025-06）
gdalbuildvrt raw/haidian/highres_optical/_mosaic/202506.vrt raw/haidian/highres_optical/*_202506*.tif
gdal_translate -of GTiff raw/haidian/highres_optical/_mosaic/202506.vrt raw/haidian/highres_optical/_mosaic/202506.tif

python scripts/data/preprocess_highres.py --config scripts/data/preprocess_highres_haidian.json
```

- [ ] **Step 3: Commit & push**

```bash
git add scripts/data/preprocess_highres.py scripts/data/preprocess_highres_harbin.json scripts/data/preprocess_highres_haidian.json
git commit -m "data: generate native-resolution highres optical patches"
git push origin feat/data-and-model-rework
```

---

### Task A5: 修复 WorldCover 类别映射（ESRI 2023 → 连续 9 类）

**Files:**
- Modify: `scripts/data/preprocess_lulc.py`
- Modify: `configs/base.yaml`
- Test: `tests/test_lulc.py`

- [ ] **Step 1: 在 `preprocess_lulc.py` 中加入重映射**

```python
import numpy as np

# ESRI 2023 WorldCover 原始值 → 连续训练索引
ESRI2023_REMAP = {
    0: 0,   # No Data / 背景（ignore_index）
    1: 1,   # Water
    2: 2,   # Trees
    4: 3,   # Flooded Vegetation
    5: 4,   # Crops
    7: 5,   # Built Area
    8: 6,   # Bare Ground
    9: 7,   # Snow/Ice
    10: 8,  # Clouds
    11: 9,  # Rangeland
}


def remap_esri2023(arr: np.ndarray) -> np.ndarray:
    """将 ESRI 2023 原始标签重映射为连续索引，未知值设为 0（nodata）。"""
    out = np.zeros_like(arr, dtype=np.uint8)
    for old, new in ESRI2023_REMAP.items():
        out[arr == old] = new
    return out
```

在切 patch 循环中调用：

```python
label = src.read(1)
label = remap_esri2023(label)
# 写出
```

- [ ] **Step 2: 修改配置**

```yaml
model:
  target_heads:
    lulc:
      loss_type: categorical
      channels: 10          # 0=ignore, 1-9 为有效类别
      ignore_index: 0
      class_weight: "inverse_sqrt"
```

- [ ] **Step 3: 更新损失函数支持 class weight / ignore_index**

```python
import torch.nn.functional as F

def masked_cross_entropy(pred, target, mask, weight=None, ignore_index=0):
    # pred: (B, T, C, H, W), target: (B, T, H, W)
    B, T, C, H, W = pred.shape
    pred = pred.reshape(B * T, C, H * W).transpose(1, 2)
    target = target.reshape(B * T, H * W)
    loss = F.cross_entropy(
        pred, target,
        weight=weight,
        ignore_index=ignore_index,
        reduction="none",
    )
    mask = mask.reshape(B * T, H * W)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
```

- [ ] **Step 4: 测试**

```python
def test_lulc_remap():
    import numpy as np
    from scripts.data.preprocess_lulc import remap_esri2023
    arr = np.array([[0,1,2,4,5,7,8,9,10,11]])
    out = remap_esri2023(arr)
    expected = np.array([[0,1,2,3,4,5,6,7,8,9]])
    assert np.array_equal(out, expected)
```

Run: `pytest tests/test_lulc.py -v`
Expected: PASS

- [ ] **Step 5: Commit & push**

```bash
git add scripts/data/preprocess_lulc.py configs/base.yaml tests/test_lulc.py
 git commit -m "fix: remap ESRI 2023 WorldCover to continuous 9-class labels"
git push origin feat/data-and-model-rework
```

---

### Task A6: 升级 Manifest 与统计量

**Files:**
- Modify: `scripts/data/generate_manifest.py`
- Create/Modify: `scripts/data/compute_statistics.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: 扩展 manifest 格式**

```python
SOURCES = ["s2", "s1", "landsat", "highres_optical", "highres_sar"]

def build_manifest_entry(patch_id, region, aoi_gdf):
    entry = {
        "patch_id": patch_id,
        "region": region,
        "crs": str(aoi_gdf.crs),
        "bounds": list(aoi_gdf[aoi_gdf['patch_id']==patch_id].geometry.bounds.values[0]),
        "sources": {},
    }
    for source in SOURCES:
        months = extract_months_for_patch(source, patch_id)
        entry["sources"][source] = {"months": sorted(months), "count": len(months)}
    return entry
```

- [ ] **Step 2: 生成统计量**

```python
def compute_source_statistics(patch_dir, source):
    files = list(Path(patch_dir).glob('*.tif'))
    # 随机采样 1000 个 patch 计算均值/标准差
    ...
    return {"mean": ..., "std": ...}
```

- [ ] **Step 3: 重新生成**

```bash
python scripts/data/generate_manifest.py --regions harbin,haidian
python scripts/data/compute_statistics.py --regions harbin,haidian
```

- [ ] **Step 4: 测试**

```bash
pytest tests/test_manifest.py -v
```

Expected: PASS

- [ ] **Step 5: Commit & push**

```bash
git add scripts/data/generate_manifest.py scripts/data/compute_statistics.py tests/test_manifest.py
git commit -m "data: multi-source monthly manifest and per-source statistics"
git push origin feat/data-and-model-rework
```

---

## 阶段 B：模型改造与最小验证训练

阶段 B 沿用 `docs/superpowers/plans/2026-06-17-aef-data-model-update-plan.md` 中的 Task 4–13，但执行前提是阶段 A 全部完成且 `pytest` 通过。

### Task B1: 无 L2 归一化 Bottleneck

- 实现 `LearnedBottleneck`（见原 Task 4）。

### Task B2: 逐月 Embedding 模块

- 实现 `MonthlyEmbeddingModule`（见原 Task 5）。

### Task B3: 高分辨率原生分辨率编码器

- 实现 `NativeResolutionHighResEncoder`（见原 Task 6）。

### Task B4: 逐月解码器头

- 改造 `ContinuousDecoder` / `CategoricalDecoder`（见原 Task 7）。

### Task B5: AEFModel 整合

- 替换 `TemporalSummarizer`，支持月度 embedding 与高分辨率分支（见原 Task 8）。

### Task B6: 训练 Batch 准备与逐月 Loss

- 改造 `prepare_batch` 与 `losses.py`（见原 Task 9）。

### Task B7: 数据集月度窗口采样（连续窗口 + 随机起始 + Temporal Dropout）

**Files:**
- Modify: `src/xuannv_embedding/data/dataset.py`
- Modify: `src/xuannv_embedding/training/batch_preparation.py`
- Modify: `src/xuannv_embedding/config.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: 实现 `TemporalSampler`**

```python
import random
from typing import List


class TemporalSampler:
    """在可用月份序列中采样连续窗口，并可选做 temporal dropout。"""

    def __init__(self, window_months: int = 12, dropout: float = 0.0):
        self.window_months = window_months
        self.dropout = dropout

    def sample(self, available_months: List[int]) -> tuple[List[int], List[int]]:
        available = sorted(available_months)
        if len(available) <= self.window_months:
            window = available
        else:
            max_start = len(available) - self.window_months
            start_idx = random.randint(0, max_start)
            window = available[start_idx:start_idx + self.window_months]

        # temporal dropout：随机丢弃部分月份（保留索引位置用 mask 表示）
        mask = [1] * len(window)
        if self.dropout > 0 and len(window) > 1:
            for i in range(len(window)):
                if random.random() < self.dropout:
                    mask[i] = 0
        return window, mask
```

- [ ] **Step 2: 在 `AEFDataset.__getitem__` 中使用**

```python
sampler = TemporalSampler(
    window_months=self.cfg.temporal_window_months,
    dropout=self.cfg.temporal_dropout,
)
window, mask = sampler.sample(available_months)
# 加载 window 内各 source 的帧；缺失月份补 0 且 mask=0
```

- [ ] **Step 3: 在 `prepare_batch` 中对齐多源时间戳**

- 用全局 `window` 月度网格作为 `timestamps`；
- 每个 source 独立构造 `(B, T, C, H, W)`，缺失月份补 0 + `source_masks` 对应位置 0；
- 高分辨率 source 仍聚合为单帧，但记录其所在月份（若无则全窗口缺失）。

- [ ] **Step 4: 配置**

```python
@dataclass
class DataConfig:
    temporal_window_months: int = 12
    temporal_dropout: float = 0.0
    use_full_sequence: bool = False
```

- [ ] **Step 5: 测试**

```python
def test_temporal_sampler():
    from src.xuannv_embedding.data.dataset import TemporalSampler
    sampler = TemporalSampler(window_months=12, dropout=0.0)
    months = list(range(202501, 202513)) + list(range(202601, 202606))
    window, mask = sampler.sample(months)
    assert len(window) == 12
    assert all(m2 > m1 for m1, m2 in zip(window, window[1:]))
    assert mask == [1] * 12
```

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 6: Commit & push**

```bash
git add src/xuannv_embedding/data/dataset.py \
        src/xuannv_embedding/training/batch_preparation.py \
        src/xuannv_embedding/config.py \
        tests/test_dataset.py
git commit -m "feat: continuous monthly window sampling with random start and temporal dropout"
git push origin feat/data-and-model-rework
```

### Task B8: 多 NPU 训练优化

- AMP + gradient checkpointing + DDP（见原 Task 11）。

### Task B9: 配置更新

- 更新 `configs/base.yaml`、`harbin_128.yaml`、`haidian_128.yaml`（见原 Task 12）。

### Task B10: 最终集成测试与最小规模训练

- `pytest` 全绿；
- 哈尔滨 + 海淀一起训练，前 100 step 不 OOM、loss 下降。

---

## 训练前 Checklist（必须全部打勾才能启动训练）

- [ ] `download_pc.py` 的 S1/Landsat 下载 bug 已修复
- [ ] 海淀区 S2/S1/Landsat 2025 与 S2/S1 2026 重新下载且非零像素正常
- [ ] `preprocess.py` 已重写并通过 benchmark（加速 >= 10×）
- [ ] S1/Landsat/高分辨率光学 patch 已生成
- [ ] WorldCover 已重映射为连续 9 类并配置 `ignore_index=0`
- [ ] Manifest 已升级为多源逐月格式
- [ ] 统计量已重新生成
- [ ] `pytest tests/` 全绿
- [ ] 模型输出形状为 `(B, T_month, D, H, W)`
- [ ] 训练脚本可在 6× NPU 上启动且不 OOM

---

## 已确认决策

| 问题 | 决策 |
|---|---|
| No-data 值 | **保持 0**，通过独立 valid mask 标识 |
| S1 / Landsat | **保留并修复下载 bug**；S1 强烈推荐，Landsat 设为可选源 |
| 时序采样 | **连续 12 个月窗口 + 随机起始 + Temporal Dropout**，不从 17 个月随机挑 |
| 高分辨率 SAR | 待用户确认（建议最小验证阶段跳过） |
| WorldCover | **重映射为连续 9 类**，0 为 `ignore_index`，class 11 保留为 Rangeland |

---

## 执行方式选择

1. **Subagent-Driven（推荐）**：每个 Task 派一个子 agent 实现，我每步 review。
2. **Inline Execution**：在当前会话中批量执行，关键 checkpoint 暂停确认。

**请确认：**
1. 是否按此修订计划先修复数据，再进入模型改造？
2. 高分辨率 SAR 是否跳过？
3. 选择哪种执行方式？
