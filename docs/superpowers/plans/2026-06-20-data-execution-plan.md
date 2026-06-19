# AEF 数据修复与预处理：详细执行计划（含每步自检与子智能体审查）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在启动模型改造与训练之前，完成全部数据修复、补齐、预处理与质量验证。每完成一个阶段必须运行自动化 QA 脚本，并调用子智能体审查，通过后才能进入下一阶段。

**Scope of this plan:** 仅覆盖数据层（阶段 A）。模型改造（阶段 B）沿用 `docs/superpowers/plans/2026-06-19-revised-aef-data-model-update-plan.md` 中的 Task B1–B10。

**Confirmed decisions from user:**
- No-data value: `0` with independent valid mask.
- S1 / Landsat: both kept and bug-fixed.
- High-res optical: Harbin 0.5m DOM + Haidian PlanetScene.
- High-res SAR: not skipped; use radiometrically-calibrated ORG products from ModelScope.
- WorldCover: ignore Clouds (class 10) and NoData (class 0); remap remaining 8 valid classes to contiguous indices 1–8.
- Temporal sampling: variable-length contiguous windows (2/3/6/12 months) within a fixed 17-month grid, masked inactive months.
- Execution: Subagent-Driven.

---

## 0. 总体流程与 QA Gate 架构

```
Gate 0: 修复 download_pc.py bug
Gate 1: 重新下载 Haidian S2/S1/Landsat → qa_download.py → subagent review
Gate 2: 重新下载 Harbin S1/Landsat 2025/2026 → qa_download.py → subagent review
Gate 3: 重写 preprocess.py（整数切片 + 多进程 + valid mask）→ benchmark + qa_patch.py → subagent review
Gate 4: 生成 S2/S1/Landsat patches for Harbin + Haidian → qa_patch.py → subagent review
Gate 5: 预处理高分辨率光学（DOM + PlanetScene）→ qa_highres_optical.py → subagent review
Gate 6: 预处理高分辨率 SAR（ORG 产品）→ qa_highres_sar.py → subagent review
Gate 7: 重新处理 WorldCover 标签（remap，ignore Clouds）→ qa_label.py → subagent review
Gate 8: 升级 manifest（多源逐月可用性）→ qa_manifest.py → subagent review
Gate 9: 重新计算统计量 → qa_statistics.py → subagent review
Gate 10: 端到端冒烟测试（dataset → collate → model forward）→ pytest → subagent review
```

每个 Gate 的产出：
- 自动化 QA 报告（JSON）
- 可视化抽检样本图
- 子智能体审查意见
- 修复记录（若有问题）

---

## Gate 0: 修复 `download_pc.py` 的 S1/Landsat 下载 bug

**Files:**
- Modify: `scripts/data/download_pc.py`
- Test: 对单个月份的 S1/Landsat 做端到端下载测试

### Step 0.1: 修改 GDAL 扩展名白名单

```python
# scripts/data/download_pc.py
# 原配置会拦截 .tiff 后缀的 S1 RTC asset
os.environ.setdefault(
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS",
    ".tif,.tiff,.TIF,.jp2,.TIFf",
)
```

### Step 0.2: 关闭 `errors_as_nodata`，fill_value 改为 NaN

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

### Step 0.3: 固定 EPSG 为 AOI 中心点所在 UTM 带

```python
def _get_aoi_utm_epsg(aoi: gpd.GeoDataFrame) -> int:
    """返回 AOI 中心点所在 UTM 带的 EPSG 代码。"""
    centroid = aoi.to_crs("EPSG:4326").geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

# 在 download_source 中替换
# epsg = _extract_epsg(items[0])
epsg = _get_aoi_utm_epsg(aoi)
```

### Step 0.4: Landsat assets 增加 `qa_pixel`

```python
"landsat": [
    "blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"
]
```

### Step 0.5: 下载后覆盖率校验

```python
def _validate_coverage(ds, min_valid_ratio=0.05):
    var = list(ds.data_vars)[0]
    arr = ds[var]
    for t in range(arr.sizes["time"]):
        band0 = arr.isel(time=t, band=0)
        valid = float(np.isfinite(band0.values).mean())
        if valid < min_valid_ratio:
            raise RuntimeError(
                f"time {t} coverage {valid:.2%} < {min_valid_ratio:.2%}"
            )
```

### Step 0.6: 提交并测试

```bash
git add scripts/data/download_pc.py
git commit -m "fix: PC download S1/Landsat silent zero-fill, UTM zone, qa_pixel"
git push origin feat/data-and-model-rework

# 测试：下载 2025-01 的 Harbin S1 和 Landsat
python scripts/data/download_pc.py \
    --region harbin --source s1 --start 2025-01-01 --end 2025-01-31 \
    --region-file configs/regions/harbin.geojson --overwrite

python scripts/data/download_pc.py \
    --region harbin --source landsat --start 2025-01-01 --end 2025-01-31 \
    --region-file configs/regions/harbin.geojson --overwrite
```

Expected: 新文件每个 time slice 非 NaN/非零像素比例 > 5%。

---

## Gate 1: 重新下载海淀区 S2/S1/Landsat

### Step 1.1: 移动错误文件到 trash

```bash
mkdir -p /data/xuannv_embedding/raw/_trash_2026-06-19
mv /data/xuannv_embedding/raw/haidian/s2/s2_20250101_20251231.nc \
   /data/xuannv_embedding/raw/haidian/s2/s2_20260101_20260531.nc \
   /data/xuannv_embedding/raw/haidian/s1/s1_20250101_20251231.nc \
   /data/xuannv_embedding/raw/haidian/s1/s1_20260101_20260531.nc \
   /data/xuannv_embedding/raw/haidian/landsat/landsat_20250101_20251231.nc \
   /data/xuannv_embedding/raw/_trash_2026-06-19/
```

### Step 1.2: 创建并运行 `download_pc_haidian.sh`

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

# S1 / Landsat 按月下载，避免 token 过期
for y in 2025; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
    download s1 "$y-$m-01" "$y-$m-$last" "$LOG_DIR/download_pc_haidian_s1_${y}${m}.log" &
    download landsat "$y-$m-01" "$y-$m-$last" "$LOG_DIR/download_pc_haidian_landsat_${y}${m}.log" &
  done
done
for m in 01 02 03 04 05; do
  last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
  download s1 "2026-$m-01" "2026-$m-$last" "$LOG_DIR/download_pc_haidian_s1_2026${m}.log" &
  download landsat "2026-$m-01" "2026-$m-$last" "$LOG_DIR/download_pc_haidian_landsat_2026${m}.log" &
done

wait
echo "[$(date)] 海淀区 PC 下载完成"
```

Run:

```bash
chmod +x scripts/data/download_pc_haidian.sh
nohup scripts/data/download_pc_haidian.sh > /data/xuannv_embedding/logs/download_pc_haidian_all.log 2>&1 &
```

### Step 1.3: QA 脚本 `scripts/qa/qa_download.py`

```python
import json
import glob
from pathlib import Path
import numpy as np
import xarray as xr


def check_file(path: Path) -> dict:
    report = {"file": str(path), "ok": True, "issues": []}
    try:
        ds = xr.open_dataset(path, chunks={"time": 1})
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"cannot open: {e}")
        return report

    var = list(ds.data_vars)[0]
    arr = ds[var]
    report["dims"] = dict(arr.sizes)

    # 检查每个时间步有效像素比例
    band0 = arr.isel(band=0)
    valid_ratios = []
    for t in range(arr.sizes["time"]):
        b = band0.isel(time=t).values
        valid = np.isfinite(b).mean() if np.issubdtype(b.dtype, np.floating) else (b != 0).mean()
        valid_ratios.append(float(valid))

    report["valid_ratio_min"] = float(np.min(valid_ratios))
    report["valid_ratio_median"] = float(np.median(valid_ratios))
    report["valid_ratio_max"] = float(np.max(valid_ratios))

    if report["valid_ratio_median"] < 0.05:
        report["ok"] = False
        report["issues"].append(f"median valid ratio too low: {report['valid_ratio_median']:.2%}")

    # 检查是否所有 slice 全 0
    if report["valid_ratio_max"] == 0:
        report["ok"] = False
        report["issues"].append("all time slices are zero")

    return report


def main(region: str):
    out = {"region": region, "sources": {}}
    for source in ["s2", "s1", "landsat"]:
        files = sorted(glob.glob(f"/data/xuannv_embedding/raw/{region}/{source}/*.nc"))
        out["sources"][source] = [check_file(Path(f)) for f in files]
    out_path = Path(f"/data/xuannv_embedding/qa/qa_download_{region}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"QA report written to {out_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
```

Run after downloads complete:

```bash
python scripts/qa/qa_download.py --region haidian
```

### Step 1.4: 可视化抽检

```bash
python3 - <<'PY'
import xarray as xr, matplotlib.pyplot as plt, numpy as np, glob, os
from pathlib import Path
out_dir = Path('/data/xuannv_embedding/qa/figures/haidian')
out_dir.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob('/data/xuannv_embedding/raw/haidian/s2/*.nc'))[:2]
for f in files:
    ds = xr.open_dataset(f, chunks={'time':1})
    v = list(ds.data_vars)[0]
    arr = ds[v]
    t = arr.sizes['time']//2
    rgb = np.stack([arr.isel(time=t, band=i).values for i in [2,1,0]], axis=0)
    rgb = np.clip(rgb / np.percentile(rgb[np.isfinite(rgb)], 98), 0, 1)
    plt.figure(figsize=(6,6))
    plt.imshow(np.transpose(rgb, (1,2,0)))
    plt.title(f'{Path(f).name} t={t}')
    plt.axis('off')
    plt.savefig(out_dir / f'{Path(f).stem}_t{t}.png')
    plt.close()
PY
```

### Step 1.5: 子智能体审查

调用一个 `explore` 子智能体，prompt：

```
请审查海淀区 PC 数据下载后的 QA 报告 /data/xuannv_embedding/qa/qa_download_haidian.json 和可视化图 /data/xuannv_embedding/qa/figures/haidian/。

任务：
1. 检查 S2/S1/Landsat 各文件的 valid_ratio 中位数是否大于 5%。
2. 检查是否存在全 0 或全 NaN 的时间步。
3. 对比新旧文件（ trash 目录）的尺寸和时间维度，确认重新下载后是否改善。
4. 检查可视化 RGB 是否正常（无全黑/全白、几何形状合理）。
5. 若有问题，指出具体文件和可能原因。

输出：PASS / CONDITIONAL / FAIL，并列出必须修复的项。
```

### Step 1.6: Commit & push 脚本

```bash
git add scripts/data/download_pc_haidian.sh scripts/qa/qa_download.py
git commit -m "data: add Haidian PC re-download script and download QA"
git push origin feat/data-and-model-rework
```

---

## Gate 2: 重新下载哈尔滨 S1/Landsat 2025/2026

哈尔滨 S2 已经按月重新下载且正常，但 S1/Landsat 2025/2026 存在跨带和静默填零问题。

### Step 2.1: 移动问题文件

```bash
mkdir -p /data/xuannv_embedding/raw/_trash_2026-06-19_harbin
mv /data/xuannv_embedding/raw/harbin/s1/s1_20250101_20251231.nc \
   /data/xuannv_embedding/raw/harbin/s1/s1_20260101_20260531.nc \
   /data/xuannv_embedding/raw/harbin/landsat/landsat_20250101_20251231.nc \
   /data/xuannv_embedding/raw/_trash_2026-06-19_harbin/
# 保留 2026 Landsat，它看起来正常
```

### Step 2.2: 创建 `download_pc_harbin_s1_landsat.sh`

类似海淀，按月下载 S1 2025/2026 和 Landsat 2025。

### Step 2.3: 运行 QA

```bash
python scripts/qa/qa_download.py --region harbin
```

### Step 2.4: 子智能体审查 + commit

同 Gate 1。

---

## Gate 3: 重写并加速 `preprocess.py`

### Step 3.1: 设计新流程

核心改变：
1. 读取 AOI GeoJSON，生成基于 `patch_size * resolution` 的 patch 网格。
2. 对每个 `(nc_file, time_idx)`：
   - 一次性加载整时相到内存；
   - 按整数像素窗口批量切片；
   - 边缘 `np.pad` 补 `nodata`；
   - 跳过有效像素比例低于 `min_valid_ratio` 的 patch；
   - 写出为 COG GeoTIFF（deflate + tiled）。
3. 对每个 patch 写出 `_mask.tif`：有效像素为 1，nodata/cloud/shadow 为 0。
4. 多进程按 `(nc_file, time_idx)` 分发。

### Step 3.2: 实现核心函数

```python
# scripts/data/preprocess.py
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import xarray as xr


def generate_patch_grid(aoi_path: str, patch_size: int, resolution: float) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path).to_crs(epsg=None)  # will set later
    # implementation details in actual file
    ...


def process_one_time(args):
    nc_path, time_idx, config = args
    ds = xr.open_dataset(nc_path, chunks={"time": 1})
    data_var = list(ds.data_vars)[0]
    arr = ds[data_var].isel(time=time_idx).values.astype(np.float32)
    timestamp = ds["time"].values[time_idx]

    dst_dir = Path(f"/data/xuannv_embedding/processed/{config['region']}/patches/{config['source']}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    patch_gdf = config["patch_gdf"]
    nodata = config.get("nodata", 0.0)
    min_valid_ratio = config.get("min_valid_ratio", 0.1)
    transform = config["transform"]

    for _, row in patch_gdf.iterrows():
        bounds = row.geometry.bounds
        win = window_from_bounds(*bounds, transform=transform)
        col_off, row_off, w, h = map(int, [win.col_off, win.row_off, win.width, win.height])

        pad_left = max(0, -col_off)
        pad_top = max(0, -row_off)
        pad_right = max(0, col_off + w - arr.shape[-1])
        pad_bottom = max(0, row_off + h - arr.shape[-2])

        if pad_left or pad_top or pad_right or pad_bottom:
            slice_arr = arr[:,
                           max(0, row_off):min(row_off + h, arr.shape[-2]),
                           max(0, col_off):min(col_off + w, arr.shape[-1])]
            slice_arr = np.pad(
                slice_arr,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=nodata,
            )
        else:
            slice_arr = arr[:, row_off:row_off + h, col_off:col_off + w]

        valid = (slice_arr[0] != nodata) & np.isfinite(slice_arr[0])
        if valid.mean() < min_valid_ratio:
            continue

        ts_str = _timestamp_str(timestamp)
        out_path = dst_dir / f"{row['patch_id']}_{ts_str}.tif"
        mask_path = dst_dir / f"{row['patch_id']}_{ts_str}_mask.tif"

        profile = {
            "driver": "GTiff",
            "height": slice_arr.shape[-2],
            "width": slice_arr.shape[-1],
            "count": slice_arr.shape[-3],
            "dtype": slice_arr.dtype,
            "crs": config["crs"],
            "transform": from_bounds(*bounds, slice_arr.shape[-1], slice_arr.shape[-2]),
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(slice_arr)

        mask_profile = profile.copy()
        mask_profile.update(count=1, dtype="uint8", nodata=None)
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(valid.astype(np.uint8), 1)


def process_file(nc_path, config):
    ds = xr.open_dataset(nc_path, chunks={"time": 1})
    tasks = [(nc_path, t, config) for t in range(ds.sizes["time"])]
    ds.close()
    for task in tasks:
        process_one_time(task)


def main(config_path: str, source: str):
    with open(config_path) as f:
        config = json.load(f)
    config["source"] = source
    aoi = gpd.read_file(config["aoi"])
    config["crs"] = aoi.estimate_utm_crs() if aoi.crs is None else aoi.to_crs(aoi.estimate_utm_crs()).crs
    config["patch_gdf"] = generate_patch_grid(config["aoi"], config["patch_size"], config["resolution"])

    nc_files = sorted(Path("/data/xuannv_embedding/raw").glob(f"{config['region']}/{source}/*.nc"))

    with ProcessPoolExecutor(max_workers=config.get("workers", 8)) as executor:
        list(executor.map(lambda p: process_file(p, config), nc_files))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
```

### Step 3.3: 更新预处理配置

`scripts/data/preprocess_harbin.json`：

```json
{
  "region": "harbin",
  "patch_size": 128,
  "resolution": 10,
  "aoi": "configs/regions/harbin.geojson",
  "min_valid_ratio": 0.3,
  "workers": 8,
  "nodata": 0.0,
  "sources": {
    "s2": {"bands": 12},
    "s1": {"bands": 2},
    "landsat": {"bands": 6}
  }
}
```

### Step 3.4: Benchmark

```bash
# 旧流程（100 patches, 1 time）
time python scripts/data/preprocess_legacy.py --config scripts/data/preprocess_harbin.json --source s2 --max-patches 100 --max-times 1

# 新流程（100 patches, 1 time）
time python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source s2 --max-patches 100 --max-times 1
```

Expected: 新流程耗时 < 旧流程的 1/50。

### Step 3.5: QA 与 commit

```bash
python scripts/qa/qa_patch_extraction.py --region harbin --source s2
git add scripts/data/preprocess.py scripts/qa/qa_patch_extraction.py
git commit -m "refactor: per-time-slice slicing + multiprocessing + valid masks"
git push origin feat/data-and-model-rework
```

Subagent review prompt:

```
请审查 /data/xuannv_embedding/qa/qa_patch_extraction_harbin_s2.json 和 sample figures。
1. 检查 patch 数量是否与 AOI 面积匹配。
2. 检查全 0 patch 比例是否显著下降。
3. 检查 S2 patch 与对应 WorldCover 标签的几何对齐。
4. 检查有效像素比例阈值 0.3 是否合理。
输出：PASS / CONDITIONAL / FAIL。
```

---

## Gate 4: 生成 S2/S1/Landsat patches

### Step 4.1: 运行预处理

```bash
for region in harbin haidian; do
  for source in s2 s1 landsat; do
    python scripts/data/preprocess.py --config scripts/data/preprocess_${region}.json --source $source
  done
done
```

### Step 4.2: 运行 QA

```bash
for region in harbin haidian; do
  for source in s2 s1 landsat; do
    python scripts/qa/qa_patch_extraction.py --region $region --source $source
  done
done
```

### Step 4.3: 子智能体审查

调用一个子智能体综合审查所有 QA 报告，重点：
- S1/Landsat patch 数量与有效覆盖；
- 跨源对齐（S2/S1/Landsat 同一 patch_id 是否覆盖同一地理范围）；
- 是否存在大量空 patch。

---

## Gate 5: 高分辨率光学预处理

### 5.1 哈尔滨 DOM

DOM 坐标系为 CGCS2000 / 高斯-克吕格，需重投影到 EPSG:32652。

```bash
# 单文件重投影示例
input=raw/harbin/highres_optical/_mosaic/dom_202512.tif
output=raw/harbin/highres_optical/_mosaic/dom_202512_32652.tif
gdalwarp -s_srs EPSG:4490 -t_srs EPSG:32652 -tr 0.5 0.5 -r bilinear -tap \
  -co COMPRESS=DEFLATE -co TILED=YES \
  $input $output
```

### 5.2 海淀 PlanetScene

按月份镶嵌，再重投影/裁剪。

```bash
# 示例：2025-06
mkdir -p raw/haidian/highres_optical/_mosaic
gdalbuildvrt raw/haidian/highres_optical/_mosaic/202506.vrt \
  raw/haidian/highres_optical/*_202506*.tif
gdalwarp -t_srs EPSG:32650 -tr 3.0 3.0 -r bilinear -tap \
  -co COMPRESS=DEFLATE -co TILED=YES \
  raw/haidian/highres_optical/_mosaic/202506.vrt \
  raw/haidian/highres_optical/_mosaic/202506.tif
```

### 5.3 切 patch

`scripts/data/preprocess_highres_optical.py`：

```python
# 对每个 1280m × 1280m 的低分辨率 patch 边界，从高分辨率 mosaic 中整数窗口裁剪
# 输出尺寸随分辨率变化：0.5m → 2560×2560，3m → ~427×427
# 同时生成 cloud/shadow mask（Planet UDM2 或 DOM 阴影 mask）
```

### 5.4 QA + subagent review

```bash
python scripts/qa/qa_highres_optical.py --region harbin
python scripts/qa/qa_highres_optical.py --region haidian
```

Subagent review prompt:

```
请审查高分辨率光学 patch：
1. 检查 DOM / PlanetScene patch 是否与对应 S2 patch 地理范围一致。
2. 检查是否存在大量全黑/全白或严重云影 patch。
3. 检查不同分辨率（0.5m vs 3m）的 patch 尺寸差异是否在预期范围。
4. 检查辐射归一化后数值范围是否合理。
输出：PASS / CONDITIONAL / FAIL。
```

---

## Gate 6: 高分辨率 SAR 预处理

### 6.1 解压 ORG 产品

只使用 `BC*-SM-ORG-*-2SVV-*.zip`（已地理编码）。忽略 SLC。

```bash
python scripts/data/unpack_haidian_sar.py \
  --input-root raw/haidian/highres_sar \
  --output-root raw/haidian/highres_sar/_unpacked
```

### 6.2 重投影到 EPSG:32650 并对齐 10m 参考网格

```bash
gdalwarp -t_srs EPSG:32650 -tr 3.0 3.0 -r bilinear -tap \
  -dstnodata 0 -co COMPRESS=DEFLATE -co TILED=YES \
  raw/haidian/highres_sar/_unpacked/bc5-sm-org-vv-20251107t142932.tif \
  raw/haidian/highres_sar/_reprojected/20251107_vv.tif
```

### 6.3 切 patch 并归一化

`scripts/data/preprocess_highres_sar.py`：

```python
import numpy as np
import rasterio

def sar_to_db(linear):
    return 10 * np.log10(linear + 1e-10)

def normalize_sar(db, pol="VV"):
    low = -25 if pol in ["VV", "HH"] else -35
    high = 0
    return np.clip((db - low) / (high - low), 0, 1)

# 对每景 ORG 产品：
# 1. 读取 VV 波段（如有 VH 也读取）
# 2. 按 1280m × 1280m 地理范围裁剪
# 3. dB 转换 + clip + 归一化到 [0,1]
# 4. 输出 float32 GeoTIFF
```

### 6.4 QA + subagent review

```bash
python scripts/qa/qa_highres_sar.py --region haidian
```

Subagent review prompt:

```
请审查高分辨率 SAR patch：
1. 检查 ORG 产品是否正确解压，是否混用了未地理编码的 SLC。
2. 检查重投影后像素是否对齐到 10m 参考网格。
3. 检查归一化后数值范围是否集中在 [0,1]，是否有异常离群值。
4. 检查叠掩/阴影区域是否被标记为无效。
5. 抽查 SAR patch 与对应 S2 光学 patch 的城市结构是否一致。
输出：PASS / CONDITIONAL / FAIL。
```

---

## Gate 7: WorldCover 标签重处理

### 7.1 更新 `preprocess_lulc.py`

```python
# ESRI 2023 原始值 → 连续训练索引（0 为 ignore_index，10 Clouds 也 ignore）
ESRI2023_REMAP = {
    0: 0,   # No Data / ignore
    1: 1,   # Water
    2: 2,   # Trees
    4: 3,   # Flooded Vegetation
    5: 4,   # Crops
    7: 5,   # Built Area
    8: 6,   # Bare Ground
    9: 7,   # Snow/Ice
    10: 0,  # Clouds → ignore
    11: 8,  # Rangeland
}


def remap_esri2023(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for old, new in ESRI2023_REMAP.items():
        out[arr == old] = new
    return out
```

### 7.2 重新切标签 patch

```bash
python scripts/data/preprocess_lulc.py --config scripts/data/preprocess_lulc_harbin.json
python scripts/data/preprocess_lulc.py --config scripts/data/preprocess_lulc_haidian.json
```

### 7.3 QA + subagent review

```bash
python scripts/qa/qa_labels.py --region harbin
python scripts/qa/qa_labels.py --region haidian
```

Subagent review prompt:

```
请审查 WorldCover 标签：
1. 检查重映射后类别值是否只在 0–8 之间。
2. 检查标签 patch 与 S2 patch 是否严格对齐。
3. 检查各类别分布是否严重不平衡，是否需要调整采样策略。
4. 抽查典型地物（建筑、水体、农田）标签是否合理。
输出：PASS / CONDITIONAL / FAIL。
```

---

## Gate 8: 升级 Manifest

### 8.1 多源逐月格式

```json
{
  "patch_id": "harbin_000001",
  "region": "harbin",
  "crs": "EPSG:32652",
  "bounds": [xmin, ymin, xmax, ymax],
  "sources": {
    "s2": {"months": [202501, 202502, ...], "count": 17},
    "s1": {"months": [...], "count": 15},
    "landsat": {"months": [...], "count": 8},
    "highres_optical": {"months": [202512, 202605], "count": 2},
    "highres_sar": {"months": [202501, ...], "count": 5}
  },
  "label": "processed/harbin/labels/worldcover/harbin_000001.tif"
}
```

### 8.2 生成

```bash
python scripts/data/generate_manifest.py --regions harbin,haidian
```

### 8.3 QA + subagent review

```bash
python scripts/qa/qa_manifest.py --regions harbin,haidian
```

Subagent review prompt:

```
请审查 manifest：
1. 检查每个 patch_id 是否都有 S2 标签。
2. 检查 S1/Landsat/高分辨率源是否存在合理的缺失月份。
3. 检查时间戳是否按月份升序排列。
4. 检查是否存在无 S2 但有其他源的异常 patch。
输出：PASS / CONDITIONAL / FAIL。
```

---

## Gate 9: 统计量

### 9.1 计算每源统计量

```bash
python scripts/data/compute_statistics.py --regions harbin,haidian
```

输出 `statistics/<region>/<source>_stats.json`：

```json
{
  "s2": {"mean": [...], "std": [...], "min": ..., "max": ...},
  "s1": {...},
  "landsat": {...},
  "highres_optical": {...},
  "highres_sar": {...}
}
```

### 9.2 QA + subagent review

```bash
python scripts/qa/qa_statistics.py --regions harbin,haidian
```

---

## Gate 10: 端到端冒烟测试

### 10.1 测试数据集 → collate → model forward

```bash
pytest tests/test_dataset.py tests/test_model.py tests/test_batch_preparation.py -v
```

### 10.2 子智能体审查

调用一个子智能体审查所有 QA 报告和测试结果，确认数据层无阻塞问题后再进入模型改造。

---

## 时序采样实现设计（供阶段 B 参考）

虽然本计划只到数据层，但时序采样策略会影响 manifest 中 `months` 的格式。最终数据集应能支持：

```python
class VariableWindowTemporalSampler:
    def __init__(self, max_months=17, lengths=(2,3,6,12), probs=None, dropout=0.0):
        self.max_months = max_months
        self.lengths = lengths
        self.probs = probs or [0.15, 0.15, 0.2, 0.3, 0.2]
        self.dropout = dropout

    def sample(self, available_months):
        available = sorted(available_months)
        L = random.choices(self.lengths, weights=self.probs)[0]
        L = min(L, len(available))
        if L == len(available):
            start = 0
        else:
            start = random.randint(0, len(available) - L)
        window = available[start:start + L]

        # 在固定 17 个月网格上构造 mask
        full_grid = list(range(available[0], available[-1] + 1))  # 简化
        # 实际应使用统一月度网格
        mask = [1 if m in window else 0 for m in full_grid]

        # temporal dropout
        if self.dropout > 0 and sum(mask) > 1:
            for i in range(len(mask)):
                if mask[i] and random.random() < self.dropout:
                    mask[i] = 0
        return full_grid, mask
```

注意：位置编码必须基于实际月份（timestamp-based sinusoidal），而不是绝对索引。

---

## 执行顺序与里程碑

| 里程碑 | 内容 | 预计时间 |
|---|---|---|
| M0 | Gate 0 完成：download_pc.py 修复 | 0.5 天 |
| M1 | Gate 1–2 完成：Haidian + Harbin 数据重新下载并通过 QA | 2–3 天 |
| M2 | Gate 3–4 完成：preprocess.py 重写 + 所有 S2/S1/Landsat patch 生成 | 2 天 |
| M3 | Gate 5 完成：高分辨率光学 patch | 1 天 |
| M4 | Gate 6 完成：高分辨率 SAR patch | 1–2 天 |
| M5 | Gate 7 完成：WorldCover 重处理 | 0.5 天 |
| M6 | Gate 8–9 完成：manifest + 统计量 | 0.5 天 |
| M7 | Gate 10 完成：端到端冒烟测试通过 | 0.5 天 |

**总计：约 8–10 天**（受下载速度和重新处理规模影响）。

---

## 执行方式

**Subagent-Driven**：每个 Gate 作为一个独立 Task，分配给专门的子智能体。
- Gate 0, 3, 5, 6, 7 可由 coder 子智能体实现代码；
- Gate 1, 2, 4 主要由 coder 子智能体调用脚本并监控；
- 每个 Gate 完成后，由 explore 子智能体进行 QA 审查；
- 我作为 root agent 负责协调、review 关键提交、向你汇报。

**下一步**：如果你批准此计划，我将：
1. 先派一个 coder 子智能体执行 Gate 0（修复 download_pc.py）；
2. Gate 0 完成后立即运行 QA 并由子智能体审查；
3. 审查通过后再进入 Gate 1（Haidian 重新下载）。

请确认是否开始执行。
