# Phase 9：数据目录清理、完整区域范围确认、全量数据重新下载

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 响应 chat2 中的三项核心要求：(1) 调研地理嵌入行业最佳实践并与当前方案对比；(2) 审计当前代码库关键节点，识别幻觉/风险；(3) 获取哈尔滨 424 patch / 海淀 320 patch 的完整区域范围，清理数据目录，重新下载全量数据并规范存储。

**Architecture:** 保持现有 `xuannv_embedding` 包结构不变；关键改动集中在数据目录结构、区域 polygon、PC 下载策略（全区域、全时段）以及数据 README 文档。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, Planetary Computer, rasterio, xarray, geopandas/shapely, BaiduPCS-Go, ModelScope SDK.

---

## 行业最佳实践调研结论

### 1. 地理嵌入的主流路线

| 模型/方法 | 核心思想 | 与当前项目的差异 |
|-----------|----------|------------------|
| **AEF (AlphaEarth Foundations)** | 多源时序数据 → per-sensor encoder → 时间/空间自注意力 → vMF 瓶颈 → 单位球面 embedding；支持连续时间、稀疏标签、10 m 分辨率 | 当前实现基本复现该路线，但**时间编码目前使用月份索引而非连续时间戳**，且**空间注意力在 128×128 输入上需要 `spatial_stride=2` 才能训练**，与论文中可能使用的 patch embedding / 下采样 stem 类似 |
| **SatMAE / Scale-MAE** | ViT-based MAE 预训练，Scale-MAE 强调多尺度 | 主要用于单源光学影像预训练，**不直接支持多源融合与月度 embedding** |
| **Clay / Prithvi** | 大规模自监督基础模型，通用 EO 表征 | 模型规模大、训练数据量巨大，**不针对月度变化检测设计**；当前项目更小、更轻量 |
| **SatCLIP** | 图像-地理坐标对比学习 | 生成的是**位置嵌入**，而非像当前项目这样从观测数据学习地理特征嵌入 |
| **DOFA** | 波长条件化通道权重，适配多光谱 | 当前项目尚未显式建模波段波长，仅用普通卷积投影 |

### 2. 关键差异总结

- **时间建模**：AEF 论文使用连续时间（continuous time），当前实现使用 2025-01 到 2026-05 的离散月份。这在月度变化检测场景下可接受，但损失了亚月度时间精度。
- **空间分辨率**：AEF 论文声称 10 m 分辨率，当前项目 1280 m × 1280 m patch + 10 m 重采样 → 128×128 像素，与论文一致。但 SpaceOperator 全图注意力需要 `spatial_stride=2` 才能稳定训练。
- **多源融合**：当前 `AvailabilityAwareFusion` 与 mask 处理已覆盖 S1/S2/Landsat/高分辨率数据的缺失场景，与 AEF 的 availability-aware 思路一致。
- **下游评估**：行业通用做法是用 kNN/linear probe 做分类、变化检测用 embedding 距离或差分 + 阈值，当前脚本已覆盖。

---

## 代码库关键节点审计

### 已识别的风险与“幻觉”

| 节点 | 风险等级 | 说明 |
|------|----------|------|
| **区域 polygon** | 🔴 高 | 当前 `configs/regions/harbin.geojson` 与 `haidian.geojson` 在 UTM 投影下仅约 8 km × 11 km，按 1280 m patch 只能切出 **63 个 patch**，与目标 424 / 320 严重不符。需要完整区域 polygon。 |
| **PC 下载策略** | 🟠 中 | 当前只下载了 2025-01 数据；全时段（2025-01 至 2026-05）需要逐月或更细粒度搜索，否则区域覆盖不全，patch 数量会远少于目标。 |
| **世界覆盖 target head** | 🟡 低 | `worldcover` categorical head 在配置中存在，但数据管道中没有对应 source，训练时 mask 为 0，不贡献梯度，属于占位配置。 |
| **时间戳统一** | 🟠 中 | `prepare_batch` 直接取第一个有效 source 的时间戳作为全局时间戳。若不同 sensor 的观测月份不一致，会引入偏差。 |
| **高分辨率 SAR 通道数** | 🟠 中 | `highres_sar` 尚未注册到 `base.yaml` 的 `sensor_channels`，若后续加入该 source 会触发 KeyError。 |
| **数据目录** | 🟠 中 | `/data/xuannv_embedding/` 下已累积 smoke/测试/旧目录，缺少顶层 README 说明各子目录用途。 |

### 未发现致命幻觉

- 模型结构（encoder → time/space operator → vMF → decoder）与 AEF 论文描述一致。
- checkpoint、trainer、loss、data collate 等实现逻辑自洽。
- 128×128 smoke 训练已跑通，证明训练链路可用。

---

## 文件结构

| 路径 | 职责 |
|------|------|
| `configs/regions/harbin.geojson` | 哈尔滨新区完整 AOI（待更新） |
| `configs/regions/haidian.geojson` | 海淀区完整 AOI（待更新） |
| `/data/xuannv_embedding/README.md` | 数据目录说明文档（待创建） |
| `scripts/data/download_pc.py` | Planetary Computer 下载入口 |
| `scripts/data/preprocess_harbin.json` | 哈尔滨预处理配置 |
| `scripts/data/preprocess_haidian.json` | 海淀区预处理配置 |
| `scripts/data/preprocess_harbin_highres.json` | 哈尔滨高分辨率预处理配置 |
| `scripts/data/preprocess_haidian_highres.json` | 海淀区高分辨率预处理配置 |
| `scripts/data/download_baidu.py` | 百度网盘高分辨率数据下载 |
| `scripts/data/download_modelscope.py` | ModelScope 天仪 SAR 下载 |

---

### Task 1: 获取并验证完整区域 polygon

**Files:**
- Create/Modify: `configs/regions/harbin.geojson`
- Create/Modify: `configs/regions/haidian.geojson`

**Context:** 当前 polygon 只能生成 63 个 patch，与目标 424 / 320 不符。需要完整 AOI。

- [ ] **Step 1: 尝试从上一个 GitHub 仓库拉取完整 polygon**

```bash
cd /tmp
rm -rf xuannv_v12
git clone --depth 1 --branch v12-clean-dynamic https://github.com/go-bananas-wwj/xuannv.git xuannv_v12
find xuannv_v12/configs -name '*.geojson' -o -name '*region*'
```

Expected: 若能连上 GitHub，复制 `configs/regions/harbin.geojson` 与 `haidian.geojson` 覆盖当前文件。

- [ ] **Step 2: 若 Step 1 失败，从百度网盘哈尔滨高分辨率 tiff 推断完整边界**

下载哈尔滨高分辨率 tiff 后读取 bounds：

```bash
cd /root/workspace/xuannv
python - <<'PY'
import rasterio
from pathlib import Path
from shapely.geometry import box, mapping
import json

src = Path('/data/xuannv_embedding/raw/harbin/highres_optical/哈尔滨_202512/哈尔滨新区_BJ3_202512_DOM_RGB.tif')
with rasterio.open(src) as ds:
    bounds = ds.bounds
    crs = ds.crs.to_string()

geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
geojson = {
    'type': 'FeatureCollection',
    'crs': {'type': 'name', 'properties': {'name': crs}},
    'features': [{'type': 'Feature', 'properties': {}, 'geometry': mapping(geom)}]
}
Path('configs/regions/harbin.geojson').write_text(json.dumps(geojson, indent=2))
print('harbin bounds', bounds, 'crs', crs)
PY
```

Expected: `configs/regions/harbin.geojson` 更新为覆盖完整哈尔滨高分辨率数据的 polygon。

- [ ] **Step 3: 验证 patch 数量**

```bash
cd /root/workspace/xuannv
python - <<'PY'
import json
from pathlib import Path
import numpy as np

try:
    from pyproj import Transformer
except Exception as e:
    print('pyproj missing', e); raise

def estimate(path, patch_m=1280):
    data = json.loads(Path(path).read_text())
    coords = np.array(data['features'][0]['geometry']['coordinates'][0])
    lons, lats = coords[:,0], coords[:,1]
    lon_c = (lons.min()+lons.max())/2
    lat_c = (lats.min()+lats.max())/2
    zone = int((lon_c + 180)//6)+1
    epsg = 32600+zone if lat_c>=0 else 32700+zone
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    xs, ys = t.transform(lons, lats)
    nx = int(np.ceil((xs.max()-xs.min())/patch_m))
    ny = int(np.ceil((ys.max()-ys.min())/patch_m))
    return epsg, nx, ny, nx*ny

for r in ['harbin','haidian']:
    epsg, nx, ny, total = estimate(f'configs/regions/{r}.geojson')
    print(r, 'epsg', epsg, 'grid', nx, 'x', ny, 'total', total)
PY
```

Expected: 哈尔滨 patch 总数接近 424，海淀接近 320。若差距仍大，需用户确认 polygon。

- [ ] **Step 4: 提交 polygon**

```bash
cd /root/workspace/xuannv
git add configs/regions/
git commit -m "feat(config): update full region polygons for harbin and haidian"
git push origin main
```

---

### Task 2: 清理数据目录并创建 README

**Files:**
- Create: `/data/xuannv_embedding/README.md`
- Modify: `.gitignore`（可选）

**Context:** 当前 `/data/xuannv_embedding/` 下存在 smoke、旧统计量、测试产物，需要规范化。

- [ ] **Step 1: 删除不再需要的中间/测试目录**

保留 `raw/`、`processed/`、`statistics/`、`outputs/` 四个顶层目录；删除或归档以下内容：

```bash
# 示例：仅保留正式数据，删除 smoke 测试数据（若用户确认）
rm -rf /data/xuannv_embedding/processed/harbin_smoke
rm -rf /data/xuannv_embedding/statistics/harbin_smoke
rm -rf /data/xuannv_embedding/outputs/smoke_test
```

注意：删除前需再次确认用户是否保留 smoke 产物。

- [ ] **Step 2: 创建数据目录 README**

创建 `/data/xuannv_embedding/README.md`：

```markdown
# xuannv_embedding 数据目录说明

本目录与代码仓库 `/root/workspace/xuannv` 分离，避免大文件进入 git。

## 目录结构

- `raw/`: 原始下载数据（NetCDF、zip、tiff 等）。
  - `harbin/` / `haidian/`: PC 下载的 S1/S2/Landsat 与百度网盘/ModelScope 高分辨率数据。
- `processed/`: 对齐、切 patch 后的训练数据。
  - `{region}/scenes/{source}/`: 按 source 存放的 GeoTIFF patches。
  - `{region}/scenes/manifest.json`: 训练 manifest。
- `statistics/`: 各 source 的 mean/std 统计量（`_stats.json`）。
- `outputs/`: 训练 checkpoint、embedding、评估结果。

## 当前状态

- 哈尔滨：...
- 海淀区：...
```

- [ ] **Step 3: 提交 README（注意 README 位于 /data，不进入 git）**

`/data/xuannv_embedding/README.md` 不受 git 管理，无需提交；但可在代码仓库中创建 `docs/data_layout.md` 作为备份说明。

```bash
cd /root/workspace/xuannv
cp /data/xuannv_embedding/README.md docs/data_layout.md
git add docs/data_layout.md
git commit -m "docs: add data directory layout documentation"
git push origin main
```

---

### Task 3: 重新下载哈尔滨与海淀区 S1/S2/Landsat 全量数据

**Files:**
- Create/Modify: `scripts/data/download_pc_harbin.sh`
- Create/Modify: `scripts/data/download_pc_haidian.sh`
- Modify: `docs/data_status/2026-06-18_harbin_sample.md`
- Create: `docs/data_status/2026-06-18_haidian_sample.md`

**Context:** 需要按完整 polygon 和 2025-01 至 2026-05 全时段下载。

- [ ] **Step 1: 哈尔滨 S2 全时段下载脚本**

创建 `scripts/data/download_pc_harbin.sh`：

```bash
#!/usr/bin/env bash
set -e
REGION=harbin
REGION_FILE=/root/workspace/xuannv/configs/regions/harbin.geojson
OUTPUT=/data/xuannv_embedding/raw/harbin

for SRC in s2 s1 landsat; do
    python /root/workspace/xuannv/scripts/data/download_pc.py \
        --region ${REGION} \
        --source ${SRC} \
        --start 2025-01-01 \
        --end 2026-05-31 \
        --region-file ${REGION_FILE} \
        --output-root /data/xuannv_embedding/raw
done
```

- [ ] **Step 2: 海淀区 S2 全时段下载脚本**

创建 `scripts/data/download_pc_haidian.sh`：

```bash
#!/usr/bin/env bash
set -e
REGION=haidian
REGION_FILE=/root/workspace/xuannv/configs/regions/haidian.geojson
OUTPUT=/data/xuannv_embedding/raw/haidian

for SRC in s2 s1 landsat; do
    python /root/workspace/xuannv/scripts/data/download_pc.py \
        --region ${REGION} \
        --source ${SRC} \
        --start 2025-01-01 \
        --end 2026-05-31 \
        --region-file ${REGION_FILE} \
        --output-root /data/xuannv_embedding/raw
done
```

- [ ] **Step 3: 执行下载（耗时较长，建议后台或分 source 执行）**

```bash
cd /root/workspace/xuannv
chmod +x scripts/data/download_pc_harbin.sh scripts/data/download_pc_haidian.sh
# 可按需分 source 执行，避免一次性占用大量磁盘/网络
scripts/data/download_pc_harbin.sh
```

Expected: `raw/harbin/` 与 `raw/haidian/` 下生成 `s2/`、`s1/`、`landsat/` 子目录，内含覆盖完整 AOI 的 NetCDF 文件。

- [ ] **Step 4: 提交脚本与状态文档**

```bash
cd /root/workspace/xuannv
git add scripts/data/download_pc_harbin.sh scripts/data/download_pc_haidian.sh docs/data_status/
git commit -m "feat(data): add full-region PC download scripts for harbin and haidian"
git push origin main
```

---

### Task 4: 下载百度网盘高分辨率数据

**Files:**
- Modify: `scripts/data/baidu_links_harbin.txt`
- Modify: `scripts/data/baidu_links_haidian.txt`
- Modify: `docs/data_status/2026-06-18_highres_sample.md`

**Context:** 百度网盘目录已确认：`/玄女数据下载/` 下包含 `beijin_rs_psscene_analytic_sr_udm2.zip`、`haidianlabel.rar`、`哈尔滨影像数据/`。

- [ ] **Step 1: 更新 links 文件**

```bash
cd /root/workspace/xuannv
cat > scripts/data/baidu_links_harbin.txt <<'EOF'
/玄女数据下载/哈尔滨影像数据/哈尔滨_202512/哈尔滨新区_BJ3_202512_DOM_RGB.tif
/玄女数据下载/哈尔滨影像数据/哈尔滨_202605/哈尔滨江北区域.tif
EOF

cat > scripts/data/baidu_links_beijing.txt <<'EOF'
/玄女数据下载/beijin_rs_psscene_analytic_sr_udm2.zip
EOF

cat > scripts/data/baidu_links_haidian.txt <<'EOF'
/玄女数据下载/haidianlabel.rar
EOF
```

- [ ] **Step 2: 下载哈尔滨高分辨率光学**

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_harbin.txt \
    --output /data/xuannv_embedding/raw/harbin/highres_optical
```

Expected: 下载两个 22 GB tiff，分别位于 `highres_optical/哈尔滨_202512/` 与 `highres_optical/哈尔滨_202605/`。

- [ ] **Step 3: 下载北京/海淀高分辨率数据**

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_beijing.txt \
    --output /data/xuannv_embedding/raw/beijing/highres_optical

python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_haidian.txt \
    --output /data/xuannv_embedding/raw/haidian/labels
```

Expected: 北京 zip 下载到 `raw/beijing/highres_optical/`，海淀标签 rar 下载到 `raw/haidian/labels/`。

- [ ] **Step 4: 提交 links 与状态更新**

```bash
cd /root/workspace/xuannv
git add scripts/data/baidu_links_*.txt docs/data_status/
git commit -m "docs(data): update baidu links and download status"
git push origin main
```

---

### Task 5: 继续并完成 ModelScope 海淀 SAR 数据下载

**Files:**
- Modify: `docs/data_status/2026-06-18_highres_sample.md`

- [ ] **Step 1: 续传下载**

```bash
cd /root/workspace/xuannv
export MODELSCOPE_TOKEN="ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
python scripts/data/download_modelscope.py \
    --dataset WeijieWu/haidian_sar_2025 \
    --output /data/xuannv_embedding/raw/haidian/highres_sar \
    --cache-dir /tmp/modelscope_cache
```

Expected: 缺失文件继续下载，已存在文件跳过。

- [ ] **Step 2: 记录下载大小**

```bash
du -sh /data/xuannv_embedding/raw/haidian/highres_sar
find /data/xuannv_embedding/raw/haidian/highres_sar -name '*.zip' | wc -l
```

---

### Task 6: 预处理全量数据

**Files:**
- Create/Modify: `scripts/data/preprocess_harbin.json`
- Create/Modify: `scripts/data/preprocess_haidian.json`
- Create: `scripts/data/preprocess_harbin_highres.json`
- Create: `scripts/data/preprocess_haidian_highres.json`

- [ ] **Step 1: 哈尔滨主数据预处理**

`scripts/data/preprocess_harbin.json`：

```json
{
  "_comment": "harbin full region 1280m patches at 10m/px -> 128x128",
  "region": "harbin",
  "raw_root": "/data/xuannv_embedding/raw/harbin",
  "output_root": "/data/xuannv_embedding/processed/harbin/scenes",
  "crs": "EPSG:32652",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["s2", "s1", "landsat"],
  "fill_missing": "zero"
}
```

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --overwrite
```

- [ ] **Step 2: 海淀区主数据预处理**

`scripts/data/preprocess_haidian.json`：

```json
{
  "_comment": "haidian full region 1280m patches at 10m/px -> 128x128",
  "region": "haidian",
  "raw_root": "/data/xuannv_embedding/raw/haidian",
  "output_root": "/data/xuannv_embedding/processed/haidian/scenes",
  "crs": "EPSG:32650",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["s2", "s1", "landsat"],
  "fill_missing": "zero"
}
```

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --overwrite
```

- [ ] **Step 3: 高分辨率数据预处理**

为哈尔滨与北京/海淀分别创建 `preprocess_*_highres.json`（source 为 `highres`，crs 与主数据一致），并运行。

- [ ] **Step 4: 生成 manifest 与统计量**

```bash
cd /root/workspace/xuannv
for region in harbin haidian; do
    python scripts/data/generate_manifest.py \
        --processed-dir /data/xuannv_embedding/processed/${region}/scenes \
        --output /data/xuannv_embedding/processed/${region}/scenes/manifest.json \
        --sources s2 s1 landsat highres

    python scripts/data/compute_statistics.py \
        --processed-dir /data/xuannv_embedding/processed/${region}/scenes \
        --output-dir /data/xuannv_embedding/statistics/${region} \
        --sources s2 s1 landsat highres
done
```

Expected: 哈尔滨 manifest 记录数 ≈ 424，海淀 ≈ 320。

- [ ] **Step 5: 提交预处理配置**

```bash
cd /root/workspace/xuannv
git add scripts/data/preprocess_*.json
git commit -m "feat(data): add full-region preprocessing configs"
git push origin main
```

---

### Task 7: 代码库修复

**Files:**
- Modify: `configs/base.yaml`
- Modify: `src/xuannv_embedding/training/batch_preparation.py`

- [ ] **Step 1: 注册 `highres_sar` source**

在 `configs/base.yaml` 中：

```yaml
model:
  sensor_channels:
    s2: 12
    s1: 2
    landsat: 6
    highres: 3
    highres_sar: 1
```

- [ ] **Step 2: 修复 `prepare_batch` 多高分辨率 source 支持**

参考 Task 7 of previous plan：将 `highres*` source 统一抽出，保留 `highres_frame`/`highres_mask` 兼容，并新增 `highres_frames` dict。

- [ ] **Step 3: 运行测试**

```bash
cd /root/workspace/xuannv
pytest tests/ -q
ruff check src tests scripts/
black --check src tests scripts/
```

Expected: 全部通过。

- [ ] **Step 4: 提交**

```bash
cd /root/workspace/xuannv
git add configs/base.yaml src/xuannv_embedding/training/batch_preparation.py
git commit -m "fix(config,train): register highres_sar and support multiple highres sources"
git push origin main
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-data-cleanup-full-region-and-research.md`.

**关键阻塞点：** 当前 `configs/regions/harbin.geojson` 与 `haidian.geojson` 面积过小，无法生成 424 / 320 个 patch。需要获取完整 AOI（优先从上一个 GitHub 仓库拉取；若网络不可达，可从哈尔滨高分辨率 tiff 推断边界，海淀需用户提供）。

确认后开始执行。
