# 修订版 Phase 9/10：数据清理、完整 AOI、全量数据下载与模型设计修正

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据 chat2 反馈：
1. 调研地理嵌入最佳实践并与当前方案对比；
2. 审计代码库幻觉/风险；
3. 获取哈尔滨 424 patch / 海淀 320 patch 的完整 AOI；
4. 清理 `/data/xuannv_embedding/` 并写 README；
5. 重新下载 2025-01 至 2026-05 全时段 S1/S2/Landsat 与高分辨率数据；
6. 修正模型结构，使其在 128×128 输入上输出 128×128 的 64-dim embedding（不再使用简单的 `spatial_stride` 下采样）。

**Architecture:** 数据侧保持现有脚本；模型侧参考 Brayden-Zhang 的 AEF 非官方复现，引入 **Space-Time-Precision 多分辨率通路**（space 1/16、time 1/8、precision 1/2），并通过一个上采样 head 将 64×64 bottleneck 输出恢复到 128×128 embedding。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, Planetary Computer, rasterio, xarray, BaiduPCS-Go, ModelScope SDK, geopandas/shapely.

---

## 一、行业最佳实践调研结论

### 1.1 AEF 复现代码的架构细节

找到了 GitHub 非官方复现 `Brayden-Zhang/alphaearth-foundations`（已通过 FetchURL 读取源码）。其关键设计：

| 通路 | 分辨率 | 操作 |
|------|--------|------|
| **Space** | 1/16 L | ViT-like 空间自注意力 |
| **Time** | 1/8 L | 时间轴自注意力 + 时间编码 |
| **Precision** | 1/2 L | 3×3 卷积保持局部精度 |

- 输入 `L×L`（例如 256×256），最终特征在 `L/2 × L/2`（128×128）。
- Decoder 通过 `F.interpolate` 把 `L/2` 重建上采样回 `L`。
- 与我们当前代码的差异：
  - 当前只有单一分辨率路径 + `spatial_stride` 下采样；
  - 复现代码显式维护三条不同分辨率路径，并通过 `LearnedSpatialResampling` 跨尺度交换信息。

### 1.2 我们的应对方案

为了在 **128×128 输入** 上得到 **128×128 的 64-dim embedding**，需要：

- 引入类似的三通路结构（或多分辨率模块），使 attention 发生在低分辨率（1/16、1/8），降低显存。
- 在 bottleneck 之后增加一个轻量 **embedding upsample head**（如 2× 反卷积或 bilinear + conv），将 `L/2 × L/2` 的 64-dim embedding 恢复到 `L × L`。
- 这样空间注意力仍然高效，但输出保持用户要求的 128×128。

> 注意：这是一个独立的模型重构 Task，放在数据准备完成后执行。

### 1.3 与其他方法对比

| 模型 | 特点 | 与当前项目差异 |
|------|------|----------------|
| AEF (Google) | 多源时序 + vMF + 多分辨率 STP | 当前复现缺少显式三通路，需补齐 |
| SatMAE/Scale-MAE | 单源光学 MAE 预训练 | 不支持多源月度融合 |
| Clay/Prithvi | 大规模通用 EO 基础模型 | 不针对月度变化检测设计 |
| SatCLIP | 图像-坐标对比学习 | 生成位置嵌入，而非观测特征嵌入 |
| ESRI 10m LULC | 2017-2023 年度 10m 分类 | 可替代不存在的 "WorldCover 2023" |

---

## 二、代码库关键节点审计（幻觉/风险）

| 节点 | 风险 | 说明 |
|------|------|------|
| **区域 polygon 过小** | 🔴 高 | 当前 harbin/haidian geojson 仅 ~8 km × 11 km，只能切 63 patch，与目标 424/320 不符 |
| **`spatial_stride` 导致 embedding 非 128×128** | 🔴 高 | 当前 `spatial_stride=2` 让 embedding_map 变成 64×64，违背用户要求 |
| **PC 数据仅 2025-01** | 🟠 中 | 需要 2025-01 至 2026-05 全时段 |
| **`worldcover` head 无 source** | 🟡 低 | 占位配置，训练时不贡献梯度 |
| **`highres_sar` 未注册** | 🟠 中 | 加入后会 KeyError |
| **时间戳取首个 source** | 🟠 中 | 多源时间不一致时可能偏差 |
| **数据目录缺少 README** | 🟠 中 | 需要规范化和清理 |

**未致命幻觉：** 整体训练链路、checkpoint、loss、data collate 自洽；128×128 smoke 训练已跑通。

---

## 三、数据目录现状

`/data/xuannv_embedding/` 当前内容：

```
raw/
  harbin/          # S1/S2/Landsat NetCDF（仅 2025-01）
  haidian/         # 主要为 highres_sar partial（33 GB）
  beijing/         # highres_optical 空
processed/
  harbin/scenes/        # 256×256 patches（20 个）
  harbin_128/scenes/    # 128×128 patches（63 个，S1/S2/Landsat）
  harbin_smoke/         # 64×64 smoke patches
statistics/
  harbin/               # 256×256 统计量
  harbin_128/           # 128×128 统计量
  harbin_smoke/         # smoke 统计量
outputs/
  smoke_test/           # smoke checkpoint
  harbin_128_v1/        # 128×128 训练 checkpoint（10 epoch）
```

需要清理 smoke/旧目录，建立统一结构。

---

## 四、修订后 Task 列表

### Task 1: 获取完整 AOI polygon

**Files:**
- Modify: `configs/regions/harbin.geojson`
- Modify: `configs/regions/haidian.geojson`

**Context:** 上一个 GitHub 仓库 `go-bananas-wwj/xuannv@v12-clean-dynamic` 无法通过 shell git 或 FetchURL 访问（443 超时/404）。需要替代方案。

- [ ] **Step 1: 再尝试一次通过 git clone 拉取上一个仓库**

```bash
cd /tmp
rm -rf xuannv_v12
git clone --depth 1 --branch v12-clean-dynamic https://github.com/go-bananas-wwj/xuannv.git xuannv_v12
```

若仍失败，记录错误并进入 Step 2。

- [ ] **Step 2: 从百度网盘哈尔滨高分辨率 tiff 推断 AOI**

下载哈尔滨高分辨率 DOM（22 GB）后读取 bounds 生成 polygon：

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_harbin.txt \
    --output /data/xuannv_embedding/raw/harbin/highres_optical
```

```python
import json, rasterio
from pathlib import Path
from shapely.geometry import box, mapping

src = Path('/data/xuannv_embedding/raw/harbin/highres_optical/哈尔滨_202512/哈尔滨新区_BJ3_202512_DOM_RGB.tif')
with rasterio.open(src) as ds:
    b = ds.bounds
    crs = ds.crs.to_string()
geojson = {
    'type': 'FeatureCollection',
    'crs': {'type': 'name', 'properties': {'name': crs}},
    'features': [{'type': 'Feature', 'properties': {}, 'geometry': mapping(box(b.left, b.bottom, b.right, b.top))}]
}
Path('configs/regions/harbin.geojson').write_text(json.dumps(geojson, indent=2))
```

- [ ] **Step 3: 海淀区 AOI 来源**

海淀 AOI 优先从上一个 GitHub 拉取；若拉取失败，从 ModelScope 天仪 SAR 数据或 Beijing Planet zip 推断边界（下载后读取 bounds）。

- [ ] **Step 4: 验证 patch 数量**

```python
from pyproj import Transformer
import json, numpy as np
from pathlib import Path

for r in ['harbin', 'haidian']:
    data = json.loads(Path(f'configs/regions/{r}.geojson').read_text())
    coords = np.array(data['features'][0]['geometry']['coordinates'][0])
    lon, lat = coords[:,0], coords[:,1]
    zone = int(((lon.min()+lon.max())/2 + 180)//6) + 1
    epsg = 32600 + zone
    xs, ys = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon, lat)
    nx = int(np.ceil((xs.max()-xs.min())/1280))
    ny = int(np.ceil((ys.max()-ys.min())/1280))
    print(r, 'epsg', epsg, 'grid', nx, 'x', ny, 'total', nx*ny)
```

Expected: 哈尔滨 ≈ 424，海淀 ≈ 320。

---

### Task 2: 清理数据目录并创建 README

**Files:**
- Create: `/data/xuannv_embedding/README.md`
- Create: `docs/data_layout.md`（备份）

- [ ] **Step 1: 删除测试/旧目录**

```bash
rm -rf /data/xuannv_embedding/processed/harbin_smoke
rm -rf /data/xuannv_embedding/statistics/harbin_smoke
rm -rf /data/xuannv_embedding/outputs/smoke_test
```

- [ ] **Step 2: 创建数据目录 README**

内容：说明 `raw/`、`processed/`、`statistics/`、`outputs/` 结构与当前状态。

- [ ] **Step 3: 提交文档备份**

```bash
cp /data/xuannv_embedding/README.md docs/data_layout.md
git add docs/data_layout.md
git commit -m "docs: add data directory layout documentation"
git push origin main
```

---

### Task 3: 重新下载全时段 S1/S2/Landsat

**Files:**
- Create: `scripts/data/download_pc_harbin.sh`
- Create: `scripts/data/download_pc_haidian.sh`

- [ ] **Step 1: 创建并运行哈尔滨下载脚本**

```bash
#!/usr/bin/env bash
set -e
for SRC in s2 s1 landsat; do
    python /root/workspace/xuannv/scripts/data/download_pc.py \
        --region harbin \
        --source ${SRC} \
        --start 2025-01-01 --end 2026-05-31 \
        --region-file /root/workspace/xuannv/configs/regions/harbin.geojson \
        --output-root /data/xuannv_embedding/raw
done
```

- [ ] **Step 2: 创建并运行海淀区下载脚本**

同上，region 换为 haidian。

Expected: `raw/harbin/` 与 `raw/haidian/` 下三个 source 均覆盖完整 AOI。

---

### Task 4: 下载百度网盘高分辨率数据

**Files:**
- Modify: `scripts/data/baidu_links_harbin.txt`
- Modify: `scripts/data/baidu_links_beijing.txt`
- Modify: `scripts/data/baidu_links_haidian.txt`

百度网盘目录已确认：`/玄女数据下载/`

- [ ] **Step 1: 更新 links 文件**

```bash
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

- [ ] **Step 2: 下载哈尔滨、北京、海淀数据**

```bash
python scripts/data/download_baidu.py --links-file scripts/data/baidu_links_harbin.txt --output /data/xuannv_embedding/raw/harbin/highres_optical
python scripts/data/download_baidu.py --links-file scripts/data/baidu_links_beijing.txt --output /data/xuannv_embedding/raw/beijing/highres_optical
python scripts/data/download_baidu.py --links-file scripts/data/baidu_links_haidian.txt --output /data/xuannv_embedding/raw/haidian/labels
```

---

### Task 5: 继续并完成 ModelScope 海淀 SAR 数据下载

```bash
export MODELSCOPE_TOKEN="ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
python scripts/data/download_modelscope.py \
    --dataset WeijieWu/haidian_sar_2025 \
    --output /data/xuannv_embedding/raw/haidian/highres_sar \
    --cache-dir /tmp/modelscope_cache
```

---

### Task 6: 预处理全量数据并生成 manifest/statistics

**Files:**
- Create/Modify: `scripts/data/preprocess_harbin.json`
- Create/Modify: `scripts/data/preprocess_haidian.json`
- Create: `scripts/data/preprocess_harbin_highres.json`
- Create: `scripts/data/preprocess_haidian_highres.json`

配置示例（哈尔滨）：

```json
{
  "region": "harbin",
  "raw_root": "/data/xuannv_embedding/raw/harbin",
  "output_root": "/data/xuannv_embedding/processed/harbin/scenes",
  "crs": "EPSG:32652",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["s2", "s1", "landsat", "highres"],
  "fill_missing": "zero"
}
```

运行：

```bash
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --overwrite
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --overwrite
python scripts/data/generate_manifest.py --processed-dir /data/xuannv_embedding/processed/harbin/scenes --output /data/xuannv_embedding/processed/harbin/scenes/manifest.json --sources s2 s1 landsat highres
python scripts/data/compute_statistics.py --processed-dir /data/xuannv_embedding/processed/harbin/scenes --output-dir /data/xuannv_embedding/statistics/harbin --sources s2 s1 landsat highres
```

Expected: 哈尔滨 manifest ≈ 424 条，海淀 ≈ 320 条。

---

### Task 7: 下载 2023 年分类标签数据

**Context:** ESA WorldCover 官方只有 2020 (v100) 和 2021 (v200)，没有 2023。替代方案：

- **ESRI 10m Annual Land Cover (2017–2023)**：Planetary Computer / GEE 上可用，10m，9 类。
- **Dynamic World (2015–present)**：GEE `projects/ee-ioannis-konstantinos/Google-Dynamic-World-V1` 或 GEE `GOOGLE/DYNAMICWORLD/V1`，10m，8 类。

建议采用 **ESRI 2023 LULC** 作为 WorldCover 2023 的替代。

- [ ] **Step 1: 从 Planetary Computer 下载 ESRI 2023 LULC**

```python
import planetary_computer
import pystac_client
import rioxarray

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
collection = catalog.get_collection("io-lulc-annual-v2")
```

或编写专用脚本 `scripts/data/download_esri_lulc.py`，按 AOI 裁剪下载。

- [ ] **Step 2: 预处理为与 patch 对齐的分类 target**

将下载的 ESRI 2023 标签重投影、裁剪到 harbin/haidian patch grid，作为 `worldcover` source。

---

### Task 8: 代码库修复（highres_sar、prepare_batch）

**Files:**
- Modify: `configs/base.yaml`
- Modify: `src/xuannv_embedding/training/batch_preparation.py`

- [ ] **Step 1: 在 base.yaml 注册 `highres_sar`**

```yaml
model:
  sensor_channels:
    s2: 12
    s1: 2
    landsat: 6
    highres: 3
    highres_sar: 1
```

- [ ] **Step 2: 让 `prepare_batch` 支持多个 highres source**

将 `source_name.startswith("highres")` 的源从时序源中分离，分别生成 `highres_frame_{name}`/`highres_mask_{name}`；保留 `highres_frame` 兼容。

---

### Task 9: 模型结构修正 — 输出 128×128 embedding

**Files:**
- Create/Modify: `src/xuannv_embedding/models/blocks.py`
- Modify: `src/xuannv_embedding/models/model.py`
- Remove: `spatial_stride` 相关逻辑（或保留仅用于 ablation）

**目标：** 参考 AEF 复现，引入三通路（Space 1/16、Time 1/8、Precision 1/2），最终 embedding 先压缩到 64×64，再通过 upsample head 恢复到 128×128。

- [ ] **Step 1: 实现多分辨率 STP 模块**

在 `src/xuannv_embedding/models/blocks.py` 新增 `MultiResolutionSTPBlock`：

```python
class MultiResolutionSTPBlock(nn.Module):
    def __init__(self, space_dim=256, time_dim=128, precision_dim=64):
        super().__init__()
        self.space_op = SpaceOperator(space_dim)
        self.time_op = TimeOperator(time_dim)
        self.precision_op = nn.Sequential(...)  # 3x3 conv
        # learned/interpolate cross-scale exchanges

    def forward(self, space_x, time_x, precision_x, timestamps):
        ...
        return space_x, time_x, precision_x
```

- [ ] **Step 2: 修改 AEFModel 使用 STP 并输出 128×128 embedding**

```python
# encoder outputs precision features (B, T, H/2, W/2, D)
precision_feats = self.stp_encoder(source_frames, timestamps)
# temporal summarizer -> (B, H/2, W/2, 64)
embeddings_half = self.summarizer(precision_feats, timestamps)
# upsample to (B, H, W, 64)
embeddings_full = self.embedding_upsample(embeddings_half)
```

- [ ] **Step 3: 更新配置**

移除 `spatial_stride` 配置项；新增 `model.use_multiresolution_stp: true`。

- [ ] **Step 4: 测试**

```bash
pytest tests/test_model.py tests/test_smoke.py -v
```

Expected: 输入 128×128，输出 `embedding_map` 形状 `(B, 64, 128, 128)`。

---

## 五、关键阻塞点与需要确认的事项

1. **完整 AOI**：上一个 GitHub 仓库无法访问。请确认是否可以：
   - 提供完整 `harbin.geojson` / `haidian.geojson` 文件；或
   - 同意我从百度网盘高分辨率 tiff 推断边界（哈尔滨）并从 SAR/Planet 数据推断海淀边界。

2. **WorldCover 2023**：ESA 官方没有 2023 版。请确认是否使用 **ESRI 10m Annual Land Cover 2023** 或 **Dynamic World 2023** 作为替代。

3. **数据目录清理**：是否同意删除 `harbin_smoke`、smoke checkpoint 等测试产物？

4. **模型重构**：是否同意按上述多分辨率 STP + upsample head 方案重构模型，以输出 128×128 embedding？

---

## Execution Handoff

修订版计划已保存到 `docs/superpowers/plans/2026-06-18-revised-data-and-model-plan.md`。

请确认上述 4 个阻塞点后，我开始按 Task 执行。
