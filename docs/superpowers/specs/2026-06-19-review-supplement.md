# 训练前审查补充：S1/Landsat 下载 bug、时序采样与 LULC 处理

> 补充时间：2026-06-19
> 来源：3 个并行的子智能体（S1/Landsat 排查、时序采样调研、LULC 调研）+ 人工抽检

---

## 1. S1 / Landsat 低覆盖原因：不是卫星稀疏，是下载脚本 bug

### 1.1 核心结论

**S1 和 Landsat 在中国东北/华北的覆盖并不稀疏。** 哈尔滨/海淀的 S1 6–12 天重访、Landsat 16 天重访，均有常规过境。当前 NetCDF 中大量全 0 切片，主要是以下脚本问题叠加导致的：

1. **PC SAS token 在长时间 `ds.load()` 期间过期**，读取失败返回 404；
2. `stackstac.stack` 默认 `errors_as_nodata=(RasterioIOError('HTTP response code: 404'),)`，会把读取失败当成 nodata；
3. 脚本使用 `fill_value=0.0`，失败/无数据区域被静默填成 0；
4. 脚本设置了 `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"`，但 S1 RTC 的 asset 后缀可能是 `.tiff`，会被 GDAL 拦截；
5. 哈尔滨 S1 跨 UTM 51/52 两个带，脚本取首个 item 的 EPSG，导致 2025（32651）与 2026（32652）网格不一致；
6. Landsat 缺少 `qa_pixel` 波段，无法做云/雪/阴影掩膜。

### 1.2 证据

| 证据 | 说明 |
|---|---|
| 哈尔滨 S1 2026 NetCDF | `29×2×3178×3439`，所有 time slice 全 0 |
| 单 item 直接读取 | 用 `rasterio` 读取同一 AOI 区域，`vv` 波段有约 1,120 万非 NaN 像素 |
| `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"` | 复现：同一 S1 `.tiff` asset 在此配置下无法打开 |
| 哈尔滨 Landsat 2025 NetCDF | 90 time slices，仅 12 个有非零像素，最大 26,568/7,294,920 |
| 单 item 直接读取 | `LC08_L2SP_118028_20250102` 的 `blue` asset AOI 区域约 120 万非零像素 |
| 日志 | S1/Landsat 日志仅显示 DONE/OK，无异常；失败被 `errors_as_nodata` 静默吞掉 |

### 1.3 修复方案

修改 `scripts/data/download_pc.py`：

```python
# 1. 扩展 allowed extensions
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.jp2")

# 2. 关闭 404 静默填值，fill_value 用 NaN
import numpy as np
from rasterio import RasterioIOError

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

# 3. 写 NetCDF 前把 NaN 转成 0（或保留 NaN 并在预处理时处理）
# 推荐：保留 NaN，在预处理阶段生成 valid mask
```

4. **固定 EPSG 为 AOI 中心点所在 UTM 带**，避免跨带网格不一致。
5. **缩短下载窗口**：S1/Landsat 建议按月或 1–2 个月切分，避免 token 过期。
6. **Landsat assets 增加 `qa_pixel`**：
   ```python
   "landsat": ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
   ```
7. **下载后校验**：对每个 NetCDF 统计每个 time slice 的非 NaN/非零像素比例，低于阈值则重试。

### 1.4 是否保留 S1 / Landsat

| 数据源 | 建议 |
|---|---|
| **S1** | **保留并修复下载**。SAR 是光学数据的重要补充，哈尔滨冬季积雪/云覆盖下尤其有价值；RTC 产品与 S2 10 m 对齐方便融合。 |
| **Landsat** | **修复后可保留，但可设为可选**。30 m 分辨率对 1280 m patch 较粗；若资源紧张，可先用 S2+S1 跑通，再决定是否加入 Landsat。 |

---

## 2. 时序采样策略：不应随机采样月份，应用“连续窗口 + 随机起始 + Temporal Dropout”

### 2.1 行业调研结论

| 模型 | 时间采样策略 | 默认长度 |
|---|---|---|
| **Prithvi-EO-2.0** | 4 个时间戳，相邻间隔 1–6 个月 | `T=4` |
| **SatMAE** | 长度为 3 的时序，独立时间掩码 | `T=3` |
| **SkySense** | S2 随机采 20 帧，S1 随机采 10 帧 | S2:20, S1:10 |
| **Presto** | **12 个月连续像素时间序列** | `T=12` |
| **TimeSenCLIP** | 月度聚合 + Random TSDrop（随机丢弃 1–11 个月） | 最多 12 |
| **AnySat** | 随机掩码 50% 时间戳 | 可变 |
| **AEF** | 年度窗口，连续时间编码 | 年度 |
| **SITS-Former** | 24 个时间步 | `T=24` |

### 2.2 关键发现

1. **不建议直接输入完整 17 个月**。时空 MAE/Transformer 的 token 数随 `H×W×T` 增长，17 个月会显著增加显存与计算开销。
2. **默认窗口长度建议 `T=12` 个月**（一个完整物候/水文年）。Presto、TimeSenCLIP 等主流月度模型均使用 12 个月。
3. **训练策略**：固定长度连续窗口 + 随机起始位置 + Temporal Dropout。
   - 连续窗口保留物候时序结构；
   - 随机起始位置在不破坏时序结构的前提下增加样本多样性；
   - Temporal Dropout（AnySat 式 50% 时间掩码 或 TimeSenCLIP 式 Random TSDrop）增强对缺失数据的鲁棒性。
4. 多模态（S2/S1/Landsat）不要求帧严格对齐：每个传感器独立维护可用性掩码，缺失月份用 mask token / 零值+掩码填充。

### 2.3 针对本项目的推荐配置

```yaml
model:
  temporal_window_months: 12        # 默认 12 个月连续窗口
training:
  temporal_dropout: 0.3             # 随机丢弃 0-30% 时间戳
  temporal_sampling: "random_start" # 随机起始位置；备选 "sliding"
  use_full_sequence: false          # 最小验证时若显存允许可设为 true
```

**注意**：不是从 17 个月中随机挑 12 个月，而是随机选一个起始月，截取连续 12 个月。

---

## 3. LULC / WorldCover 处理

### 3.1 ESRI 2023 WorldCover 类别

- **ESRI 2023（及 2017–2024 v3）使用 9 个有效类别**，原始像元值为 **1, 2, 4, 5, 7, 8, 9, 10, 11**。
- **Class 11 = Rangeland**（牧场/草原与灌木混合区），是合法类别，应保留。
- **Class 0 = No Data**，应作为 `ignore_index` 不参与损失。
- 类别值不连续（缺少 3, 6），必须做索引重映射。

### 3.2 推荐方案

**重映射为 9 个连续类别**（包含 Clouds）：

| 原始值 | 新索引 | 类别 |
|---|---|---|
| 0 | ignore_index | No Data |
| 1 | 0 | Water |
| 2 | 1 | Trees |
| 4 | 2 | Flooded Vegetation |
| 5 | 3 | Crops |
| 7 | 4 | Built Area |
| 8 | 5 | Bare Ground |
| 9 | 6 | Snow/Ice |
| 10 | 7 | Clouds |
| 11 | 8 | Rangeland |

配置：

```yaml
model:
  target_heads:
    lulc:
      loss_type: categorical
      channels: 9
      ignore_index: 0
      class_weight: "inverse_sqrt"
```

**损失函数**：加权 CrossEntropy + Focal Loss + Lovász/Dice 组合，处理类别不平衡。

### 3.3 是否继续使用 WorldCover

可以继续使用 ESRI 2023 WorldCover。主流基础模型（Prithvi、SatMAE、DOFA、GFM）在预训练阶段通常不直接用 LULC 作为重建目标，而是作为下游分割任务。Satlas 是例外，它使用 137 类监督预训练。本项目把 LULC 作为多任务解码头是合理的。

---

## 4. 已确认的用户决策

| 问题 | 用户决策 | 备注 |
|---|---|---|
| No-data 值 | **保持 0** | 通过独立 valid mask 标识 |
| S1 / Landsat | **保留并修复下载 bug** | S1 强烈推荐保留；Landsat 可选 |
| 时序采样 | **连续窗口 + 随机起始 + Temporal Dropout** | 默认 `T=12`，不从 17 个月随机挑 |
| WorldCover class 11 | **保留并重映射** | 9 通道输出，0 为 ignore_index |

---

## 5. 对实施计划的修正

1. 阶段 A 中重新下载 S1/Landsat 时，必须同步修复 `download_pc.py` 的 bug。
2. 阶段 B 的 `MonthlyEmbeddingModule` / `AEFDataset` 按 `T=12` 连续窗口 + 随机起始实现。
3. 在 `prepare_batch` / loss 中实现 Temporal Dropout 与模态缺失掩码。
4. `preprocess_lulc.py` 增加 ESRI → 连续索引的重映射。
5. `configs/base.yaml` 中 `lulc.channels` 改为 9，并配置 `ignore_index`。
