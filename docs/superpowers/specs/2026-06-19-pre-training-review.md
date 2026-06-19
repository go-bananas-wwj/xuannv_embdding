# 训练前数据与代码审查报告

> 审查时间：2026-06-19
> 审查方式：4 个并行的子智能体分别审计数据、代码、行业最佳实践、patch 提取加速；并人工抽检关键 NetCDF。
> 结论：**当前不能启动训练**。必须先修复海淀区 PC 数据、重写 patch 提取流程、并完成模型架构改造。

---

## 1. 数据层关键问题

### 1.1 海淀区 S2 / S1 / Landsat 2025 数据严重异常（已确认）

| 文件 | 维度 | 问题 | 根因推断 |
|------|------|------|----------|
| `raw/haidian/s2/s2_20250101_20251231.nc` | (174, 12, 3098, 3107) | 仅 2 个时间步有 910 个非零像素，其余全 0 | 该文件来自早期/错误的下载命令，未使用当前 `download_pc_parallel.sh` 重跑 |
| `raw/haidian/s2/s2_20260101_20260531.nc` | (66, 12, 3098, 3107) | 仅 3 个时间步有 910 个非零像素 | 同上 |
| `raw/haidian/s1/s1_20250101_20251231.nc` | (98, 2, 3098, 3107) | 仅 12 个时间步有 910 个非零像素 | 同上 |
| `raw/haidian/s1/s1_20260101_20260531.nc` | (29, 2, 3098, 3107) | 仅 12 个时间步有 910 个非零像素 | 同上 |
| `raw/haidian/landsat/landsat_20250101_20251231.nc` | (88, 6, 1033, 1036) | 仅 4 个时间步有 108 个非零像素 | 同上 |
| `raw/haidian/landsat/landsat_20260101_20260531.nc` | (37, 6, 1033, 1036) | 37 个时间步均正常，最大 ~107 万非零像素 | 由 `download_pc_parallel.sh` 重新下载，数据正常 |

**证据**：抽检代码显示 Haidian S2 第 95 时间步的 910 个非零像素集中在 `x≈449000, y≈4415200`（EPSG:32650），对应经纬度 `(116.40, 39.88)`，即 bbox 的东南角，其余区域全 0。

**结论**：海淀区 2025 年的 PC 时序数据（S2/S1/Landsat）以及 2026 年的 S2/S1 都是基于旧/错误查询生成的，**必须删除后重新下载**。

### 1.2 哈尔滨 S1 / Landsat 覆盖稀疏但可用

| 文件 | 覆盖情况 |
|------|----------|
| `raw/harbin/s1/s1_20250101_20251231.nc` | 44 个时间步中部分有覆盖（最大 482 万非零像素 / 1120 万总像素），中位数为 0 |
| `raw/harbin/s1/s1_20260101_20260531.nc` | 29 个时间步全 0 |
| `raw/harbin/landsat/landsat_20250101_20251231.nc` | 90 个时间步中大部分为 0，最大仅 4428 非零像素 |
| `raw/harbin/landsat/landsat_20260101_20260531.nc` | 38 个时间步大部分覆盖良好（中位数 83 万 / 121 万总像素） |

**结论**：S1/Landsat 不应作为强制输入源；应在 manifest 中标记可用性，缺失时走 `missing_token` 路径，否则模型会退化为单 S2。

### 1.3 全 0 patch 与 nodata 混淆

- 当前 `preprocess.py` 使用 `fill_missing=zero`，对缺失时间步和 patch 外区域均填充 0。
- 哈尔滨 S2 已生成的 43,402 个 patch 中约 **76% 全为 0**，因为 patch 网格基于第一个 NetCDF 的整景边界，而非 AOI GeoJSON。
- 海淀区 S2 patch 几乎全部全 0（受原始数据影响）。
- 损失函数目前无法区分“nodata 0”与“有效反射率 0”。

**结论**：
1. patch 网格应基于 AOI GeoJSON 生成，避免整景边界外的大量无效 patch。
2. 必须记录 per-pixel valid mask（或至少过滤有效像素比例低于阈值的 patch）。
3. no-data 值保持 `0` 作为默认，但需通过独立 mask 标识；用户若坚持改为 `-1`，需要额外数据转换。

### 1.4 WorldCover 标签类别越界

- 当前 `configs/base.yaml` 中 `worldcover.channels: 11`（类别 0-10）。
- 已切出的标签 patch 中出现类别 `11`，会导致 CrossEntropy 索引越界。

**结论**：需核实 ESRI 2023 类别 11 的真实含义。可选方案：
- A. 将 `channels` 改为 12（0-11），把 0 作为 nodata/背景不参与 loss；
- B. 在 `preprocess_lulc.py` 中将 11 映射为 0（nodata）。

### 1.5 高分辨率数据未进入 processed

- 哈尔滨 DOM（0.5m，202512/202605）仍在 `raw/harbin/highres_optical/_mosaic/`，未重投影/裁剪。
- 海淀 PlanetScene 283 个瓦片（3m）未镶嵌成月度 mosaic。
- 海淀 SAR 仍为原始 zip/SLC，未地理编码。

**结论**：最小验证阶段至少需要完成哈尔滨和海淀的高分辨率光学 patch；高分辨率 SAR 可延后。

### 1.6 Manifest 与统计量落后

- 当前 manifest 只含 S2 路径列表，无 S1/Landsat/worldcover，也无 `sources.{source}.months/count`。
- 统计量仅 S2，且海淀区 S2 统计量失真。

**结论**：修复数据后必须重新生成 manifest 与统计量。

---

## 2. 代码层关键问题

### 2.1 模型架构仍为旧版“单帧 AEF”

| 组件 | 当前实现 | 需要状态 |
|------|----------|----------|
| Bottleneck | `VMFBottleneck` 含 L2 归一化 | `LearnedBottleneck` 无 L2 归一化 |
| 时序摘要 | `TemporalSummarizer` 池化为单 embedding | `MonthlyEmbeddingModule` 逐月网格 |
| 高分辨率输入 | 假设与低分辨率同尺寸 | 原生分辨率独立模态 + adaptive pooling 对齐 |
| 解码器 | 单帧输入/输出 | 支持 `(B, T, C, H, W)` 月度输入/输出 |
| 损失 | 基于时序平均 target | 逐月 masked loss |
| 数据集 | 无月度窗口采样 | `window_months` / `use_full_sequence` |
| 多 NPU | 普通 DDP，无 AMP/checkpointing | AMP + gradient checkpointing + `torch_npu.amp` |

### 2.2 关键 bug / 形状不匹配

1. `batch_preparation.py` 用 `_weighted_temporal_mean` 把 target 压为单帧；与逐月 embedding 冲突。
2. `prepare_batch` 只取第一个 source 的 timestamps 作为全局时间，若 S2/S1 观测月份不同则 binning 错误。
3. `AEFOutput` 当前为 `(B, D, H, W)`，需改为 `(B, T_month, D, H, W)`。
4. `losses.py` 中的 `batch_uniformity_loss` 假设 `embedding` 为 `(B, D)`，月度化后会 shape 报错。
5. 配置中 `data.patch_size` 在 `base.yaml` 为 256，与区域配置 128 不一致；高分辨率 source 名 `highres` 与设计稿 `highres_optical` 不一致。

### 2.3 未发现 `einops`

代码中 reshape 均为 PyTorch 原生操作，符合约束。

---

## 3. 行业最佳实践要点

### 3.1 AEF 原始工作

- 论文：*AlphaEarth Foundations* (Brown et al., arXiv:2507.22291)。
- AEF 输出为年度 64 维嵌入，使用 **vMF + 单位球面约束**；但用户明确要求移除 L2 归一化。
- AEF 训练规模极大（30 亿帧），本项目应优先做小尺度月度原型。
- 预处理：统一 UTM、1.28 km patch、log-scaling、全局 z-score + 6σ clip、Cloud Score+ 掩膜。

### 3.2 时序建模

- SatMAE / Prithvi 系列采用 **sinusoidal 时间编码 + 3D positional embedding**，优于简单 channel 堆叠。
- 独立时间掩码（Independent Masking）优于一致掩码。
- 时间窗口建议 **4–12 帧**（月度任务可对应 12 个月）。
- 不要把 64-D embedding 的每个 channel 简单对应到月份。

### 3.3 多源预处理

- S2：优先 L2A，用 SCL/Fmask 软掩膜，反射率 DN/10000，log-scaling + z-score。
- S1：GRD → 热噪声去除 → 辐射定标（σ⁰）→ 去斑 → 地形校正；常用 dB clip [-35, 0]。
- Landsat：Collection 2 L2 SR，QA_PIXEL 掩膜；与 S2 融合前做 BRDF/波段匹配（HLS 产品更优）。
- 输出统一为 UTM、10 m、COG/Zarr。

### 3.4 高分辨率融合

- 高分辨率光学/SAR 应作为**独立模态分支**，通过 cross-attention 或 feature pyramid 注入低分辨率时序主分支。
- 使用 **learned zero tokens / presence mask** 处理缺失模态，避免全零填充。
- 几何配准误差必须控制在亚像素级。

### 3.5 Patch 提取加速

- 当前 `preprocess.py` 对每个 patch、每个时间步重复做“整景重投影”，是主要瓶颈。
- 当 CRS/分辨率已一致时，应改为**一次性加载整时相 + NumPy 整数窗口切片**，实测可加速 **100–300 倍**。
- 多进程按 `(文件, 时间步)` 分发，worker 数建议 8–16。
- **NPU 不适合 patch 提取阶段**；NPU 收益在训练阶段。
- 训练阶段建议把 patch 写成 **WebDataset shards 或 Zarr**，chunk 对齐 128×128，可再提升 2–5 倍 I/O。

---

## 4. 训练前必须完成的事项（按优先级）

### P0：阻塞训练

1. **删除并重新下载海淀区 S2/S1/Landsat 2025 及 S2/S1 2026 的 PC 数据**。当前文件基于错误查询，几乎全 0。
2. **重写 `scripts/data/preprocess.py`**：
   - 按 AOI GeoJSON 生成 patch 网格；
   - 一次加载整时相，批量整数切片；
   - 记录 per-pixel valid mask；
   - 过滤有效像素比例低于阈值的 patch。
3. **修复 WorldCover 类别越界**（`channels` 改为 12 或映射 11→0）。
4. **升级 manifest 格式**为 `sources.{s2,s1,landsat,highres_*}.months/count`。
5. **完成模型架构改造**：`LearnedBottleneck`、`MonthlyEmbeddingModule`、原生分辨率 highres encoder、逐月 decoder/loss、数据集月度窗口采样。

### P1：最小验证前完成

6. 生成哈尔滨 S1/Landsat patch（带 availability mask）。
7. 处理哈尔滨 DOM 高分辨率光学 patch（CGCS2000 → EPSG:32652，原生 0.5m 裁剪）。
8. 处理海淀 PlanetScene 高分辨率光学 patch（镶嵌为月度 mosaic，原生 3m 裁剪）。
9. 重新生成所有统计量。
10. 统一配置：`patch_size=128`、source 名 `highres_optical`/`highres_sar`、Landsat 通道数 6。

### P2：训练阶段优化

11. 预生成 WebDataset / Zarr shards 供训练读取。
12. 开启 AMP、gradient checkpointing、DDP on NPU。
13. 用 `num_workers=4–8`、`pin_memory=True`。

---

## 5. 待用户决策

1. **No-data 值**：默认保持 `0` 并通过独立 mask 标识，是否改为 `-1`？（推荐保持 `0`）
2. **时间窗口长度**：默认 `window_months=12`；最小验证时是否使用全部 17 个月？
3. **S1 / Landsat 低覆盖源**：
   - A. 作为可选源，缺失时走 `missing_token`；
   - B. 仅当 patch 有足够覆盖时才参与训练；
   - C. 暂时放弃 S1/Landsat，先用纯 S2 + 高分辨率光学验证。
4. **高分辨率 SAR**：最小验证阶段是否跳过海淀 SAR（预处理链较长）？
5. **WorldCover 类别 11**：改为 12 类还是映射为 0？

---

## 6. 修订后的实施策略

建议把原计划拆为两个阶段：

- **阶段 A：数据修复与预处理加速**（1–2 天）
  - 删除错误 Haidian PC 文件并重新下载；
  - 重写 `preprocess.py`；
  - 生成 S1/Landsat/高分辨率光学 patch；
  - 修复标签类别；
  - 升级 manifest。

- **阶段 B：模型改造与最小验证训练**（2–3 天）
  - 实现 `LearnedBottleneck`、`MonthlyEmbeddingModule`、原生 highres encoder、逐月 decoder/loss；
  - 数据集月度窗口采样；
  - 多 NPU 优化；
  - 哈尔滨 + 海淀一起训练一个模型，观察 loss 与 OOM。

**不建议现在启动训练。** 必须先完成阶段 A，并确保 `pytest` 全绿后，再进入阶段 B。
