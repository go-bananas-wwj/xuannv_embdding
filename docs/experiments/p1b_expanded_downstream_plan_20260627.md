# P1B 扩展下游任务计划 - 2026-06-27

本文档规划如何使用 P1B embedding 作为固定底座，通过轻量下游头完成建筑分割、路网提取、土地利用/土地覆盖、水体提取等任务，用于证明 embedding 的通用遥感表达能力，并生成可汇报的指标和可视化。

## 1. 固定底座

本轮不重新训练 embedding，固定使用 P1B：

- Config：`configs/v2_p1_sparse_sampler_hardneg_20260627_090500.yaml`
- Checkpoint：`/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- Embedding root：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- 已有汇报包：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627`

核心原则：所有新任务都只训练下游头，embedding 冻结。这样能直接回答“P1B embedding 本身有没有泛化表达能力”。

## 2. 当前数据盘点

已确认可用：

| 数据/标签 | 当前状态 | 说明 |
|---|---|---|
| P1B embedding | 可用 | Haidian / Harbin，202512 与 202605 两期。 |
| WorldCover | 可用 | Haidian 约 320 个 patch，Harbin 约 424 个 patch，可直接做土地覆盖分割和派生弱标签。 |
| construction | 可用 | 已用于施工地/建筑施工相关任务。 |
| building_change | 可用 | 这是变化标签，不等同于静态建筑 footprint。 |
| farm_change / rubbish | 可用 | 已在 P1B 汇报包中覆盖。 |

未确认或缺失：

| 数据/标签 | 当前状态 | 处理策略 |
|---|---|---|
| 静态建筑 footprint | 未发现可信 processed 标签 | 第一阶段用 WorldCover built-up 类做弱建筑分割；第二阶段再接入 OSM/Microsoft/本地矢量 footprint。 |
| 道路/路网标签 | 未发现 processed 标签 | 第一阶段先准备标签构建脚本；若无本地道路矢量，不能把伪标签当强监督结果汇报。 |
| JRC Water / 水体标签 | raw 目录存在但当前没有文件 | 第一阶段从 WorldCover 水体类派生 weak water mask；若后续补 JRC/NDWI 标签，再升级为强水体测评。 |

## 3. 任务设计

### 3.1 第一批：立刻可做的任务

这些任务不依赖额外下载或人工标注，可基于现有 WorldCover / 已有标签直接跑。

| 任务 ID | 中文任务 | 标签来源 | 类型 | 输入 embedding | 下游头 | 指标 | 汇报定位 |
|---|---|---|---|---|---|---|---|
| `landcover_segmentation` | 土地覆盖多类分割 | WorldCover | 多类语义分割 | 单时相 202605，必要时比较 202512 | linear probe + lightweight UperNet | mIoU、mean F1、per-class IoU | 证明 embedding 保留地物语义。 |
| `landuse_patch_classification` | patch 级土地利用分类 | WorldCover 多数类/面积比例派生 | 多类分类 | patch pooled embedding | MLP classifier | accuracy、macro F1、confusion matrix | 证明 embedding 可做区域级检索/分类。 |
| `builtup_weak_segmentation` | 建成区/建筑弱分割 | WorldCover built-up 类 | 二分类分割 | 单时相 202605 | UperNet / linear probe | AUC、F1_best、mIoU | 作为建筑分割的弱标签版本。 |
| `water_weak_segmentation` | 水体弱分割 | WorldCover water 类 | 二分类分割 | 单时相 202605 | UperNet / linear probe | AUC、F1_best、mIoU | 快速展示水体提取能力。 |
| `vegetation_farmland_segmentation` | 植被/耕地类分割 | WorldCover vegetation/cropland 类 | 二分类或少类分割 | 单时相 202605 | UperNet / linear probe | mIoU、F1_best | 补充自然地物能力展示。 |

说明：WorldCover 属于较粗标签，部分任务应标注为 `weak_gt`。汇报时可以讲“弱监督/公开地表覆盖标签验证”，不要把它包装成精细建筑轮廓或道路真值。

### 3.2 第二批：需要补标签后做的任务

这些任务更有汇报价值，但必须先补齐可信标签。

| 任务 ID | 中文任务 | 推荐标签来源 | 类型 | 难点 | 验收指标 |
|---|---|---|---|---|---|
| `building_footprint_segmentation` | 静态建筑轮廓分割 | 本地矢量建筑、OSM、Microsoft Building Footprints、人工标注 | 二分类分割 | 需要栅格化到 128x128 patch，对齐坐标 | mIoU、boundary F1、F1_best |
| `road_network_extraction` | 路网提取 | 本地道路矢量、OSM road centerline | 细线二分类分割 | 道路很细，普通 IoU 会低估，需要 relaxed F1 / skeleton 指标 | relaxed F1、IoU、connectivity proxy |
| `strong_water_segmentation` | 强水体提取 | JRC Water、Sentinel-2 NDWI 人工确认、Hydro 数据 | 二分类分割 | 时间对齐和季节性水体 | mIoU、F1_best、AUC |

## 4. 下游模型方案

### 4.1 输入策略

- 静态语义任务：默认使用 `202605_embedding_map.pt`。
- 变化任务：继续使用 `concat_diff = [emb_202512, emb_202605, abs(diff)]`。
- 汇报中强调：静态任务测试语义表达，变化任务测试时序敏感性。

### 4.2 下游头

保留两个复杂度级别，避免只用一个 head 造成说服力不足：

| Head | 用途 | 解释 |
|---|---|---|
| Linear probe | 最纯粹评估 embedding 线性可分性 | 如果 linear probe 好，说明 embedding 本身表达强。 |
| Lightweight UperNet / 小型 CNN head | 展示实用效果 | 允许一点空间上下文融合，更接近真实下游应用。 |
| MLP classifier | patch 级土地利用分类 | 对 embedding 做 average/max pooling 后分类。 |

### 4.3 Loss

- 多类土地覆盖：CrossEntropy + ignore_index。
- 二分类稀疏任务：BCEWithLogits + Dice/Tversky。
- 道路：BCE/Dice + distance transform 或 relaxed skeleton metric，避免细线任务被普通 IoU 误伤。

## 5. 需要改的代码

| 模块 | 改动 | 目的 |
|---|---|---|
| `downstreams/downstreams/data/embedding_dataset.py` | 支持非 `masks/patch_id.tif` 的 WorldCover 文件命名；支持多类 mask remap；支持 weak_gt 元数据 | 让 WorldCover/水体/建成区等任务能统一加载。 |
| `downstreams/downstreams/tasks/` | 新增 generic segmentation task / classification task | 不再把所有任务都塞进 construction task。 |
| `downstreams/downstreams/metrics/segmentation.py` | 增加 multi-class mIoU、per-class IoU、macro F1；增加 road relaxed metric 预留 | 支持土地覆盖和路网任务。 |
| `downstreams/configs/` | 新增完整自包含配置，禁止 `_base_` | 符合项目配置规范。 |
| `scripts/scale/` 或 `scripts/report/` | 新增 expanded downstream suite runner | 一键跑多任务、出报告和可视化。 |
| `scripts/report/build_p1b_presentation_package.py` | 扩展到能力矩阵 | 把新任务加入汇报包。 |

## 6. 执行阶段

### Phase A：标签 QA 与任务清单固化

目标：确认哪些任务是强标签，哪些任务是弱标签。

产物：

- `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_label_inventory_YYYYMMDD/`
- `docs/experiments/p1b_expanded_downstream_label_inventory_YYYYMMDD.md`

检查项：

- WorldCover 类别分布；
- built-up / water / vegetation 派生 mask 的正样本比例；
- embedding 与 mask 尺寸是否完全一致；
- 是否有缺失 patch；
- 每个任务是否能做 5-fold stratified split。

### Phase B：先跑 quick benchmark

目标：用 1 fold 或 20% patch 快速验证流程，避免完整 5 折浪费时间。

首批任务：

1. `landcover_segmentation`
2. `landuse_patch_classification`
3. `builtup_weak_segmentation`
4. `water_weak_segmentation`
5. `vegetation_farmland_segmentation`

产物：

- quick 指标表；
- 每类任务 3-5 张可视化；
- 失败任务的问题定位。

### Phase C：完整 5-fold benchmark

目标：对 quick 表现稳定的任务做完整 5 折，作为正式汇报数字。

输出目录建议：

`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_capability_suite_YYYYMMDD_HHMMSS/`

每个任务输出：

- `summary.json`
- `summary_5fold.json`
- `per_class_metrics.json`
- `threshold_calibration.json`，仅二分类任务
- `predictions/`
- `visualizations/`

### Phase D：汇报包升级

目标：把原 P1B 汇报包升级成“能力矩阵”。

新增内容：

- 能力矩阵表：变化检测、建筑/建成区、水体、土地覆盖、土地利用分类；
- 每个任务 2-4 张代表图；
- 强标签/弱标签明确标注；
- 可直接放 PPT 的横向条带图；
- 与 AEF 或现有 baseline 的可比项保留对比，不可比项只展示 P1B 绝对效果。

## 7. 推荐执行顺序

建议按这个顺序做，风险最低：

1. `landcover_segmentation`：标签已存在，最能证明通用地物语义。
2. `water_weak_segmentation`：从 WorldCover 水体类派生，视觉效果通常清晰。
3. `builtup_weak_segmentation`：作为建筑分割的第一版弱标签验证。
4. `landuse_patch_classification`：汇报友好，指标容易解释。
5. `vegetation_farmland_segmentation`：补充自然地物能力。
6. `road_network_extraction`：等道路标签或 OSM 栅格化流程确认后再做。
7. `building_footprint_segmentation`：等可信建筑 footprint 标签后再做强版本。

## 8. 成功标准

第一阶段成功标准：

- 跑通至少 4 个新增任务，其中至少 2 个有完整 5-fold 结果。
- 每个任务都有指标、预测图、GT 图、embedding PCA 图。
- 汇报包能明确区分 strong_gt 和 weak_gt。
- 不影响原 P1B sparse-change benchmark。

汇报标准：

- 土地覆盖：展示 per-class IoU 和混淆矩阵。
- 水体：展示预测概率和 GT 边界，突出边界清晰度。
- 建成区/建筑弱分割：明确是 WorldCover built-up 弱标签，不声称精细建筑轮廓。
- 路网：没有可信标签前，只做计划和标签构建，不输出夸大指标。

## 9. 需要你确认的点

1. 第一阶段是否接受用 WorldCover 派生的 `builtup_weak_segmentation` 作为“建筑分割弱标签版”？
2. 路网提取是否有你本地可提供的道路矢量/栅格标签？如果没有，我会先做 OSM 栅格化脚本和弱标签标注说明。
3. 水体提取是否可以先用 WorldCover water 类做弱标签？后续再补 JRC/NDWI 强验证。
4. 汇报时是否需要把 weak_gt 和 strong_gt 分开成两页，避免听众误解标签精度？

## 10. 我建议的默认决策

如果没有额外标签，我建议先执行：

- `landcover_segmentation`
- `water_weak_segmentation`
- `builtup_weak_segmentation`
- `landuse_patch_classification`
- `vegetation_farmland_segmentation`

暂缓：

- `road_network_extraction`，等道路标签；
- `building_footprint_segmentation`，等可信建筑 footprint。

这样可以最快生成一版“P1B 通用能力汇报包”，同时不把弱标签包装成强结论。
