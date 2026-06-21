# xuannv_embedding 项目工作总结报告设计文档

> 日期：2026-06-21
> 目标：整理项目阶段成果，生成可汇报的 HTML 报告及相关原始材料
> 输出目录：`/root/workspace/report/`

---

## 1. 背景与目标

用户需要一份面向汇报的 xuannv_embedding 项目阶段工作总结，涵盖：

- 数据准备、模型训练、架构设计、多源数据运用的整体进展；
- 与 AEF（AlphaEarth Foundations）官方 2025 年度 embedding 的对比效果；
- 模型升级方向（V2.0），但**不明确提及“蒸馏”**；
- 具体的可视化案例。

报告需输出为 HTML（图片单独存放于 `report/assets/`），并同步整理原始材料到 `/root/workspace/report/`。

---

## 2. 受众与基调

- 受众：项目相关决策者、合作方或汇报对象，具备一定技术背景但不需要底层实现细节。
- 语言：中文。
- 风格：数据驱动、结论清晰、可视化为主，避免堆砌技术公式与超参数细节。

---

## 3. 报告结构

### 3.1 封面
- 标题：xuannv_embedding 项目阶段工作总结
- 副标题：月度地理嵌入预训练与下游任务验证进展
- 日期、区域（哈尔滨 / 海淀）

### 3.2 项目目标回顾
- 输入：Sentinel-1 / Sentinel-2 / Landsat + 稀疏高分辨率光学/SAR
- 输出：128×128 像素级月度地理 embedding
- 下游：轻量头即可做变化检测、土地覆盖分类等任务

### 3.3 工作总览 / 里程碑
用卡片/时间线列出关键节点：
- 数据下载与预处理 pipeline 完成
- 128×128 patch 切分与 manifest 生成
- 多源月度 embedding 模型（阶段一/阶段二）训练完成
- V1.0 下游 5 任务 5-fold 评测完成
- AEF 官方 embedding benchmark 完成
- 已生成约 1.6T embedding 产物

### 3.4 数据准备
- 两个实验区：哈尔滨、海淀
- 数据源：S2(12 波段)、S1(2 波段)、Landsat(7 波段)、WorldCover 标签、高分光学/SAR
- 预处理：对齐到统一 10m UTM 网格、切 patch、统计量与 manifest
- 当前状态：低分辨率源已就绪；高分源已在阶段二融入训练

### 3.5 模型架构设计（高层描述）
- 整体流程图：多源时序输入 → SensorEncoderBank → STP Encoder → MonthlyEmbeddingModule → EmbeddingUpsampleHead → vMF Bottleneck → Decoder Heads
- 关键创新点：月度 binning、多分辨率时空编码、vMF 单位球面约束、可用性感知高分融合
- 不展开具体维度、loss 公式、训练超参数

### 3.6 多源数据如何运用
- 低分辨率时序源：按 YYYYMM 月度 bin，缺失月份用 learnable missing_token
- 高分辨率源：独立编码分支，adaptive pool 对齐后逐月融合
- 阶段一 vs 阶段二的数据源差异

### 3.7 模型训练进展
- 阶段一：哈尔滨 / 海淀，S2+S1+Landsat+WorldCover
- 阶段二：加入高分光学 / SAR 融合
- 训练平台：6×Ascend NPU，DDP + AMP + gradient checkpointing
- 可放 loss 收敛趋势图（不列具体超参数）

### 3.8 效果展示与 AEF 对比（重点章节）
- 评测方式：5 个下游任务 5-fold 交叉验证
- 对比指标：AUC-ROC、F1-best、F1@0.5、mIoU
- 具体数字（来源：`/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md`）：

| 任务 | V1.0 AUC | AEF AUC | V1.0 F1-best | AEF F1-best | V1.0 mIoU | AEF mIoU |
|---|---:|---:|---:|---:|---:|---:|
| construction | 0.746 | 0.865 | 0.214 | 0.212 | 0.027 | 0.055 |
| building_change | 0.883 | 0.683 | 0.116 | 0.025 | 0.033 | 0.006 |
| farm_change | 0.888 | 0.671 | 0.121 | 0.015 | 0.038 | 0.002 |
| rubbish | 0.888 | 0.812 | 0.121 | 0.037 | 0.038 | 0.004 |
| construction_joint | 0.812 | 0.893 | 0.190 | 0.365 | 0.080 | 0.178 |

- 结论：
  - **自研 V1.0 优于 AEF**：变化检测类任务（building_change、farm_change、rubbish）在 AUC、F1、mIoU 上均领先；其中 farm_change AUC 领先 0.217，F1-best 领先 0.106。
  - **AEF 优于自研 V1.0**：construction 单时相与 construction_joint 联合任务，AEF 在 AUC/F1/mIoU 上均更高；construction_joint 任务 AEF 的 F1-best 高出 0.175，mIoU 高出 0.098。
- 可视化：AEF vs V1.0 指标对比柱状图

### 3.9 模型升级方向（V2.0）
- 不提“蒸馏”，表述重点为：
  - **数据规模扩大**：扩大预训练数据覆盖范围与多样性
  - **时序连续性增强**：更精细的连续时间建模
  - **多尺度/多任务学习目标**：引入对比学习等信号
  - **下游任务增强**：更强分割头、稀疏标注策略优化
  - **端到端微调优化**：释放 encoder 能力
- 已做铺垫：阶段二高分融合、AEF benchmark 验证

### 3.10 典型案例可视化
- 5 个任务各选 1–2 个代表性 patch
- 每张图包含：RGB 底图、GT mask、预测概率热力图、预测 mask 叠加
- 优先展示：
  - construction / construction_joint（AEF 领先的正面参照）
  - building_change / farm_change / rubbish（自研领先的案例）

### 3.11 附录 / 原始材料清单
- 列出 `report/assets/` 与 `report/data/` 中所有原始图表、JSON、说明文件的路径与用途

---

## 4. 输出文件布局

```
/root/workspace/report/
├── index.html              # 完整 HTML 报告
├── assets/                 # 图片素材
│   ├── aef_vs_v1.0.png
│   ├── task_metrics_comparison.png
│   ├── architecture_diagram.png
│   ├── construction_overlay_*.png
│   ├── building_change_overlay_*.png
│   ├── farm_change_overlay_*.png
│   ├── rubbish_overlay_*.png
│   └── construction_joint_overlay_*.png
├── data/                   # 原始指标与说明
│   ├── v1.0_summary.json
│   ├── aef_benchmark_summary.json
│   └── README.md
└── sources/                # 引用来源记录
    └── sources.md
```

---

## 5. 数据来源

- 项目代码：`/root/workspace/xuannv/`
- 训练输出与实验结果：`/data/xuannv_embedding/experiments/v1.0/`、`/data/xuannv_embedding/experiments/aef_benchmark/`
- 可视化文件：`/data/xuannv_embedding/experiments/v1.0/visualizations/`、`/data/xuannv_embedding/experiments/aef_benchmark/visualizations/`
- 设计文档：`docs/superpowers/specs/2026-06-21-work-summary-report-design.md`

---

## 6. 成功标准

- `/root/workspace/report/` 目录存在且包含 `index.html`、`assets/`、`data/`、`sources/`
- HTML 报告包含上述 11 个章节，AEF 对比使用具体数字
- 所有引用的图片路径正确，可在浏览器中正常打开
- 不提及“蒸馏”，模型升级章节重点突出数据规模扩大
- 设计文档已提交到 git 并 push

---

## 7. 后续步骤

1. 用户批准本设计文档。
2. 调用 `writing-plans` skill 生成实施计划。
3. 按 plan 整理素材、生成 HTML、提交并 push。
