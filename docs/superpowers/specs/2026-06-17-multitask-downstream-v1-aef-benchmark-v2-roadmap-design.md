# 多任务下游分割：V1.0 版本、AEF Benchmark 与 V2.0 路线图设计

## 1. 背景与目标

当前 `feat/multitask-downstream` 分支已完成 5 个下游任务的 5-fold 训练，但 F1 远低于目标。为便于后续迭代与公平对比，需要：

1. **固定当前状态为 V1.0**：代码、权重、实验结果统一归档并发布。
2. **下载 AEF 官方 2025 年度 embedding**，在相同数据/划分/任务上跑 benchmark，判断自研 embedding 与官方 embedding 的差距。
3. **根据 benchmark 结果启动 V2.0 改进**：若 AEF 显著更优，则同步推进（a）embedding 预训练重新设计 与（b）下游分割头增强。

## 2. 设计原则

- **路径清晰**：所有新产物按 `v1.0/`、`aef_benchmark/`、`v2.0/` 阶段分目录存放。
- **可复现**：每个阶段保留 commit/tag、config、依赖、指标汇总。
- **最小侵入**：尽量复用现有 `downstreams/scripts/train_task.py` 与数据集接口。
- **安全**：ModelScope token 等敏感信息只通过环境变量传入，不写入任何文件。

## 3. 阶段一：V1.0 版本发布

### 3.1 产物清单

| 产物 | 位置/方式 | 内容 |
|---|---|---|
| Git tag | `v1.0-multitask-downstream` on `feat/multitask-downstream` | 当前代码快照 |
| GitHub Release | 关联上述 tag | `FINAL_REPORT.md`、`all_tasks_summary_final.json`、关键可视化 |
| ModelScope Release | `WeijieWu/xuannv_train_data` tag `v1.0` | 预训练 AEF encoder checkpoint + 5 任务 best fold head 权重 + config + README |
| 本地快照清单 | `/data/xuannv_embedding/experiments/v1.0/v1.0_manifest.json` | commit、tag、sha256、指标、路径、依赖版本 |
| 本地归档 | `/data/xuannv_embedding/experiments/v1.0/` | 符号链接/复制 5 任务实验目录、最终报告、汇总 JSON |

### 3.2 ModelScope 发布方式

参考 ModelScope 最佳实践（模型版本即 git tag）：

1. 使用 `modelscope.hub.api.HubApi` 登录（token 来自 `MODELSCOPE_SDK_TOKEN` 环境变量）。
2. 使用 `modelscope.hub.repository.Repository` 将本地模型目录（encoder + heads）推送到 `WeijieWu/xuannv_train_data`。
3. 本地打 `git tag v1.0 -m "v1.0 multitask downstream release"` 并 push tag。
4. 在 ModelScope 网页基于该 tag 创建 Release，填写说明。

### 3.3 文件路径结构

```
/data/xuannv_embedding/experiments/v1.0/
├── v1.0_manifest.json
├── FINAL_REPORT.md
├── all_tasks_summary_final.json
├── encoder/
│   └── stage2_v1_best/
│       ├── checkpoint/
│       └── meta.json
├── heads/
│   ├── construction_upernet_harbin_fold0_best.pt
│   ├── construction_upernet_harbin_fold1_best.pt
│   ├── ...（每个任务 best fold 一个 head）
│   └── configs/
└── visualizations/
    ├── construction_overlay.png
    ├── building_change_overlay.png
    ├── ...
    └── pr_curves_all_tasks.png
```

## 4. 阶段二：AEF 官方 Embedding Benchmark

### 4.1 数据源

- **数据集**：AlphaEarth Foundations Satellite Embedding Dataset on Source Cooperative
- **推荐访问路径**：Zarr mosaic
  - URI：`s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/`
  - Shape：`(9, 64, H, W)`，time 维度对应 2017–2025
  - CRS：`EPSG:4326`
  - 分辨率：约 10 m
  - 数据类型：int8 量化，`nodata=-128`
  - 反量化公式：`float = ((x / 127.5) ** 2) * sign(x)`

### 4.2 时间对齐策略

由于 AEF 公开 embedding 为年度汇总，时间粒度为年，无法直接对齐现有任务的月份级双时相（202512 vs 202605）。经确认，benchmark 采用以下策略：

- **construction / construction_joint**：使用 2025 年度 embedding 作为单时相输入。
- **building_change / farm_change / rubbish**：同样使用 2025 年度 embedding，将任务退化为单时相语义分割任务，以保持与 construction 一致的公平对比。

> 注：此策略会削弱变化检测任务的语义，但能在统一输入下横向对比 embedding 质量。结果解读时需明确说明。

### 4.3 数据准备流程

1. 读取哈尔滨与海淀所有标注 patch 的经纬度 bbox。
2. 对每个 patch，从 AEF Zarr mosaic 中读取 2025 年（time index 8）的 `(64, H, W)` embedding。
3. 反量化并转置为 `(H, W, 64)` float32。
4. 按现有 patch 的 GeoTIFF 投影/范围重采样到 128×128 像素。
5. 保存为 `.pt` 文件，复用现有 `EmbeddingDataset` 的读取逻辑。
6. 输出目录：`/data/xuannv_embedding/embeddings/aef_official_2025_annual/`。

### 4.4 训练与评估

- 复用 `downstreams/scripts/train_task.py` 与 `downstreams/configs/`。
- 每个任务修改 `embedding_root` 指向 AEF embedding。
- 输入 channel 从自研 embedding 的 128（或 192 双时相）改为 64。
- 5-fold 交叉验证，保存 `summary_5fold.json`。
- 输出目录：`/data/xuannv_embedding/experiments/aef_benchmark/<task>_aef_2025_5fold/`。

### 4.5 对比报告

生成 `/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md`，包含：

- 每个任务 V1.0 vs AEF 的 AUC/F1_best/mIoU 并排表
- 相对提升/下降百分比
- 典型 patch 可视化对比
- 结论：AEF 是否在下游任务上显著优于自研 embedding

## 5. 阶段三：V2.0 改进路线图

若 AEF benchmark 显示官方 embedding 显著更优，则同步推进两条线：

### 5.1 线 A：重新设计 Embedding 预训练（根本改进）

调研方向（基于 AEF 论文与社区最佳实践）：

- **数据规模与多样性**：AEF 使用 30 亿+观测、9 种数据源 + 文本，覆盖 1.1% 陆地面积。V2.0 需要扩大预训练数据。
- **模型容量**：检查当前 AEFModel 是否完整复现了 Space Time Precision (STP) encoder、vMF bottleneck、teacher-student-text 框架。
- **训练目标**：除重建损失外，加入对比学习、文本-影像对齐、batch uniformity loss。
- **时间连续性**：支持连续时间编码，而不仅是月度/年度离散 embedding。
- **量化与部署**：参考 AEF int8 量化策略，减少存储。

### 5.2 线 B：增强下游分割头（快速验证）

在固定当前 embedding 上尝试：

- 更强的 head：DeepLabV3+、SegFormer、SAM-style decoder
- 更针对稀疏标注的 loss：Lovász loss、OHEM、Focal Tversky
- 数据增强：Mixup/CutMix on embeddings、大规模随机裁剪、时序差分增强
- 端到端微调：在更大 NPU 显存或更小 crop 下尝试释放 encoder 能力
- 后处理：CRF、连通域过滤、集成

### 5.3 V2.0 路径结构

```
/data/xuannv_embedding/experiments/v2.0/
├── embedding_retrain/
│   ├── experiments/
│   └── checkpoints/
└── downstream_enhance/
    ├── construction_deeplabv3plus/
    ├── building_change_deeplabv3plus/
    └── ...
```

## 6. 可视化方案

每个任务选择 1–2 个典型 patch，生成：

- RGB 底图
- GT mask 透明叠加
- Prediction probability 热力图 + 阈值轮廓
- 单任务 PR 曲线
- 5 任务指标对比柱状图

输出到 `/data/xuannv_embedding/experiments/v1.0/visualizations/`。

## 7. 不确定性与风险

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| source.coop 下载慢/不稳定 | AEF benchmark 延迟 | 分块下载、断点续传、限制只下载标注 patch 周边小区域 |
| AEF 为年度 embedding，时间对齐弱 | 变化检测任务不公平 | 报告明确说明，并优先关注 construction 单时相结果 |
| NPU 内存不足 | V2.0 端到端微调困难 | 使用梯度检查点、更小 crop、混合精度 |
| ModelScope 上传失败 | v1.0 发布不完整 | 本地保留完整备份，支持手动上传 |
| V2.0 数据量扩大后训练时间极长 | 进度不可控 | 先小批量 ablation，验证收益后再放大 |

## 8. 后续步骤

1. 用户批准本设计文档。
2. 使用 `writing-plans` skill 为三个阶段分别输出详细实施计划。
3. 进入 goal 模式，按顺序执行三个阶段。
