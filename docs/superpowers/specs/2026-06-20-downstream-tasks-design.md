# 下游任务（Downstream）框架设计草案

> 日期：2026-06-20
> 状态：草案，待子智能体审查与最终确认
> 关联需求：`chats/chat6.txt`

## 目标

为 `xuannv_embedding` 预训练模型建立一个**轻量、可扩展**的下游任务评测框架，满足：

1. 预训练模型（stage1/stage2 `best.pt`）**离线生成并保存 embedding**。
2. 每个下游任务用**少量标注样本（few-shot）**训练一个独立的下游头。
3. 框架同时支持：
   - 单月份像素级分割（如海淀建筑工地检测）。
   - 双时相变化检测（预留接口，待后续标注）。
4. 评测使用 **5-fold 交叉验证**，输出 IoU、F1、PR/AP 和可视化结果。
5. 所有路径、依赖、版本信息必须文档化，避免后续混乱。

## 关键决策

| 问题 | 决策 |
|---|---|
| 下游策略 | 冻结预训练 backbone，只训练轻量任务头 |
| 首个任务 | 海淀建筑工地像素级分割 |
| 输入 embedding | 像素级 `embedding_map`（分割）；场景级 `embedding` 也一并保存，供分类/变化检测使用 |
| 时序使用 | 只使用标注所在月份（2026-04）的 embedding |
| 数据划分 | 5-fold 交叉验证（每 fold 约 100 训练 / 13 验证 / 13 测试） |
| 生成时机 | 等阶段二训练完成后，用 `harbin_128_stage2_v1/best.pt` 生成 embedding |

## 目录与路径

### 代码目录

```
Downstreams/
├── downstreams/                    # Python 包
│   ├── __init__.py
│   ├── data/
│   │   ├── embedding_dataset.py    # 从预生成 embedding 加载数据
│   │   └── label_loaders.py        # labelme / GeoTIFF mask 读取
│   ├── heads/
│   │   ├── base.py                 # TaskHead 统一接口
│   │   ├── segmentation_head.py    # 1x1 conv / 轻量 U-Net 分割头
│   │   ├── classification_head.py  # 场景级分类头
│   │   └── change_detection_head.py# 双时相变化检测接口（预留）
│   ├── tasks/
│   │   ├── base.py                 # BaseTask：训练/评测流程抽象
│   │   └── construction_segmentation.py
│   ├── metrics/
│   │   ├── segmentation.py         # IoU、F1、Precision、Recall、AP
│   │   └── visualization.py        # mask 叠加、PR 曲线
│   └── utils/
│       └── fold_split.py           # 5-fold 划分
├── scripts/
│   ├── precompute_embeddings.py    # 离线生成 embedding
│   ├── convert_labelme.py          # labelme json -> mask tif
│   ├── train_task.py               # 训练 + 5-fold 评测入口
│   └── visualize_results.py        # 结果可视化
├── configs/
│   └── construction_segmentation.yaml
├── tests/
└── README.md                       # 路径、依赖、使用步骤总览
```

### 数据目录

```
/data/xuannv_embedding/
├── downstream/
│   └── labels/
│       └── haidian/
│           └── construction_site/
│               ├── masks/                    # 每张 patch 的 .tif mask
│               ├── labelme_raw/              # 解压后的原始 labelme json
│               └── split_5fold.json          # 5-fold 划分索引
│
└── embeddings/
    └── {YYYYMMDD}_{experiment_name}/         # 例如 20260621_harbin_128_stage2_v1
        ├── meta.json                         # 生成时间、模型路径、配置、stage
        ├── haidian/
        │   └── {patch_id}/
        │       ├── {YYYYMM}_embedding_map.pt     # (D, H, W)
        │       └── {YYYYMM}_scene_embedding.pt   # (D,)
        └── harbin/
            └── {patch_id}/
                ├── {YYYYMM}_embedding_map.pt
                └── {YYYYMM}_scene_embedding.pt
```

### 输出目录

```
/data/xuannv_embedding/outputs/downstream/
└── construction_site_haidian/
    ├── fold_0/
    │   ├── checkpoints/
    │   ├── metrics.json
    │   ├── predictions/              # 测试集预测 mask
    │   └── visualizations/           # 叠加图、PR 曲线
    ├── fold_1/...
    └── summary_5fold.json            # 5-fold 均值 ± 标准差
```

## 模块设计

### 1. 统一任务头接口

```python
class TaskHead(nn.Module):
    def forward(self, embedding_map: Tensor, scene_emb: Tensor | None) -> Tensor:
        """
        embedding_map: (B, D, H, W)
        scene_emb:     (B, D) 或 None
        return:        任务相关输出
        """
        ...
```

- `SegmentationHead`：输入 `(B, D, H, W)`，输出 `(B, num_classes, H, W)`。
  - 默认：2 层 1×1 conv + BN + ReLU。
  - 可选：轻量 2 层 U-Net decoder（skip connection 来自 embedding_map 自身）。
- `ClassificationHead`：输入 scene embedding `(B, D)`，输出 logits `(B, num_classes)`。
- `ChangeDetectionHead`：接收两个时相的 `embedding_map`，输出变化 mask `(B, 1, H, W)`；**当前仅定义接口和 dummy forward**。

### 2. 数据集

`EmbeddingDataset`：

- 读取 `split_5fold.json` 中指定 fold 的 patch id 列表。
- 对每个 patch，读取标注月份（如 `202604`）的：
  - `embedding_map.pt` -> `(D, H, W)`
  - `mask.tif` -> `(H, W)`，值为 `{0: 背景, 1: 建筑工地}`
- 返回字典：`{"embedding_map": Tensor, "mask": Tensor, "patch_id": str}`。
- 支持在线数据增强：随机水平/垂直翻转（embedding 与 mask 同步）。

### 3. 5-fold 划分

- 输入：126 张有标注 patch 的 id 列表。
- 输出：`split_5fold.json`，结构：

```json
{
  "seed": 42,
  "folds": [
    {"fold": 0, "train": [...], "val": [...], "test": [...]},
    ...
  ]
}
```

- 每层 fold 训练约 100 张、验证约 13 张、测试约 13 张。

### 4. 训练流程（每 fold）

- 优化器：AdamW，lr=1e-3，weight_decay=1e-2。
- Scheduler：cosine annealing。
- 损失：BCEWithLogitsLoss，正样本权重按工地像素比例动态计算。
- 早停：验证 mIoU 连续 10 个 epoch 不提升则停止。
- 每个 epoch 记录：train loss、val loss、val mIoU、val F1。

### 5. 评测指标

对每 fold 测试集计算：

- **mIoU**：工地类别的 IoU。
- **F1-score**：按最佳阈值（默认 0.5，可选 Otsu/PR 最大 F1）。
- **Precision / Recall**。
- **Average Precision (AP)**：按预测概率排序计算。

最终报告 5-fold 均值 ± 标准差。

### 6. 可视化

- `visualize_results.py`：
  - 把预测 mask、真值 mask、S2 RGB（如可用）叠加成 PNG。
  - 绘制 PR 曲线并保存。
  - 生成每 fold 和 5-fold 平均的指标表格（Markdown + JSON）。

## 脚本使用流程

```bash
# 1. 阶段二训练完成后，生成 embedding
python Downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --output-root /data/xuannv_embedding/embeddings

# 2. 转换 labelme 标注为 mask
python Downstreams/scripts/convert_labelme.py \
  --rar /data/xuannv_embedding/raw/haidian/labels/6511215751_/haidianlabel.rar \
  --patch-dir /data/xuannv_embedding/processed/haidian/patches/s2 \
  --out-dir /data/xuannv_embedding/downstream/labels/haidian/construction_site

# 3. 训练 + 5-fold 评测
python Downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config Downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1 \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian

# 4. 可视化
python Downstreams/scripts/visualize_results.py \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian
```

## 依赖

- `torch` / `torch_npu`（与主项目一致）
- `numpy`, `tqdm`
- `rasterio`（读写 .tif mask，主项目已用）
- `matplotlib`, `seaborn`（可视化）
- `scikit-learn`（5-fold 划分、指标）
- `scikit-image`（可选，mask 后处理）
- `pyyaml`

> 所有依赖写入 `Downstreams/pyproject.toml` 或在主项目 `pyproject.toml` 的 `optional-dependencies` 中新增 `downstream` 组。

## 版本与溯源

- `meta.json` 必须包含：
  - `created_at`
  - `checkpoint_path`
  - `config_path`
  - `model_class`
  - `embed_dim`, `num_months`, `patch_size`
- 评测结果 `metrics.json` 必须包含：
  - `embedding_meta_path`（指向对应 `meta.json`）
  - `task_name`, `fold`, `seed`, `head_config`

## 风险与回退

| 风险 | 回退方案 |
|---|---|
| 126 张样本太少，分割头过拟合 | 增加数据增强、使用更大 weight decay、改用线性探测（1x1 conv 单一层） |
| 工地像素占比极小导致 mIoU 为 0 | 使用 focal loss / dice loss 替代 BCE；指标同时看 AP |
| embedding 对工地不敏感 | 改用 scene embedding + 分类头；或阶段二结束后重新生成更优 embedding |
| 变化检测无标注 | 仅保留接口和 dummy 实现，不训练 |

## 待审查点

1. 冻结 backbone + 轻量头是否足够？是否需要部分微调？
2. 分割头用 1x1 conv 还是轻量 U-Net？哪种更适合 few-shot 遥感分割？
3. 行业最佳实践（SatMAE / Scale-MAE / DOFA / Clay 等）如何处理地理嵌入下游任务？
4. 路径与依赖文档是否足够清晰？是否有遗漏？
5. 变化检测接口设计是否合理？
