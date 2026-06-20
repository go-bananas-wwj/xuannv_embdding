# 下游任务（Downstream）框架设计

> 日期：2026-06-20
> 状态：最终设计，待用户确认后进入实现
> 关联需求：`chats/chat6.txt`

## 目标

为 `xuannv_embedding` 预训练模型建立一个**轻量、可扩展、文档完整**的下游任务评测框架：

1. 阶段二训练完成后，离线生成并保存每个 patch、每个月的 `embedding_map` 与 `scene_embedding`。
2. 每个下游任务用**少量标注样本**训练独立的任务头，预训练 backbone **默认冻结**。
3. 首个落地任务：**海淀建筑工地像素级分割**（5-fold CV + 可选 10/25/50/100% 标签比例）。
4. 同时预留**双时相变化检测**接口，待后续标注接入。
5. 对齐行业惯例（GEO-Bench / PANGEA）：报告 mIoU、per-class IoU、F1、AP/AUPRC、对象级 F1，并与随机初始化基线对比。

## 关键决策

| 问题 | 决策 |
|---|---|
| 下游策略 | 默认冻结 backbone；任务头可选 1×1 linear probe、UperNet、UNet decoder |
| 首个任务 | 海淀建筑工地像素级分割 |
| embedding 层级 | 分割用 `embedding_map`；分类/变化检测用 `embedding_map` 或 scene embedding |
| 时序使用 | 只使用标注所在月份（2026-04）的 embedding |
| 数据划分 | 分层 5-fold CV（按 patch 正像素比例分层）；同时支持 10/25/50/100% 标签比例 |
| 损失函数 | 默认 Focal Loss + Dice Loss；BCE 加权作为消融对照 |
| 生成时机 | 等阶段二 `harbin_128_stage2_v1/best.pt` 完成后生成 |

## 目录与路径

### 代码目录

```
downstreams/                       # 与仓库其余目录风格一致，全小写
├── downstreams/                   # Python 包
│   ├── __init__.py
│   ├── data/
│   │   ├── embedding_dataset.py
│   │   ├── label_loaders.py
│   │   └── split.py               # 分层 5-fold + 有限标签比例划分
│   ├── heads/
│   │   ├── base.py                # TaskHead 统一接口
│   │   ├── linear_probe.py        # 1x1 conv：严格线性探测基线
│   │   ├── segmentation_head.py   # UperNet / UNet / FCN decoder
│   │   ├── classification_head.py
│   │   └── change_detection_head.py
│   ├── tasks/
│   │   ├── base.py
│   │   └── construction_segmentation.py
│   ├── metrics/
│   │   ├── segmentation.py
│   │   └── visualization.py
│   └── utils/
│       ├── device.py
│       └── reproducibility.py
├── scripts/
│   ├── prepare_downstream.py      # 解压 labelme -> 转 mask -> 生成 split
│   ├── precompute_embeddings.py
│   ├── train_task.py
│   └── visualize_results.py
├── configs/
│   ├── _base_.yaml
│   └── construction_segmentation.yaml
├── tests/
│   ├── test_dataset.py
│   ├── test_head_forward.py
│   └── test_metrics.py
└── README.md
```

### 数据目录

```
/data/xuannv_embedding/
├── downstream/
│   └── labels/
│       └── haidian/
│           └── construction_site/
│               ├── labelme_raw/              # 解压后的 labelme json
│               ├── masks/                    # {patch_id}.tif (uint8, 0=背景, 1=工地)
│               ├── split_5fold.json          # 分层 5-fold 索引
│               └── label_meta.json           # 标注来源、类别映射、栅格化参数
│
└── embeddings/
    └── {YYYYMMDD}_{experiment_name}_{short_sha}/   # 例：20260621_harbin_128_stage2_v1_a1b2c3d
        ├── meta.json
        ├── haidian/
        │   └── {patch_id}/
        │       ├── {YYYYMM}_embedding_map.pt      # (D, H, W), fp16/fp32
        │       └── {YYYYMM}_scene_embedding.pt    # (D,)
        └── harbin/
            └── ...
```

### 输出目录

```
/data/xuannv_embedding/outputs/downstream/
└── construction_site_haidian_{YYYYMMDD}_{embedding_suffix}/
    ├── fold_0/
    │   ├── checkpoints/best_fold_0.pt
    │   ├── metrics.json
    │   ├── predictions/{patch_id}_pred.tif
    │   └── visualizations/{patch_id}_overlay.png
    ├── fold_1/...
    ├── summary_5fold.json
    └── README.md
```

## 模块设计

### 1. TaskHead 统一接口

```python
class TaskHead(nn.Module):
    def forward(self, embedding_map: Tensor, scene_emb: Tensor | None = None) -> Tensor:
        """
        embedding_map: (B, D, H, W)
        scene_emb:     (B, D) or None
        """
```

- `LinearProbeHead`: 单 1×1 conv，无激活，用于严格线性探测。
- `SegmentationHead`: 支持 `head_type ∈ {"fcn", "unet", "upernet"}`。
  - `fcn`: 2 层 1×1 conv。
  - `unet`: 轻量 UNet decoder，从 embedding_map 自身做 skip。
  - `upernet`: 标准 UperNet decoder（优先实现，作为对标 PANGEA 的默认选项）。
- `ClassificationHead`: MLP on scene embedding。
- `ChangeDetectionHead`: 接收 `emb_map_t1` 与 `emb_map_t2`，输出变化 mask；当前仅定义接口与 dummy concat-forward。

### 2. 数据集

`EmbeddingDataset`：

- 读取 `split_5fold.json` 中指定 fold 与 `label_fraction` 的 patch id 列表。
- 对每张 patch 读取标注月份（默认 202604）的 `embedding_map.pt` 与 `masks/{patch_id}.tif`。
- 尺寸校验：`mask.shape == (H, W)`，且与 `embedding_map` 最后两维一致。
- 增强：同步随机水平/垂直翻转（90° 旋转可选，禁止任意角度旋转以避免插值破坏单位球面约束）。
- 对无数据/云掩膜区域设置 `ignore_index`，不进入损失和指标。

### 3. 5-fold 与有限标签划分

`create_split.py` / `downstreams.data.split`：

- 按 patch 级正像素比例分层，生成 5-fold。
- 每 fold：4 folds 训练（约 100）、1 fold 测试（约 25）；训练集内部再随机切 10% 做验证。
- 同时生成 10/25/50/100% 标签比例的子集（按正像素比例分层抽样），便于绘制数据效率曲线。
- 输出 `split_5fold.json`：

```json
{
  "seed": 42,
  "stratify_by": "positive_pixel_ratio",
  "folds": [
    {"fold": 0, "train": [...], "val": [...], "test": [...]},
    ...
  ],
  "fractions": {
    "0.1": {"fold_0": {"train": [...]}, ...},
    "0.25": {...},
    "0.5": {...}
  }
}
```

### 4. 训练流程（每 fold）

- 优化器：AdamW，lr=1e-4（UperNet/UNet）或 1e-3（linear probe），weight_decay=1e-2。
- Scheduler：cosine annealing。
- 损失：默认 `FocalLoss + DiceLoss`；配置可切换为 `BCEWithLogitsLoss(pos_weight=...)`。
- 早停：验证 mIoU 连续 10 个 epoch 不提升则停止；同时记录最佳 AP 作为备选。
- 设备：支持 `cpu` / `cuda` / `npu`，单卡即可。
- 可复现：固定 `torch/numpy/random` seed，记录到 `metrics.json`。

### 5. 评测指标

对每 fold 测试集：

- **mIoU**：工地类 IoU（ foreground IoU）。
- **per-class IoU**：背景、工地两类。
- **F1 / Precision / Recall**：0.5 阈值与 PR 曲线最大 F1 阈值。
- **AP / AUPRC**。
- **对象级 F1（可选）**：按连通域分析，适合小目标场景。

最终 `summary_5fold.json` 报告均值 ± 标准差。

### 6. 基线对比

- **Linear probe（1×1 conv）**：衡量 embedding 本身质量。
- **Random-init UNet decoder**：同架构但 backbone 随机初始化，证明预训练价值。
- **（可选）ImageNet-pretrained ViT**：若资源允许，作为外部基线。

### 7. 可视化

- 预测 mask、真值 mask、S2 RGB（若指定 `--rgb-source`）叠加。
- PR 曲线 PNG。
- 每 fold metrics 表格 + 5-fold summary Markdown。

## 脚本使用流程

```bash
# 0. 安装下游依赖
pip install -e ".[downstream]"

# 1. 准备下游数据：解压 labelme -> mask -> 5-fold 划分
python downstreams/scripts/prepare_downstream.py \
  --task construction_site \
  --region haidian \
  --labelme-rar /data/xuannv_embedding/raw/haidian/labels/6511215751_/haidianlabel.rar \
  --patch-dir /data/xuannv_embedding/processed/haidian/patches/s2 \
  --out-dir /data/xuannv_embedding/downstream/labels/haidian/construction_site

# 2. 阶段二完成后，预生成 embedding
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --regions haidian harbin \
  --output-root /data/xuannv_embedding/embeddings

# 3. 训练 + 5-fold 评测
python downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_a1b2c3d \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2

# 4. 可视化
python downstreams/scripts/visualize_results.py \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2 \
  --rgb-source /data/xuannv_embedding/processed/haidian/patches/s2
```

## 依赖

在主项目 `pyproject.toml` 新增 optional dependency 组：

```toml
[project.optional-dependencies]
downstream = [
  "matplotlib>=3.7",
  "seaborn>=0.12",
  "scikit-learn>=1.3",
  "scikit-image>=0.21",
  "shapely>=2.0",
  "pyyaml",
]
```

系统依赖：`bsdtar` 或 `unrar`（用于解压 labelme rar，若系统无则需安装）。

## 版本与溯源

### `meta.json`（每次生成 embedding 必须写入）

```json
{
  "created_at": "2026-06-21T08:00:00Z",
  "command_line": "python downstreams/scripts/precompute_embeddings.py ...",
  "git_commit": "a1b2c3d",
  "git_dirty": false,
  "python_version": "3.11.15",
  "torch_version": "2.x.x",
  "torch_npu_version": "...",
  "checkpoint_path": "/data/.../best.pt",
  "checkpoint_sha256": "...",
  "config_path": "configs/harbin_128_stage2.yaml",
  "model_class": "AEFModel",
  "embed_dim": 64,
  "num_months": 17,
  "patch_size": 128,
  "output_resolution_m": 10,
  "input_sources": ["s2", "s1", "landsat", "highres_optical_harbin"],
  "month_format": "YYYYMM"
}
```

### `metrics.json`（每 fold 必须写入）

```json
{
  "task": "construction_segmentation",
  "fold": 0,
  "embedding_meta": ".../meta.json",
  "seed": 42,
  "head_type": "upernet",
  "head_params": 120000,
  "optimizer": "AdamW",
  "lr": 1e-4,
  "loss": "focal+dice",
  "best_epoch": 23,
  "training_time_sec": 1800,
  "test_metrics": {
    "miou": 0.42,
    "per_class_iou": {"background": 0.98, "construction": 0.42},
    "f1_0.5": 0.58,
    "f1_best": 0.62,
    "precision": 0.55,
    "recall": 0.70,
    "ap": 0.48,
    "auprc": 0.35
  }
}
```

### `label_meta.json`

```json
{
  "task": "construction_site",
  "region": "haidian",
  "labelme_version": "6.3.1",
  "class_map": {"jiazhudongdi": 1},
  "crs": "EPSG:...",
  "resolution_m": 10,
  "num_patches": 126,
  "source_rar": "/data/.../haidianlabel.rar"
}
```

## 测试与冒烟

- `tests/test_dataset.py`：验证 `EmbeddingDataset` 能加载 embedding + mask，尺寸对齐。
- `tests/test_head_forward.py`：验证所有 head 前向输出 shape 正确。
- `tests/test_metrics.py`：验证极端不平衡情况下 mIoU/AP 计算正确。
- 提供一个 `smoke` 配置，用 2 fold、2 epoch 快速跑通端到端。

## 风险与回退

| 风险 | 回退方案 |
|---|---|
| 126 张样本过拟合 | 使用 linear probe 基线；增大 weight decay / dropout；减少 decoder 深度 |
| 工地像素极少先验 mIoU=0 | 默认使用 Focal+Dice；同时报告 AP/AUPRC / 对象级 F1 |
| embedding 对工地不敏感 | 与 random-init 基线对比；若差距小则阶段二后重新生成 |
| 5-fold 空间泄漏 | 检查 patch 地理坐标；必要时改用空间 block-out 划分 |
| 变化检测无标注 | 仅保留 `ChangeDetectionHead` 接口，不训练 |

## 与行业最佳实践的对齐

- 冻结 backbone + UperNet/UNet decoder 是 SatMAE、DOFA、SoftCon、Prithvi、PANGEA 的标准协议。
- 使用 1×1 linear probe 作为强基线，隔离 embedding 质量与 decoder 容量的影响。
- Focal + Dice 是遥感不平衡分割的常用组合。
- 5-fold CV + 10/25/50/100% 标签比例曲线，便于对标 GEO-Bench / PANGEA 的 limited-label 协议。
- 报告 per-class IoU、AP、AUPRC，符合领域评测规范。
