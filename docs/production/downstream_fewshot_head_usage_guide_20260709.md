# 玄女 Embedding 下游 Few-shot 检测头使用说明

本文档说明：如果已经有玄女 64 维 embedding，想针对某一类地物做少量标注快速制图，应该选什么下游头、怎么训练、代码在哪里。

## 一句话建议

做 few-shot 单类别检测时，默认先用 `mlp` 或 `binary_conv3x3`：

| 场景 | 推荐 head | 原因 |
| --- | --- | --- |
| 严格验证 embedding 是否线性可分 | `linear` | 最简单，几乎没有模型容量；适合做能力诊断 |
| 少量标注快速出图，且类别边界不太碎 | `mlp` | 只做逐像素通道组合，能读出 embedding 里的非线性语义 |
| 建筑、道路、水体、边界细碎类别 | `binary_conv3x3` | 加一点 3×3 局部空间上下文，实际可视化和 F1 通常更稳 |
| 想看 embedding 语义容量上限，但仍保持轻量 | `binary_wide_mlp` / `binary_deep_wide_mlp` | 比普通 MLP 容量更大，但不直接引入大 decoder |
| 强监督、标签很多、只追求最高分 | `unet` / `upernet` | 这是强下游模型，不适合作为 few-shot embedding 能力的主展示 |

我们的经验是：`linear` 用来证明“embedding 本身有没有把类别分开”；`mlp` 用来做公平 few-shot 对比；`binary_conv3x3` 用来做更好看的业务制图结果。

## Few-shot 是什么意思

这里的 `5-shot / 10-shot / 50-shot` 指：训练时只选取 5、10、50 个“含有目标类别正样本”的 patch，再配同等数量的负样本 patch。验证集和测试集不变，仍然在全局 patch 上评价。

例如做“操场检测”：

- `5-shot`：只人工标 5 个有操场的 patch，加 5 个没有操场的 patch 作为负样本。
- 训练一个很小的下游头。
- 用这个头去预测海淀全域 320 个 patch。

这样可以模拟真实业务：只标一点点样本，就快速铺开到整个区域。

## 两条训练路径

### 路径 A：公平 few-shot 评测，推荐给论文/汇报对比

使用脚本：

```bash
scripts/eval/train_aef_downstream_probe.py
```

这个脚本直接在像素 embedding 上采样训练，支持：

- `linear`
- `mlp`
- `mlp_deep`
- `5/10/20/50/full shot`
- 验证集自动选择最佳阈值
- NPU 缓存像素特征，加速训练

适合用来公平比较玄女、AEF 或其他 embedding，因为所有模型接同一个简单 probe。

示例：用 10-shot 训练“公园/绿地”检测头：

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python scripts/eval/train_aef_downstream_probe.py \
  --embedding-root /data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/embeddings/20260705_haidian_embedding_v1_p10c_epoch800_production_epoch_80_p10c_epoch800_capability \
  --label-root /data/xuannv_embedding/processed/haidian/labels/merged_park_green \
  --region haidian \
  --month 202604 \
  --output-root /data/xuannv_embedding/outputs/downstream_examples/park_green_mlp_10shot \
  --fold 0 \
  --device npu:0 \
  --head mlp \
  --epochs 50 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --hidden-dim 128 \
  --max-pixels-per-patch 4096 \
  --positive-fraction 0.5 \
  --train-positive-patches 10 \
  --eval-every 5 \
  --cache-device npu \
  --eval-cache-device npu \
  --save-predictions
```

输出：

- `metrics.json`：F1、AUC、AP、最佳阈值等指标。
- `predictions/`：每个测试 patch 的概率 GeoTIFF。
- `run.log`：训练日志。

### 路径 B：全图制图/业务可视化，推荐用 `binary_conv3x3`

使用脚本：

```bash
downstreams/scripts/train_task.py
```

这个脚本按 patch 读取 embedding map，训练一个分割 head，保存预测概率图。它支持更多 head，包括 `binary_conv3x3`。

推荐配置模板：

```yaml
experiment:
  name: v2_probe_binary_conv3x3_single_202604
  seed: 42
  device: auto

training:
  epochs: 50
  lr: 0.0003
  weight_decay: 0.0001
  batch_size: 32
  num_workers: 4
  prefetch_factor: 2
  use_early_stopping: false
  eval_every: 5
  save_predictions: true
  head_type: binary_conv3x3
  loss: bce_dice_tversky
  pos_weight: 1.0
  tversky_beta: 0.7
  months:
  - 202604
  temporal_mode: single
  early_stop_metric: f1_best
  use_threshold_tuning: true

data:
  embed_dim: 64
  num_classes: 1
```

仓库已有配置：

```bash
downstreams/configs/v2_probe_binary_conv3x3_single_202604.yaml
```

训练命令：

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config downstreams/configs/v2_probe_binary_conv3x3_single_202604.yaml \
  --embedding-root /data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/embeddings/20260705_haidian_embedding_v1_p10c_epoch800_production_epoch_80_p10c_epoch800_capability \
  --label-root /data/xuannv_embedding/processed/haidian/labels/building_osm \
  --region haidian \
  --output-root /data/xuannv_embedding/outputs/downstream_examples/building_conv3x3_full \
  --fold 0 \
  --save-predictions
```

## Head 代码在哪里

### 1. Linear head

文件：

```bash
downstreams/downstreams/heads/linear_probe.py
```

核心代码：

```python
class LinearProbeHead(TaskHead):
    """严格线性探测：单 1x1 conv，无激活/无 BN。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, embedding_map, scene_emb=None):
        return self.conv(embedding_map)
```

含义：每个像素的 64 维 embedding 直接线性映射成类别 logits。它不看邻域，只看这个像素自己的 embedding。

### 2. Pixel MLP head

文件：

```bash
downstreams/downstreams/heads/segmentation_head.py
```

核心代码：

```python
class MLPProbeHead(TaskHead):
    """逐像素 MLP 探针：只做 1x1 通道混合，不引入空间上下文。"""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x, scene_emb=None):
        return self.net(x)
```

含义：仍然不看邻域，只是在每个像素的 64 维 embedding 上做一个小非线性分类器。

### 3. Binary 3×3 Conv head

文件：

```bash
downstreams/downstreams/heads/segmentation_head.py
```

核心代码：

```python
class BinaryConv3x3ProbeHead(TaskHead):
    """二分类 3x3 Conv head：增加局部上下文，适合道路和边界类任务。"""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x, scene_emb=None):
        return self.net(x)
```

含义：每个像素不仅看自己的 embedding，也看周围 3×3 的局部邻域。建筑边缘、道路细线、水体边界这类任务通常更需要这种局部上下文。

## 如何选择 loss 和阈值

推荐默认：

```yaml
loss: bce_dice_tversky
pos_weight: 1.0
tversky_beta: 0.7
early_stop_metric: f1_best
use_threshold_tuning: true
```

原因：

- BCE 负责像素级二分类。
- Dice 让模型更关注前景区域整体重叠。
- Tversky 可以缓解前景很少时的漏检/误检问题。
- 不建议固定 0.5 阈值；应在验证集上扫阈值，选择最佳 F1 阈值，再固定到测试集或全域制图。

## 结果怎么看

常用指标：

- `AUC`：排序能力。AUC 高说明前景像素整体排在背景前面，但不保证阈值切出来好看。
- `AP`：前景稀疏时更有参考价值，反映 precision-recall 曲线下面积。
- `F1_best`：在验证集选择最佳阈值后，前景检测的综合结果。
- `threshold`：最终采用的概率阈值。不要默认认为 0.5 一定合理。

实际汇报时建议同时看：

1. 指标表：F1、AUC、AP。
2. 全域 320 patch 拼图。
3. 若干单 patch 叠加原始高分影像的可视化。

## 推荐实验矩阵

如果别人要快速验证一个新类别，建议按下面顺序跑：

| 步骤 | head | shot | 目的 |
| --- | --- | --- | --- |
| 1 | `linear` | 5, 10, 50 | 判断 embedding 是否线性可分 |
| 2 | `mlp` | 5, 10, 50 | 判断简单非线性 head 是否明显提升 |
| 3 | `binary_conv3x3` | 10 或 50 | 生成更稳的业务制图结果 |
| 4 | `unet` 或原始影像强基线 | full | 只作为强监督上限，不作为 few-shot 主结论 |

## 常见问题

### 1. 为什么不用 UNet 做 few-shot 主结果？

UNet 容量比较大，容易把下游模型能力和 embedding 能力混在一起。few-shot 的核心是证明“embedding 本身已经把语义压进去了”，所以主结果应该优先用 `linear/mlp/binary_conv3x3` 这类轻量头。

### 2. 为什么有时候 AUC 高，但可视化不好？

AUC 只看排序，不看固定阈值。很多情况下预测概率都挤在一个很窄的区间，例如 0.46 到 0.48，AUC 仍可能不错，但 0.5 阈值会切得很差。因此必须做验证集阈值搜索。

### 3. 如果假正例很多怎么办？

优先尝试：

- 从 `linear` 换到 `mlp`，确认是否是简单非线性不足。
- 从 `mlp` 换到 `binary_conv3x3`，引入局部空间一致性。
- 增加 shot 数，例如从 5-shot 提到 10-shot 或 50-shot。
- 检查标签质量，尤其是 OSM 弱标签是否把背景误标成前景。
- 看概率图而不是只看二值图，判断是阈值问题还是 embedding 真的分不开。

### 4. 如果类别非常细，比如操场、校园、公园，选哪个？

建议先用 `mlp` 做 10-shot/50-shot。如果目标形状有明显边界或局部纹理，再用 `binary_conv3x3` 做制图。不要一开始就上 UNet，否则很难解释到底是 embedding 好，还是下游模型自己学出来的。

## 最小复现 Checklist

- 已导出 embedding，目录形如：

```bash
/data/.../embeddings/<run_name>/haidian/<patch_id>/202604_embedding_map.pt
```

- 标签目录包含：

```bash
<label_root>/masks/*.tif
<label_root>/split_5fold.json
```

- 使用固定 split，不要每次随机重分。
- 训练、验证、测试都使用同一套 patch split。
- 阈值只在验证集选择，不能在测试集挑阈值。
- 对比 AEF 或其他模型时，下游头、shot、fold、epoch、采样像素数、阈值规则必须完全一致。

