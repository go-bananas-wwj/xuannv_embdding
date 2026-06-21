# 多任务下游分割、多下游头对比、双时相融合与实验目录整理实施计划 v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已有 `downstreams/` 框架下，为建筑工地、建筑变化、农业用地变化、垃圾检测、海淀+哈尔滨建筑工地联合这 5 个下游任务建立独立二分类头；系统对比 linear / MLP / FCN / UNet / UperNet / DiffUNet 等下游头在少样本、稀疏标注、类别不平衡下的效果；融入 202512 与 202605 双时相嵌入；支持 6 卡 NPU 并行训练；并按最佳实践整理实验目录。

**Architecture:** 新增 `MultiTaskEmbeddingDataset` 支持多任务、双时相与联合区域；扩展 `segmentation_head.py` 实现多种候选下游头并统一注册；改造 `train_task.py` 支持 `--task`、`--months`、`--regions`、`--head-type` 以及 `torchrun` 6 卡 DDP；通过 5-fold 交叉验证对每个任务-头组合进行指标对比，最终选择最佳头。

**Tech Stack:** PyTorch (`torch==2.1.0` / `torch_npu==2.1.0.post17`), `torchrun`, rasterio, shapely, scikit-learn, YAML, NPU 6 卡 (`Ascend 910B`).

---

## 背景与现状

- 哈尔滨 LabelMe 标注含 8 个类别，但本次只聚焦其中 4 个独立任务：
  - `construction site`（施工工地）
  - `building change`（建筑变化）
  - `farm_change`（农业用地变化）
  - `rubbish`（垃圾/堆土）
- 第 5 个任务是**施工工地跨域联合检测**：海淀区 + 哈尔滨新区一起训练。
- 当前下游头只有 `ConstructionSegmentationTask`，且仅用一个 UNet 头训练，未做头结构对比。
- 当前哈尔滨下游实验 mIoU≈0.03、F1≈0.06、AUC-ROC≈0.74，存在严重类别不平衡（正像素 ~1%）。
- 训练脚本目前单卡运行，未利用 6 卡 NPU。

---

## 调研结论与关键设计决策

### 下游头选择（基于最新论文/博客）

| 下游头 | 特点 | 适用场景 |
|---|---|---|
| **Linear Probe** | 1×1 卷积，参数量最小 | 特征与任务语义高度对齐、标注极少时不易过拟合 |
| **MLP Head** | 1-2 层全连接/1×1 卷积 + BN + ReLU | Few-shot 分割中表现稳健，实现简单，适合中等稀疏标注 |
| **FCN Head** | 两层 3×3 卷积 | 在 GeoFM 下游分割中常用，表达能力适中 |
| **UNet Head** | 轻量解码器 + skip | 需要恢复空间细节时使用 |
| **UperNet Head** | PSP pooling + fusion | 需要多尺度上下文时使用 |
| **DiffUNet Head** | 双时相 early-fusion + 差分分支 | 变化检测任务首选，遥感 CD 文献验证有效 |
| **LoRA/Adapter** | 微调 backbone | 当 frozen embedding 效果不足时考虑，实现较重 |

**决策：**
1. 第一阶段 frozen embedding，对所有任务对比 **linear / MLP / FCN / UNet / UperNet / DiffUNet** 6 种头。
2. 每个任务跑单 fold 快速筛选出 top-2，再跑完整 5-fold 确定最佳头。
3. 若 frozen embedding 头效果仍不足，第二阶段再引入 LoRA/Adapter 微调 backbone。

### 双时相融合策略

- **Early fusion（通道拼接）** 是变化检测最稳健的基线（FC-EF）。
- **拼接 + 差分图** 在多项遥感 CD 工作中被证明能稳定提升 F1（DGMA2-Net、MFPF-Net）。
- 因此 `DiffUNetHead` 实现为：输入 `[emb_t1, emb_t2, |emb_t1-emb_t2|]` 或 `[emb_t1, emb_t2]` 加差分分支。

### 少样本 / 稀疏标注 / 类别不平衡 Tricks

- **WeightedRandomSampler**：按正像素比例加权采样，确保每个 batch 都有正样本。
- **pos_prior 初始化**：根据训练集前景比例初始化输出层 bias，加速收敛。
- **复合损失**：`BCE + Dice + Tversky` 组合，Tversky 的 `β>α` 更重视 recall。
- **FocalTverskyLoss**：对极不平衡任务进一步聚焦难例。
- **验证集阈值搜索**：不用固定 0.5，而是在验证集上搜最优 F1 阈值。
- **早停指标**：用 `f1_best` 或 `ap` 而非 loss。
- **数据增强**：仅做同步水平/垂直翻转，避免改变 embedding 分布。
- **学习率调度**：cosine + warmup。
- **梯度裁剪**：防止 NPU 上 loss 爆炸。

### 6 卡并行

- 训练脚本改造为支持 `torchrun --nproc_per_node=6` 的 DDP。
- 由于 NPU 多进程 CANN 算子编译缓存冲突，每个进程需设置独立 `ACL_OP_COMPILER_CACHE_DIR`（通过 `local_rank` 区分）。
- 每个 GPU 的 batch size 可设为 4-8，总 effective batch size = 单卡 bs × 6。

---

## 文件结构总览

- 创建：`downstreams/downstreams/data/multi_task_dataset.py`
- 创建：`downstreams/downstreams/heads/mlp_head.py`
- 创建：`downstreams/downstreams/heads/diff_head.py`
- 创建：`downstreams/downstreams/losses/segmentation_losses.py`
- 创建：`downstreams/downstreams/utils/ddp.py`
- 创建：`downstreams/scripts/benchmark_heads.py`
- 创建：`downstreams/configs/<task>_bitemporal_harbin.yaml`（4 个）
- 创建：`downstreams/configs/construction_joint_bitemporal.yaml`
- 创建：`scripts/reorganize_experiment_dirs.py`
- 创建：`docs/experiments/README.md`
- 修改：`scripts/process_harbin_patches.py`（支持 `--task`、`--class-map`）
- 修改：`downstreams/downstreams/heads/segmentation_head.py`（注册 mlp / diff_unet）
- 修改：`downstreams/downstreams/tasks/construction_segmentation.py`（添加 focal_tversky）
- 修改：`downstreams/scripts/train_task.py`（支持 task/months/regions/head_type + DDP）
- 修改：`downstreams/downstreams/data/split.py`（支持跨区域 split）

---

## Task 1: 梳理选定类别并生成多任务 mask

**目标：** 为 4 个哈尔滨任务 + 1 个联合任务生成 mask；统计每个任务的样本量与正像素比例。

**Files:**
- 修改：`scripts/process_harbin_patches.py`
- 输出：`/data/xuannv_embedding/processed/harbin/labels/{construction,building_change,farm_change,rubbish}/masks/`
- 输出：`/data/xuannv_embedding/processed/harbin/labels/label_distribution.json`

### Step 1: 统计哈尔滨 LabelMe 类别分布

```bash
cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
python - <<'PY'
import json
from pathlib import Path
from collections import Counter
root = Path('/data/xuannv_embedding/Data-raw/haerbin_label_2026')
labels = Counter()
for p in root.rglob('*.json'):
    with open(p) as f:
        data = json.load(f)
    for s in data.get('shapes', []):
        labels[s.get('label','').strip().lower()] += 1
print(json.dumps(dict(labels.most_common()), ensure_ascii=False, indent=2))
PY
```

### Step 2: 修改 `scripts/process_harbin_patches.py`

新增：
```python
p.add_argument("--task", type=str, default="construction")
p.add_argument("--class-map", type=str, default=None)
```

`process_harbin()` 输出布局改为：
```python
mask_dir = output_root / "masks"
labelme_dir = output_root / "labelme_raw"
```

### Step 3: 生成 4 个任务的 mask

```bash
RAW=/data/xuannv_embedding/Data-raw/haerbin_label_2026
OUT=/data/xuannv_embedding/processed/harbin/labels

python scripts/process_harbin_patches.py --raw-root $RAW --output-root $OUT/construction \
  --class-map '{"construction site":1,"jiazhudongdi":1,"gongdi":1}'
python scripts/process_harbin_patches.py --raw-root $RAW --output-root $OUT/building_change \
  --class-map '{"building change":1}'
python scripts/process_harbin_patches.py --raw-root $RAW --output-root $OUT/farm_change \
  --class-map '{"farm_change":1}'
python scripts/process_harbin_patches.py --raw-root $RAW --output-root $OUT/rubbish \
  --class-map '{"rubbish":1}'
```

### Step 4: 验证

```bash
for task in construction building_change farm_change rubbish; do
  echo "=== $task ==="
  ls $OUT/$task/masks/*_202512.tif 2>/dev/null | wc -l
  ls $OUT/$task/masks/*_202605.tif 2>/dev/null | wc -l
done
```

---

## Task 2: 实现多任务/双时相/联合区域数据集

**目标：** 让下游训练脚本按 `--task` 找到 mask，按 `--months` 单/双时相，按 `--regions` 联合区域。

**Files:**
- 创建：`downstreams/downstreams/data/multi_task_dataset.py`

### Step 1: 实现 `MultiTaskEmbeddingDataset`

```python
class MultiTaskEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        task_name: str = "construction",
        months: list[str] | str = "202605",
        augment: bool = False,
        bitemporal: bool = False,
        include_diff: bool = True,
    ) -> None:
        ...

    def __getitem__(self, idx):
        if bitemporal:
            emb_t1 = self._load_embedding(pid, months[0])
            emb_t2 = self._load_embedding(pid, months[1])
            emb = torch.cat([emb_t1, emb_t2], dim=0)
            if include_diff:
                emb = torch.cat([emb, (emb_t1 - emb_t2).abs()], dim=0)
        ...
```

---

## Task 3: 扩展下游头库（linear / MLP / FCN / UNet / UperNet / DiffUNet）

**目标：** 实现并注册所有候选头，支持双时相输入通道自适应。

**Files:**
- 创建：`downstreams/downstreams/heads/mlp_head.py`
- 创建：`downstreams/downstreams/heads/diff_head.py`
- 修改：`downstreams/downstreams/heads/segmentation_head.py`

### Step 1: MLP Head

```python
class MLPHead(TaskHead):
    def __init__(self, embed_dim, num_classes, hidden_dim=256, pos_prior=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        _init_foreground_bias(self.net[-1], pos_prior)

    def forward(self, x, scene_emb=None):
        return self.net(x)
```

### Step 2: DiffUNet Head

输入通道自动适配：
```python
in_ch = x.shape[1]  # 支持 64/128/192（拼接+差分）
```

### Step 3: 注册

```python
if head_type == "mlp":
    return MLPHead(embed_dim, num_classes, pos_prior=pos_prior)
if head_type == "diff_unet":
    return DiffUNetHead(embed_dim, num_classes, pos_prior=pos_prior)
```

---

## Task 4: 改造训练脚本支持 task/months/regions/head-type 与 6 卡 DDP

**目标：** `train_task.py` 支持新参数，并能在 6 卡 NPU 上并行训练。

**Files:**
- 修改：`downstreams/scripts/train_task.py`
- 创建：`downstreams/downstreams/utils/ddp.py`
- 修改：`downstreams/downstreams/data/split.py`

### Step 1: DDP 工具

```python
# downstreams/downstreams/utils/ddp.py
import os
import torch
import torch.distributed as dist


def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["ACL_OP_COMPILER_CACHE_DIR"] = f"/tmp/acl_op_cache_{local_rank}"
    if world_size > 1:
        dist.init_process_group(backend="hccl")
    torch.npu.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
```

### Step 2: 训练脚本改造要点

- 新增 `--task`、`--months`、`--regions`、`--head-type` 参数。
- 用 `DistributedSampler` 替换 shuffle。
- 模型包装 `DistributedDataParallel`。
- 只在 rank 0 保存 checkpoint 和写 summary。
- 指标汇总时 all-reduce 各 rank 的 logits/masks 或各自独立计算后取平均。

### Step 3: 跨区域 split

```python
def create_combined_stratified_folds(mask_dirs: dict[str, Path], ...):
    # mask_dirs = {"haidian": ..., "harbin": ...}
    # patch_id 加前缀避免冲突
```

---

## Task 5: 实现 FocalTverskyLoss 与损失调优

**Files:**
- 创建：`downstreams/downstreams/losses/segmentation_losses.py`
- 修改：`downstreams/downstreams/tasks/construction_segmentation.py`

### Step 1: FocalTverskyLoss

```python
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0):
        ...
```

### Step 2: 注册

在 `ConstructionSegmentationTask.build_loss` 中加入 `focal_tversky` 分支。

---

## Task 6: 下游头对比实验（Benchmark Heads）

**目标：** 对 5 个任务各跑 6 种头，快速筛选最佳头。

**Files:**
- 创建：`downstreams/scripts/benchmark_heads.py`
- 输出：`/data/xuannv_embedding/experiments/head_benchmark_<task>.json`

### Step 1: 编写 benchmark 脚本

```bash
python downstreams/scripts/benchmark_heads.py \
  --task construction \
  --regions harbin \
  --months 202512 202605 \
  --heads linear mlp fcn unet upernet diff_unet \
  --fold 0 \
  --epochs 30 \
  --output-root /data/xuannv_embedding/experiments/head_benchmark_construction
```

### Step 2: 对每个任务运行

```bash
for task in construction building_change farm_change rubbish; do
  python downstreams/scripts/benchmark_heads.py \
    --task $task --regions harbin --months 202512 202605 \
    --heads linear mlp fcn unet upernet diff_unet \
    --fold 0 --epochs 30 \
    --output-root /data/xuannv_embedding/experiments/head_benchmark_$task
done
```

### Step 3: 联合任务单独跑

```bash
python downstreams/scripts/benchmark_heads.py \
  --task construction --regions haidian harbin --months 202512 202605 \
  --heads linear mlp fcn unet upernet diff_unet \
  --fold 0 --epochs 30 \
  --output-root /data/xuannv_embedding/experiments/head_benchmark_construction_joint
```

---

## Task 7: 为每个任务创建基于最佳头的 YAML 配置

**目标：** 根据 benchmark 结果，为每个任务创建完整 5-fold 配置。

**Files:**
- 创建：`downstreams/configs/construction_site_<best_head>_harbin.yaml`
- 创建：`downstreams/configs/building_change_<best_head>_harbin.yaml`
- 创建：`downstreams/configs/farm_change_<best_head>_harbin.yaml`
- 创建：`downstreams/configs/rubbish_<best_head>_harbin.yaml`
- 创建：`downstreams/configs/construction_site_<best_head>_joint.yaml`

每个配置包含：
- `head_type`
- `loss`（默认 `bce_dice_tversky`，对极不平衡任务可切 `focal_tversky`）
- `pos_weight`、`tversky_beta`、`pos_prior`
- `months: [202512, 202605]`
- `batch_size`、`lr`、`scheduler`、`warmup_epochs`、`grad_clip`
- `use_weighted_sampler: true`、`use_threshold_tuning: true`

---

## Task 8: 完整 5-fold 训练与 6 卡并行

**目标：** 用最佳头对每个任务跑完整 5-fold，利用 6 卡 NPU 加速。

### Step 1: 单任务 5-fold（以 construction 为例）

```bash
torchrun --nproc_per_node=6 downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_site_<best_head>_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \
  --label-root /data/xuannv_embedding/processed/harbin/labels \
  --task construction --months 202512 202605 \
  --output-root /data/xuannv_embedding/experiments/20260617_construction_site_harbin_202512_202605_<best_head>_v1
```

### Step 2: 其余任务

```bash
# building_change
python downstreams/scripts/train_task.py --config ... --task building_change ...

# farm_change
python downstreams/scripts/train_task.py --config ... --task farm_change ...

# rubbish
python downstreams/scripts/train_task.py --config ... --task rubbish ...

# joint construction (haidian + harbin)
python downstreams/scripts/train_task.py --config ... --task construction --regions haidian harbin ...
```

> 是否用 `torchrun` 取决于任务数据量。每个任务单独决定是否启用 6 卡。

---

## Task 9: 整理实验目录与结果汇总

**Files:**
- 创建：`scripts/reorganize_experiment_dirs.py`
- 创建：`docs/experiments/README.md`
- 创建：`docs/experiments/results_20260617.md`

### Step 1: 目录规范

所有新实验统一放到 `/data/xuannv_embedding/experiments/`，命名：
```
<YYYYMMDD>_<task>_<region>_<months>_<head>_<version>/
```

### Step 2: 清理旧目录

```bash
python scripts/reorganize_experiment_dirs.py --dry-run
python scripts/reorganize_experiment_dirs.py
```

### Step 3: 结果汇总

```bash
python downstreams/scripts/summarize_experiments.py \
  --root /data/xuannv_embedding/experiments \
  --output docs/experiments/results_20260617.md
```

---

## Task 10: 偏差与幻觉自检

- [ ] 核对 4 个任务的 mask 与 embedding 在 202512/202605 都对齐。
- [ ] 检查联合任务 split 中无 train/val/test 泄漏。
- [ ] 核对哈尔滨四象限布局假设（01-TL/02-TR/03-BL/04-BR）。
- [ ] 确认 6 卡 DDP 每个进程使用独立 ACL cache。
- [ ] 对比单卡与 6 卡结果一致性（指标差异 < 1%）。

---

## Spec Coverage 自查

| 用户需求 | 对应 Task |
|---|---|
| 只做建筑工地、建筑变化、农业用地变化、垃圾检测 | Task 1, 6, 7, 8 |
| 海淀+哈尔滨施工工地合并训练 | Task 4, 6, 7, 8 |
| 调研并对比不同下游头 | Task 3, 6 |
| 少样本/稀疏标注最佳实践 | 设计决策、Task 5, 7 |
| 6 卡并行 | Task 4, 8 |
| 整理实验目录 | Task 9 |

---

## 执行方式

**Plan complete and saved to `docs/superpowers/plans/2026-06-17-multitask-downstream-reorganization-v2.md`.**

1. **Subagent-Driven (recommended)** — 每个 Task 分派独立子代理，顺序实现、中间 review。
2. **Inline Execution** — 当前会话中批量执行，关键节点暂停检查。

**Which approach?**
