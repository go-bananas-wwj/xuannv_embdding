# 多任务下游分割、双时相融合与实验目录整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有单任务建筑工地分割扩展为多个独立二分类下游头，融入 2025 年 12 月与 2026 年 5 月双时相嵌入，联合海淀区与哈尔滨新区数据训练建筑区域任务，并按行业最佳实践重新整理实验目录。

**Architecture:** 在现有 `downstreams/` 框架下，引入 `task_name` 概念与 `MultiTaskEmbeddingDataset`，为每个类别生成独立的 mask 目录与 split 文件；新增支持双时相输入的 `BitemporalEmbeddingDataset` 与 `DiffUNetHead`；通过统一的实验目录命名规范与归档脚本清理旧输出，所有实验产物集中存放到 `/data/xuannv_embedding/experiments/`。

**Tech Stack:** PyTorch (`torch==2.1.0` / `torch_npu==2.1.0.post17`), rasterio, shapely, scikit-learn, YAML, NPU 6 卡 (`Ascend 910B`).

---

## 背景与现状（执行前必读）

- 哈尔滨 LabelMe 标注共 8 个类别：`construction site`、`building change`、`farm_change`、`river_building`、`river_farm`、`river_in`、`river_ship`、`rubbish`（旧名 `jiazhudongdi`、`gongdi` 等价于 `construction site`）。
- 当前 `scripts/process_harbin_patches.py` 只把 `construction site` 相关映射为前景，其余类别被忽略。
- 当前下游头只有 `ConstructionSegmentationTask`，所有类别混在一起用一个二分类头训练。
- 已有 embedding：哈尔滨 191 patch × 17 个月（202501–202605），海淀区 320 patch × 17 个月。
- 当前哈尔滨下游实验 mIoU≈0.03、F1≈0.06、AUC-ROC≈0.74，存在严重类别不平衡（正像素 ~1%）。
- `/data/xuannv_embedding/outputs/` 与 `/data/xuannv_embedding/embeddings/` 已显杂乱，调试目录、重复 shard、未命名实验混杂。

---

## 关键设计决策

1. **每个类别一个独立二分类头**。不共享最后一层，而是为每个任务保存独立的 `best.pt`、`metrics.json`、`predictions/`、`split_5fold.json`。原因：不同类别的空间分布、季节模式、正负比例差异大，共享头会导致梯度冲突。
2. **双时相作为输入通道拼接**（early fusion）。将 202512 与 202605 两个月的 `embedding_map` 在通道维度拼接（64+64=128 维），再送入分割头。这是遥感变化检测文献中验证最稳的基线（FC-EF / early-concat）。
3. **建筑区域任务联合训练海淀+哈尔滨**。统一 patch_id 前缀为 `haidian_xxx` / `harbin_xxx`，合并 embedding 目录与 mask 目录，生成跨区域的 `split_5fold.json`。
4. **类别不平衡处理**：保留并调优 `BceDiceTverskyLoss` + `pos_prior` 初始化 + `WeightedRandomSampler` + 验证集阈值搜索；对极不平衡任务额外引入 `FocalTverskyLoss` 作为可选损失。
5. **目录整理**：所有新实验统一放到 `/data/xuannv_embedding/experiments/`，按 `YYYYMMDD_<task>_<region>_<month(s)>_<head>_<notes>` 命名；旧目录通过脚本归档或删除。

---

## 文件结构总览

- 创建：`downstreams/downstreams/data/multi_task_dataset.py`
- 创建：`downstreams/downstreams/data/bitemporal_dataset.py`
- 创建：`downstreams/downstreams/heads/diff_head.py`
- 创建：`downstreams/downstreams/tasks/task_registry.py`
- 创建：`downstreams/downstreams/losses/segmentation_losses.py`
- 创建：`downstreams/configs/construction_site_bitemporal_harbin.yaml`
- 创建：`downstreams/configs/building_change_bitemporal_harbin.yaml`
- 创建：`downstreams/configs/building_area_bitemporal_joint.yaml`
- 创建：`scripts/reorganize_experiment_dirs.py`
- 创建：`docs/experiments/README.md`
- 修改：`scripts/process_harbin_patches.py`（支持多任务 class_map）
- 修改：`downstreams/downstreams/data/embedding_dataset.py`（兼容月份回退与 mask 目录参数）
- 修改：`downstreams/downstreams/tasks/construction_segmentation.py`（拆出可复用的损失与评估）
- 修改：`downstreams/scripts/train_task.py`（支持 task_name、bitemporal、joint-region）
- 修改：`downstreams/scripts/precompute_embeddings.py`（可选：为海淀区补生成 202512 与 202605 embedding，若缺失）

---

## Task 1: 梳理标注类别并生成多任务 mask

**目标：** 为哈尔滨 8 个类别分别生成独立 mask；为海淀区已有标注按类别拆分；统计每个任务的样本量与正像素比例。

**Files:**
- 修改：`scripts/process_harbin_patches.py`
- 创建：`downstreams/scripts/prepare_multitask_masks.py`
- 输出：`/data/xuannv_embedding/processed/harbin/labels/<task_name>/masks/`
- 输出：`/data/xuannv_embedding/processed/haidian/labels/<task_name>/masks/`（若海淀原始标注含多类别）

### Task 1.1: 统计哈尔滨 LabelMe 类别分布

- [ ] **Step 1: 运行统计脚本**

```bash
cd /root/workspace/xuannv
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

- [ ] **Step 2: 将统计结果保存到 `/data/xuannv_embedding/processed/harbin/labels/label_distribution.json`**

```bash
mkdir -p /data/xuannv_embedding/processed/harbin/labels
# 上一步输出重定向即可
```

### Task 1.2: 修改哈尔滨预处理脚本支持按任务输出

- [ ] **Step 3: 在 `scripts/process_harbin_patches.py` 末尾新增 `--task` 与 `--class-map` 参数**

```python
# 添加到 argparse
p.add_argument("--task", type=str, default="construction",
               help="输出任务名，决定 labels/<task>/masks 子目录")
p.add_argument("--class-map", type=str, default=None,
               help='JSON 字符串，例：\'{"construction site":1,"jiazhudongdi":1}\'')

# 在 main() 中解析并传入
class_map = None
if args.class_map:
    class_map = json.loads(args.class_map)
output_root = args.output_root / args.task
process_harbin(args.raw_root, output_root, tuple(args.mask_size), class_map)
```

- [ ] **Step 4: 将 `process_harbin()` 中的 mask_dir 改为包含 task 层级**

```python
mask_dir = output_root / "masks"  # 原 output_root/labels/construction/masks -> output_root/masks
labelme_dir = output_root / "labelme_raw"
```

> 说明：这样 `output_root` 本身即为 `processed/harbin/labels/<task>`，保持与海淀目录结构一致。

- [ ] **Step 5: 为每个任务运行预处理**

```bash
cd /root/workspace/xuannv
RAW=/data/xuannv_embedding/Data-raw/haerbin_label_2026
OUT=/data/xuannv_embedding/processed/harbin/labels

# construction site（建筑工地）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/construction \
  --class-map '{"construction site":1,"jiazhudongdi":1,"gongdi":1}'

# building change（建筑变化/新增建筑）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/building_change \
  --class-map '{"building change":1}'

# farm_change（农田变化）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/farm_change \
  --class-map '{"farm_change":1}'

# river_building（临河建筑）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/river_building \
  --class-map '{"river_building":1}'

# river_farm（临河农田）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/river_farm \
  --class-map '{"river_farm":1}'

# river_in（河流入口/水域侵入）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/river_in \
  --class-map '{"river_in":1}'

# river_ship（船只）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/river_ship \
  --class-map '{"river_ship":1}'

# rubbish（垃圾/堆土）
python scripts/process_harbin_patches.py \
  --raw-root $RAW \
  --output-root $OUT/rubbish \
  --class-map '{"rubbish":1}'
```

- [ ] **Step 6: 验证每个任务目录下存在 202512 与 202605 两个月份的 mask**

```bash
for task in construction building_change farm_change river_building river_farm river_in river_ship rubbish; do
  echo "=== $task ==="
  ls /data/xuannv_embedding/processed/harbin/labels/$task/masks/*_202512.tif 2>/dev/null | wc -l
  ls /data/xuannv_embedding/processed/harbin/labels/$task/masks/*_202605.tif 2>/dev/null | wc -l
done
```

- [ ] **Step 7: 提交代码与脚本**

```bash
git add scripts/process_harbin_patches.py
git commit -m "feat(process_harbin): support per-task class_map output"
```

---

## Task 2: 实现多任务 EmbeddingDataset 与数据流水线

**目标：** 让下游训练脚本按 `task_name` 自动找到对应 mask 目录；支持单时相 fallback 与显式双时相。

**Files:**
- 创建：`downstreams/downstreams/data/multi_task_dataset.py`
- 修改：`downstreams/downstreams/data/embedding_dataset.py`

### Task 2.1: 创建 MultiTaskEmbeddingDataset

- [ ] **Step 1: 编写 `downstreams/downstreams/data/multi_task_dataset.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
import torch
from torch.utils.data import Dataset


class MultiTaskEmbeddingDataset(Dataset):
    """支持多任务、多月份、多区域的 embedding 数据集。"""

    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        task_name: str = "construction",
        months: list[str] | str = "202605",
        augment: bool = False,
        bitemporal: bool = False,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.task_name = task_name
        if isinstance(months, str):
            months = [months]
        self.months = [str(m) for m in months]
        self.augment = augment
        self.bitemporal = bitemporal
        self.mask_dir = self.label_root / "masks"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _load_embedding(self, patch_id: str, month: str) -> torch.Tensor:
        emb_path = self.embedding_root / patch_id / f"{month}_embedding_map.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"embedding 不存在: {emb_path}")
        return torch.load(emb_path, map_location="cpu", weights_only=True)

    def _load_mask(self, patch_id: str) -> torch.Tensor:
        # 优先精确匹配月份，否则回退到 {patch_id}.tif
        candidates = [self.mask_dir / f"{patch_id}_{m}.tif" for m in self.months]
        candidates.append(self.mask_dir / f"{patch_id}.tif")
        for p in candidates:
            if p.exists():
                with rasterio.open(p) as src:
                    return torch.from_numpy(src.read(1)).long()
        raise FileNotFoundError(f"mask 不存在 for {patch_id}, months={self.months}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        if self.bitemporal:
            if len(self.months) != 2:
                raise ValueError("bitemporal 需要恰好两个月份")
            emb_t1 = self._load_embedding(patch_id, self.months[0])
            emb_t2 = self._load_embedding(patch_id, self.months[1])
            emb = torch.cat([emb_t1, emb_t2], dim=0)
        else:
            emb = self._load_embedding(patch_id, self.months[0])
        mask = self._load_mask(patch_id)

        if emb.shape[-2:] != mask.shape:
            raise ValueError(
                f"{patch_id} embedding {emb.shape[-2:]} 与 mask {mask.shape} 尺寸不一致"
            )

        if self.augment:
            emb, mask = self._apply_augment(emb, mask)

        return {
            "embedding_map": emb,
            "mask": mask,
            "patch_id": patch_id,
        }

    def _apply_augment(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return emb, mask


def collate_embeddings(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embedding_map": torch.stack([b["embedding_map"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_ids": [b["patch_id"] for b in batch],
    }
```

- [ ] **Step 2: 提交新数据集**

```bash
git add downstreams/downstreams/data/multi_task_dataset.py
git commit -m "feat(dataset): add MultiTaskEmbeddingDataset with bitemporal support"
```

---

## Task 3: 实现双时相融合头 DiffUNetHead

**目标：** 当输入为双月份拼接（128 维）时，使用轻量 UNet 解码器；同时保留单时相能力。

**Files:**
- 创建：`downstreams/downstreams/heads/diff_head.py`
- 修改：`downstreams/downstreams/heads/segmentation_head.py`

### Task 3.1: 实现双时相 UNet 头

- [ ] **Step 1: 编写 `downstreams/downstreams/heads/diff_head.py`**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from downstreams.heads.base import TaskHead
from downstreams.heads.segmentation_head import _init_foreground_bias


class DiffUNetHead(TaskHead):
    """双时相 early-fusion UNet 头。

    输入通道 = 2 * embed_dim（两个月 embedding 拼接）。
    可选：在编码器中加入跨时相差分分支，强化变化特征。
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        use_diff_branch: bool = True,
        pos_prior: float | None = None,
    ) -> None:
        super().__init__()
        in_ch = 2 * embed_dim
        self.use_diff_branch = use_diff_branch

        # 编码
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        bottleneck_ch = embed_dim // 2
        if use_diff_branch:
            # 差分分支输出与编码特征拼接
            self.diff_conv = nn.Sequential(
                nn.Conv2d(in_ch, embed_dim // 4, 1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.ReLU(inplace=True),
            )
            bottleneck_ch += embed_dim // 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_ch, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        # 解码
        self.up1 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(embed_dim // 2 + embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(embed_dim // 4 + embed_dim, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(embed_dim // 4, num_classes, 1)
        _init_foreground_bias(self.final, pos_prior)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 2D, H, W)
        e1 = self.enc1(x)           # (B, D, H, W)
        p1 = self.pool1(e1)         # (B, D, H/2, W/2)
        e2 = self.enc2(p1)          # (B, D/2, H/2, W/2)
        p2 = self.pool2(e2)         # (B, D/2, H/4, W/4)

        feats = [p2]
        if self.use_diff_branch:
            diff = self.diff_conv(x)
            diff_p = F.adaptive_avg_pool2d(diff, p2.shape[-2:])
            feats.append(diff_p)
        bottle = self.bottleneck(torch.cat(feats, dim=1))

        d1 = self.up1(bottle)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        # 上采样 e1 到与 d2 同尺寸后拼接
        e1_up = F.interpolate(e1, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1_up], dim=1))
        out = self.final(d2)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
```

- [ ] **Step 2: 在 `downstreams/downstreams/heads/segmentation_head.py` 注册新头**

```python
from downstreams.heads.diff_head import DiffUNetHead

# 在 build_segmentation_head 中加入
if head_type == "diff_unet":
    return DiffUNetHead(embed_dim, num_classes, pos_prior=pos_prior)
```

- [ ] **Step 3: 提交新头**

```bash
git add downstreams/downstreams/heads/diff_head.py downstreams/downstreams/heads/segmentation_head.py
git commit -m "feat(head): add DiffUNetHead for bitemporal early fusion"
```

---

## Task 4: 改造训练脚本支持 task_name、双时相与联合区域

**目标：** `train_task.py` 通过 `--task` 选择 mask 子目录，通过 `--months` 控制单/双时相，通过 `--regions` 支持多区域联合。

**Files:**
- 修改：`downstreams/scripts/train_task.py`
- 修改：`downstreams/downstreams/data/split.py`（支持跨区域生成 split）

### Task 4.1: 扩展 split.py 支持跨区域与按任务生成

- [ ] **Step 1: 新增 `create_combined_stratified_folds`**

```python
def create_combined_stratified_folds(
    mask_dirs: dict[str, Path],
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """合并多个区域的 mask 目录生成统一 stratified fold。

    mask_dirs: {region: mask_dir}
    patch_id 会加上 region 前缀避免冲突。
    """
    all_patch_ids = []
    all_ratios = []
    region_of = {}
    for region, mask_dir in mask_dirs.items():
        paths = sorted(mask_dir.glob("*.tif"))
        for p in paths:
            pid = f"{region}_{p.stem}"
            all_patch_ids.append(pid)
            all_ratios.append(_positive_ratio(p))
            region_of[pid] = region

    ratios = np.array(all_ratios)
    pid_to_idx = {pid: i for i, pid in enumerate(all_patch_ids)}
    is_positive = ratios > 0
    ratio_strata = _quantize(ratios, n_bins=3)
    strata = is_positive.astype(int) * 10 + ratio_strata
    _, counts = np.unique(strata, return_counts=True)
    use_stratified = counts.min() >= n_folds

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) \
        if use_stratified else KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        splitter.split(all_patch_ids, strata if use_stratified else None)
    ):
        train_val_ids = [all_patch_ids[i] for i in train_val_idx]
        test_ids = [all_patch_ids[i] for i in test_idx]

        rng = np.random.default_rng(seed + fold_idx)
        n_val = max(1, int(len(train_val_ids) * val_ratio))
        pos_in_train_val = [pid for pid in train_val_ids if is_positive[pid_to_idx[pid]]]
        if pos_in_train_val:
            val_ids = rng.choice(pos_in_train_val, size=min(n_val, len(pos_in_train_val)), replace=False).tolist()
        else:
            val_ids = rng.choice(train_val_ids, size=n_val, replace=False).tolist()
        train_ids = [pid for pid in train_val_ids if pid not in val_ids]

        if not any(is_positive[pid_to_idx[pid]] for pid in test_ids):
            pos_train = [pid for pid in train_ids if is_positive[pid_to_idx[pid]]]
            if pos_train:
                swap_out = test_ids[0]
                swap_in = rng.choice(pos_train)
                test_ids = [swap_in if pid == swap_out else pid for pid in test_ids]
                train_ids = [swap_out if pid == swap_in else pid for pid in train_ids]

        folds.append({"fold": fold_idx, "train": train_ids, "val": val_ids, "test": test_ids})

    # fractions 省略：与 create_stratified_folds 相同逻辑
    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_ratio": val_ratio,
        "stratify_by": "positive_pixel_ratio_combined",
        "folds": folds,
        "fractions": {},  # 按原逻辑填充
        "region_of": region_of,
    }
```

### Task 4.2: 扩展 train_task.py

- [ ] **Step 2: 新增 CLI 参数**

```python
p.add_argument("--task", type=str, default="construction", help="任务名，决定 label_root 下的子目录")
p.add_argument("--months", type=str, nargs="+", default=["202605"], help="使用月份，双月份时启用双时相")
p.add_argument("--regions", type=str, nargs="+", default=None, help="联合训练区域，如 haidian harbin")
p.add_argument("--head-type", type=str, default=None, help="覆盖配置中的 head_type")
```

- [ ] **Step 3: 根据参数构造 mask_dir / embedding_root / patch_ids**

```python
label_root = args.label_root / args.task
mask_dir = label_root / "masks"

bitemporal = len(args.months) == 2

if args.regions:
    region = "joint"
    emb_region_root = args.embedding_root  # expects subdirs haidian/ harbin/
    split_path = label_root / "split_joint_5fold.json"
    mask_dirs = {
        r: args.label_root.parent / r / "labels" / args.task / "masks"
        for r in args.regions
    }
    if not split_path.exists():
        split = create_combined_stratified_folds(mask_dirs, seed=cfg["experiment"]["seed"])
        # ...
else:
    region = args.region if args.region else args.label_root.parent.name
    emb_region_root = args.embedding_root / region
    split_path = label_root / "split_5fold.json"
    # ...
```

- [ ] **Step 4: 实例化 `MultiTaskEmbeddingDataset` 替代 `EmbeddingDataset`**

```python
from downstreams.data.multi_task_dataset import MultiTaskEmbeddingDataset, collate_embeddings

train_ds = MultiTaskEmbeddingDataset(
    emb_region_root, label_root, train_ids,
    task_name=args.task, months=args.months, augment=True, bitemporal=bitemporal,
)
```

- [ ] **Step 5: 保存 checkpoint 时包含 task 与 months 信息**

```python
save_info = {
    "task": args.task,
    "months": args.months,
    "regions": args.regions or [region],
    "head_type": cfg["training"]["head_type"],
    "best_threshold": best_threshold,
}
torch.save({"state_dict": best_state, "info": save_info}, out_dir / "checkpoints" / "best.pt")
```

- [ ] **Step 6: 提交训练脚本改造**

```bash
git add downstreams/scripts/train_task.py downstreams/downstreams/data/split.py
git commit -m "feat(train): support task_name, bitemporal, and joint-region training"
```

---

## Task 5: 为关键任务创建 YAML 配置

**目标：** 为建筑工地、建筑变化、建筑区域（joint）分别建立最佳实践配置。

**Files:**
- 创建：`downstreams/configs/construction_site_bitemporal_harbin.yaml`
- 创建：`downstreams/configs/building_change_bitemporal_harbin.yaml`
- 创建：`downstreams/configs/building_area_bitemporal_joint.yaml`

### Task 5.1: 建筑工地双时相配置

- [ ] **Step 1: 编写配置**

```yaml
_base_: _base_.yaml

experiment:
  name: construction_site_harbin_bitemporal_v1

training:
  head_type: diff_unet
  loss: bce_dice_tversky
  lr: 3.0e-5
  pos_weight: 250.0
  tversky_beta: 0.7
  bce_weight: 1.0
  dice_weight: 1.0
  tversky_weight: 1.5
  early_stop_metric: f1_best
  early_stop_patience: 20
  use_weighted_sampler: true
  sampler_min_weight: 0.0001
  sampler_oversample_factor: 1.0
  use_threshold_tuning: true
  warmup_epochs: 5
  scheduler: cosine
  grad_clip: 1.0
  pos_prior: 0.015
  batch_size: 8
  months: [202512, 202605]
```

### Task 5.2: 建筑变化双时相配置

- [ ] **Step 2: 编写配置**

```yaml
_base_: _base_.yaml

experiment:
  name: building_change_harbin_bitemporal_v1

training:
  head_type: diff_unet
  loss: bce_dice_tversky
  lr: 3.0e-5
  pos_weight: 300.0
  tversky_beta: 0.8
  tversky_weight: 2.0
  early_stop_metric: f1_best
  early_stop_patience: 20
  use_weighted_sampler: true
  use_threshold_tuning: true
  pos_prior: 0.005
  batch_size: 8
  months: [202512, 202605]
```

### Task 5.3: 建筑区域联合训练配置

- [ ] **Step 3: 编写配置**

```yaml
_base_: _base_.yaml

experiment:
  name: building_area_joint_bitemporal_v1

training:
  head_type: diff_unet
  loss: bce_dice_tversky
  lr: 3.0e-5
  pos_weight: 200.0
  tversky_beta: 0.7
  tversky_weight: 1.5
  early_stop_metric: f1_best
  early_stop_patience: 20
  use_weighted_sampler: true
  use_threshold_tuning: true
  pos_prior: 0.02
  batch_size: 8
  months: [202512, 202605]
```

- [ ] **Step 4: 提交配置**

```bash
git add downstreams/configs/construction_site_bitemporal_harbin.yaml \
  downstreams/configs/building_change_bitemporal_harbin.yaml \
  downstreams/configs/building_area_bitemporal_joint.yaml
git commit -m "feat(configs): add bitemporal configs for construction, change, and joint building area"
```

---

## Task 6: 实现 FocalTverskyLoss 与类别不平衡最佳实践

**目标：** 对极不平衡任务提供更强的不平衡损失选项。

**Files:**
- 创建：`downstreams/downstreams/losses/segmentation_losses.py`
- 修改：`downstreams/downstreams/tasks/construction_segmentation.py`

### Task 6.1: 实现 FocalTverskyLoss

- [ ] **Step 1: 编写 `downstreams/downstreams/losses/segmentation_losses.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss：对极不平衡分割更稳定。"""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky) ** self.gamma
```

- [ ] **Step 2: 在 `construction_segmentation.py` 中注册 `focal_tversky`**

```python
from downstreams.losses.segmentation_losses import FocalTverskyLoss

elif loss_name == "focal_tversky":
    self._loss = FocalTverskyLoss(
        alpha=training.get("tversky_alpha", 0.3),
        beta=training.get("tversky_beta", 0.7),
        gamma=training.get("focal_gamma", 1.33),
    )
```

- [ ] **Step 3: 提交损失扩展**

```bash
git add downstreams/downstreams/losses/segmentation_losses.py downstreams/downstreams/tasks/construction_segmentation.py
git commit -m "feat(loss): add FocalTverskyLoss for extreme class imbalance"
```

---

## Task 7: 整理实验目录

**目标：** 将现有杂乱输出迁移到统一目录结构，删除可确认的调试目录，建立 README 与映射表。

**Files:**
- 创建：`scripts/reorganize_experiment_dirs.py`
- 创建：`docs/experiments/README.md`

### Task 7.1: 设计新目录规范

- [ ] **Step 1: 在 `docs/experiments/README.md` 中定义规范**

```markdown
# 实验目录规范

所有实验产物统一放在 `/data/xuannv_embedding/experiments/`。

## 目录命名

```
experiments/
  <YYYYMMDD>_<task>_<region>_<months>_<head>_<version>/
    config.yaml
    split_5fold.json
    fold_0/
      checkpoints/best.pt
      metrics.json
      predictions/
    summary_5fold.json
    logs/
```

- `task`: construction_site, building_change, building_area, farm_change, ...
- `region`: harbin, haidian, joint
- `months`: 202605, 202512_202605
- `head`: unet, diff_unet, upernet, linear
- `version`: v1, v2, ...
```

### Task 7.2: 编写整理脚本

- [ ] **Step 2: 编写 `scripts/reorganize_experiment_dirs.py`**

```python
#!/usr/bin/env python3
"""整理 /data/xuannv_embedding/outputs/ 与 embeddings/ 中的调试目录。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SAFE_DELETE_PATTERNS = [
    "smoke_*",
    "test_trainer_*",
    "debug_*",
    "*shard*",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/data/xuannv_embedding"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    outputs = args.data_root / "outputs"
    for sub in ["downstream", ""]:
        root = outputs / sub if sub else outputs
        if not root.exists():
            continue
        for p in root.iterdir():
            if any(p.match(pat) for pat in SAFE_DELETE_PATTERNS):
                print(f"{'[DRY-RUN] ' if args.dry_run else ''}DELETE {p}")
                if not args.dry_run:
                    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 先 dry-run 再实际执行**

```bash
cd /root/workspace/xuannv
python scripts/reorganize_experiment_dirs.py --dry-run
# 人工检查列表无误后
python scripts/reorganize_experiment_dirs.py
```

- [ ] **Step 4: 创建新实验根目录**

```bash
mkdir -p /data/xuannv_embedding/experiments
ln -s /data/xuannv_embedding/experiments /root/workspace/xuannv/experiments
```

- [ ] **Step 5: 提交整理脚本与文档**

```bash
git add scripts/reorganize_experiment_dirs.py docs/experiments/README.md
git commit -m "chore(dirs): add experiment directory reorganization script and convention"
```

---

## Task 8: 运行关键实验并生成对比报告

**目标：** 跑通建筑工地、建筑变化、建筑区域联合三个核心实验；汇总指标与可视化。

### Task 8.1: 建筑工地双时相实验

- [ ] **Step 1: 运行单 fold 验证**

```bash
cd /root/workspace/xuannv
python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_site_bitemporal_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \
  --label-root /data/xuannv_embedding/processed/harbin/labels \
  --task construction \
  --months 202512 202605 \
  --output-root /data/xuannv_embedding/experiments/20260617_construction_site_harbin_202512_202605_diff_unet_v1 \
  --fold 0
```

- [ ] **Step 2: 若 fold 0 指标优于单时相基线，运行完整 5-fold**

```bash
python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_site_bitemporal_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \
  --label-root /data/xuannv_embedding/processed/harbin/labels \
  --task construction \
  --months 202512 202605 \
  --output-root /data/xuannv_embedding/experiments/20260617_construction_site_harbin_202512_202605_diff_unet_v1
```

### Task 8.2: 建筑变化双时相实验

- [ ] **Step 3: 运行 5-fold**

```bash
python downstreams/scripts/train_task.py \
  --config downstreams/configs/building_change_bitemporal_harbin.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \
  --label-root /data/xuannv_embedding/processed/harbin/labels \
  --task building_change \
  --months 202512 202605 \
  --output-root /data/xuannv_embedding/experiments/20260617_building_change_harbin_202512_202605_diff_unet_v1
```

### Task 8.3: 建筑区域联合训练

- [ ] **Step 4: 确认海淀区 202512 与 202605 embedding 已存在**

```bash
ls /data/xuannv_embedding/embeddings/20260620_haidian_128_stage2_v1_best_stage2_v1/haidian/patch_000001/ 2>/dev/null | head
```

- [ ] **Step 5: 若缺失则补生成**

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/haidian_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --regions haidian \
  --output-root /data/xuannv_embedding/embeddings \
  --batch-size 1
```

- [ ] **Step 6: 运行联合 5-fold**

```bash
python downstreams/scripts/train_task.py \
  --config downstreams/configs/building_area_bitemporal_joint.yaml \
  --embedding-root /data/xuannv_embedding/embeddings \
  --label-root /data/xuannv_embedding/processed \
  --task construction \
  --regions haidian harbin \
  --months 202512 202605 \
  --output-root /data/xuannv_embedding/experiments/20260617_building_area_joint_202512_202605_diff_unet_v1
```

### Task 8.4: 结果汇总

- [ ] **Step 7: 编写汇总脚本并运行**

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/data/xuannv_embedding/experiments')
rows = []
for exp in root.glob('20260617_*'):
    summary = exp / 'summary_5fold.json'
    if not summary.exists():
        continue
    with open(summary) as f:
        data = json.load(f)
    miou = sum(d['miou'] for d in data) / len(data)
    f1 = sum(d['f1_best'] for d in data) / len(data)
    auc = sum(d.get('auc_roc', 0) for d in data) / len(data)
    rows.append(f"{exp.name}: mIoU={miou:.4f} F1={f1:.4f} AUC={auc:.4f}")
print('\n'.join(sorted(rows)))
PY
```

- [ ] **Step 8: 提交实验配置与结果摘要**

```bash
git add docs/experiments/results_20260617.md
git commit -m "docs: add experiment results summary for bitemporal downstream tasks"
```

---

## Task 9: 偏差与幻觉自检清单

在最终交付前，必须人工/脚本核对以下项目：

- [ ] **Step 1: 核对标注类别无遗漏**

```bash
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
print(json.dumps(dict(labels), ensure_ascii=False, indent=2))
PY
```

- [ ] **Step 2: 核对每个任务的 mask 与 embedding 月份对齐**

```bash
for task in construction building_change farm_change; do
  python downstreams/scripts/verify_alignment.py \
    --task $task \
    --label-root /data/xuannv_embedding/processed/harbin/labels \
    --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \
    --months 202512 202605
done
```

> 若 `verify_alignment.py` 不存在，则新增该脚本检查每个 patch 在指定月份是否存在 embedding 与 mask。

- [ ] **Step 3: 核对哈尔滨四象限布局假设**

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('/data/xuannv_embedding/Data-raw/haerbin_label_2026')
for p in list(root.rglob('*.json'))[:5]:
    with open(p) as f:
        data = json.load(f)
    print(p, data.get('imageWidth'), data.get('imageHeight'))
PY
```

> 若 01/02/03/04 的实际布局不是 2×2 TL/TR/BL/BR，需修正 `_QUADRANT_ORDER`。

- [ ] **Step 4: 检查下游指标是否存在数据泄漏**

```bash
python - <<'PY'
import json
from pathlib import Path
for split in Path('/data/xuannv_embedding/experiments').glob('*/split_5fold.json'):
    with open(split) as f:
        data = json.load(f)
    for fold in data['folds']:
        train = set(fold['train'])
        val = set(fold['val'])
        test = set(fold['test'])
        assert not (train & val), f"train/val overlap in {split}"
        assert not (train & test), f"train/test overlap in {split}"
        assert not (val & test), f"val/test overlap in {split}"
    print(f"OK: {split}")
PY
```

---

## Spec Coverage 自查

| chat8.txt 需求 | 对应 Task |
|---|---|
| 总结标注类别，不要用一个分类头 | Task 1 |
| 不同下游任务用单独二分类头；2025年12月加入训练 | Task 1, 2, 4, 5 |
| 建筑区域任务海淀+哈尔滨一起训练 | Task 4, 5.3, 8.3 |
| 分别使用 202512 与 202605 两期嵌入 | Task 2, 3, 5 |
| 整理 Auto 部署目录 | Task 7 |
| 调研最佳实践、检查偏差/幻觉 | Task 6, 9 |

---

## 执行方式

**Plan complete and saved to `docs/superpowers/plans/2026-06-17-multitask-downstream-reorganization.md` (will be copied after approval). Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Required sub-skill: `superpowers:subagent-driven-development`.
2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review.

**Which approach?**
