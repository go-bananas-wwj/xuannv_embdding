# V2.0 多任务下游分割改进路线图实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 AEF benchmark 结论，通过重新设计 embedding 预训练和/或增强下游分割头，使 5 任务 5-fold 平均 AUC>0.8 且 F1_best>0.6。

**Architecture:** 双轨并行：轨 A 参照 AEF 论文与社区最佳实践扩大预训练数据、调整模型容量与训练目标，生成 V2.0 embedding；轨 B 在固定 embedding 上引入更强分割头、稀疏标注损失、数据增强与端到端微调。两者分别进行小批量 ablation，验证有效后合并为最终 V2.0 方案。

**Tech Stack:** PyTorch, NPU, rasterio, timm, segmentation-models-pytorch (optional), xarray, zarr

---

## Phase 0：分析 AEF Benchmark 结论

### Task 1: 对比分析并确定 V2.0 优先级

**Files:**
- Read: `/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md`
- Create: `/data/xuannv_embedding/experiments/v2.0/v2.0_decision.md`

- [ ] **Step 1: 计算 AEF 相对 V1.0 的胜率**

Run:
```bash
python -c "$(cat <<'EOF'
import json
v1 = json.load(open('/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json'))
aef = json.load(open('/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md'))  # parse table if needed
# Placeholder: actual parsing depends on report format
print('Inspect AEF_BENCHMARK_REPORT.md manually to decide priority.')
EOF
)"
```

- [ ] **Step 2: 撰写 V2.0 决策文档**

Create: `/data/xuannv_embedding/experiments/v2.0/v2.0_decision.md`

```markdown
# V2.0 Priority Decision

## AEF vs V1.0 Summary

| Task | V1.0 AUC | AEF AUC | V1.0 F1 | AEF F1 | Winner |
|---|---|---|---|---|---|
| construction | TBD | TBD | TBD | TBD | TBD |
| building_change | TBD | TBD | TBD | TBD | TBD |
| farm_change | TBD | TBD | TBD | TBD | TBD |
| rubbish | TBD | TBD | TBD | TBD | TBD |
| construction_joint | TBD | TBD | TBD | TBD | TBD |

## Decision

- If AEF wins on >= 3 tasks: proceed with both Track A and Track B.
- If AEF wins on < 3 tasks: prioritize Track B (downstream enhancement).
```

---

## Track A：Embedding 预训练重新设计

### Task 2: 调研 AEF 训练最佳实践

**Files:**
- Read: `docs/superpowers/specs/2026-06-20-final-training-spec.md`
- Read: AEF paper sections on data, architecture, losses
- Create: `/data/xuannv_embedding/experiments/v2.0/embedding_retrain/best_practices_review.md`

- [ ] **Step 1: 系统梳理 AEF 关键成功因素**

Research topics:
1. Data quantity: AEF used >3B observations, ~1.1% land area.
2. Data sources: Sentinel-2, Landsat, Sentinel-1, PALSAR2, GEDI, GLO-30, ERA5, GRACE, NLCD, Wikipedia.
3. Architecture: Space Time Precision (STP) encoder with spatial/time/precision operators.
4. Bottleneck: vMF (von Mises-Fisher) distribution on unit hypersphere S^63.
5. Training: teacher-student + text alignment + batch uniformity.
6. Output: 64-dim quantized annual embeddings.

Use web search and read current AEF paper if needed.

- [ ] **Step 2: 列出 V2.0 可改进点**

Create: `/data/xuannv_embedding/experiments/v2.0/embedding_retrain/improvement_list.md`

```markdown
# V2.0 Embedding Improvements

1. Increase pretraining data volume and diversity.
2. Verify STP encoder implementation matches paper.
3. Add batch uniformity loss.
4. Add text-EO contrastive objective (if text data available).
5. Train with continuous time encoding.
6. Reduce embedding dim to 64 and quantize to int8 for fair AEF comparison.
7. Increase model capacity (layers, hidden dim).
```

---

### Task 3: 扩大预训练数据集

**Files:**
- Modify: `src/xuannv_embedding/data/builder.py`
- Modify: `configs/` pretraining configs

- [ ] **Step 1: 评估当前预训练数据覆盖**

Run:
```bash
python scripts/data/audit_processed.py --region harbin --region haidian
python scripts/data/audit_processed.py --region global_samples  # if exists
```

- [ ] **Step 2: 增加新的训练区域**

Identify additional regions/cities with similar characteristics (urban, agricultural, seasonal change). Download and process:
- Beijing other districts
- Shanghai
- Guangzhou
- Other Tier-2/3 cities

Use existing `scripts/data/download_pc.py` and `scripts/process_harbin_patches.py` patterns.

- [ ] **Step 3: 增加数据源**

Current project already uses S2/S1/Landsat/高分光学/高分 SAR/WorldCover. Consider adding:
- ERA5-Land climate variables
- GLO-30 DEM
- GEDI canopy height
- NLCD land cover (for North America samples)

Create data loader additions in `src/xuannv_embedding/data/dataset.py`.

---

### Task 4: 检查并完善 AEFModel 架构

**Files:**
- Read: `src/xuannv_embedding/models/model.py`
- Read: `src/xuannv_embedding/models/bottleneck.py`
- Read: `src/xuannv_embedding/models/decoders.py`
- Modify if gaps found

- [ ] **Step 1: 对照 AEF 论文逐层检查**

Checklist:
- Space Time Precision encoder has 3 operators (spatial 1/16L, temporal 1/8L, precision 1/2L).
- vMF bottleneck with L2 normalization.
- Teacher-student shared parameters.
- Implicit decoders conditioned on time/sensor metadata.
- Batch uniformity loss.

- [ ] **Step 2: 补充缺失模块**

If missing:
- Add temporal self-attention operator in `src/xuannv_embedding/models/blocks.py`.
- Add batch uniformity loss in `src/xuannv_embedding/training/losses.py`.
- Add text projection head if text data available.

---

### Task 5: 训练 V2.0 Embedding（小批量 ablation）

**Files:**
- Create: `/data/xuannv_embedding/experiments/v2.0/embedding_retrain/ablations/`
- Create: configs in `configs/v2.0_ablation_*.yaml`

- [ ] **Step 1: 设计 ablation 实验**

Create: `/data/xuannv_embedding/experiments/v2.0/embedding_retrain/ablation_plan.md`

```markdown
# V2.0 Embedding Ablation Plan

1. Baseline: current AEFModel on current data.
2. + batch uniformity loss.
3. + 2x data volume.
4. + continuous time encoding.
5. + 64-dim + int8 quantization.
6. Full V2.0: all improvements combined.
```

- [ ] **Step 2: 训练并评估每个 ablation**

Use existing pretraining script:
```bash
python src/xuannv_embedding/training/train.py --config configs/v2.0_ablation_1.yaml
```

Each ablation should produce a checkpoint and be evaluated on a small downstream construction task (1-fold) to rank them.

---

### Task 6: 选择最佳 V2.0 Embedding 并全量预训练

**Files:**
- Create: `/data/xuannv_embedding/experiments/v2.0/embedding_retrain/final_config.yaml`
- Output: `/data/xuannv_embedding/embeddings/2026MMDD_<region>_v2.0_best/`

- [ ] **Step 1: 选择最佳 ablation**

Based on 1-fold construction downstream AUC/F1, select the best config.

- [ ] **Step 2: 全量预训练**

Run:
```bash
python src/xuannv_embedding/training/train.py --config /data/xuannv_embedding/experiments/v2.0/embedding_retrain/final_config.yaml
```

Expected: checkpoint saved under `/data/xuannv_embedding/outputs/`.

- [ ] **Step 3: 预生成 V2.0 embedding**

Run:
```bash
python downstreams/scripts/precompute_embeddings.py \
  --checkpoint /data/xuannv_embedding/outputs/v2.0_final/best.pt \
  --region harbin \
  --out-root /data/xuannv_embedding/embeddings/20260620_harbin_v2.0_best
```

Repeat for haidian.

---

## Track B：下游分割头增强

### Task 7: 引入更强分割头

**Files:**
- Modify: `downstreams/downstreams/heads/segmentation_head.py`
- Create: `downstreams/configs/v2.0_<task>_deeplabv3plus.yaml`

- [ ] **Step 1: 实现 DeepLabV3+ head**

Add to `downstreams/downstreams/heads/segmentation_head.py`:

```python
class DeepLabV3PlusHead(TaskHead):
    def __init__(self, in_channels, num_classes, channels=256):
        super().__init__()
        # ASPP + decoder; implement or wrap segmentation-models-pytorch
        from segmentation_models_pytorch import DeepLabV3Plus
        self.head = DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
        # Replace encoder with identity since input is already embedding
        self.head.encoder = torch.nn.Identity()

    def forward(self, x):
        return self.head(x)
```

- [ ] **Step 2: 添加 SegFormer head（可选）**

Add `SegFormerHead` similarly.

- [ ] **Step 3: 1-fold ablation 对比 head 效果**

Run construction task with each head for 1 fold, compare AUC/F1.

---

### Task 8: 实现稀疏标注专用损失

**Files:**
- Modify: `downstreams/downstreams/tasks/construction_segmentation.py`
- Create: `downstreams/downstreams/losses/lovasz_loss.py`

- [ ] **Step 1: 实现 Lovász loss**

Create: `downstreams/downstreams/losses/lovasz_loss.py`

```python
import torch
import torch.nn as nn

# Reference: https://github.com/bermanmaxim/LovaszSoftmax

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.flip(0).cumsum(0).flip(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        loss = torch.stack([_lovasz_hinge_flat(_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0)))
                            for log, lab in zip(logits, labels)])
    else:
        loss = _lovasz_hinge_flat(_flatten_binary_scores(logits, labels))
    return loss.mean()


def _flatten_binary_scores(scores, labels):
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels


def _lovasz_hinge_flat(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss
```

- [ ] **Step 2: 实现 OHEM BCE loss**

Add OHEM to existing loss combinations.

- [ ] **Step 3: 1-fold ablation 对比损失效果**

Run construction with FocalDice vs Lovasz+Focal vs OHEM+BCE.

---

### Task 9: 端到端微调尝试

**Files:**
- Modify: `downstreams/scripts/train_task.py` to optionally load encoder and unfreeze
- Create: `downstreams/configs/v2.0_<task>_end2end.yaml`

- [ ] **Step 1: 支持 encoder 可训练**

In `downstreams/scripts/train_task.py`, add flag `--unfreeze-encoder` and set `param.requires_grad = True` for encoder parameters.

- [ ] **Step 2: 使用更小 crop / 更大显存策略**

Set `crop_size: 320`, `batch_size: 2`, `accumulation_steps: 4`.

- [ ] **Step 3: 运行 1-fold 端到端 ablation**

Run construction end-to-end for 1 fold and compare to frozen embedding baseline.

---

## Phase 3：整合与全量 5-fold 验证

### Task 10: 选择最佳 V2.0 配置

**Files:**
- Create: `/data/xuannv_embedding/experiments/v2.0/downstream_enhance/final_config_decision.md`

- [ ] **Step 1: 汇总 ablation 结果**

Table format:
| Variant | Embedding | Head | Loss | End2end | 1-fold AUC | 1-fold F1 |
|---|---|---|---|---|---|---|
| baseline | V1.0 | UperNet | FocalDice | No | TBD | TBD |
| A | V2.0 | UperNet | FocalDice | No | TBD | TBD |
| B | V1.0 | DeepLabV3+ | Lovasz | No | TBD | TBD |
| C | V2.0 | DeepLabV3+ | Lovasz | Yes | TBD | TBD |

- [ ] **Step 2: 选择最终配置**

Pick the combination with highest 1-fold F1 on construction.

---

### Task 11: 全量 5-fold V2.0 下游训练

**Files:**
- Use: `downstreams/scripts/train_task.py`
- Output: `/data/xuannv_embedding/experiments/v2.0/<task>_v2.0_5fold/`

- [ ] **Step 1: 创建 V2.0 下游 configs**

Create 5 configs under `/data/xuannv_embedding/experiments/v2.0/downstream_enhance/configs/`.

- [ ] **Step 2: 批量训练 5 任务 5-fold**

Create: `/root/workspace/xuannv/downstreams/scripts/run_v2.0_benchmark.py`

```python
import subprocess
from pathlib import Path

configs = list(Path("/data/xuannv_embedding/experiments/v2.0/downstream_enhance/configs").glob("*.yaml"))
for cfg in configs:
    subprocess.run(
        ["python", "downstreams/scripts/train_task.py", "--config", str(cfg)],
        cwd="/root/workspace/xuannv/.worktrees/feat-multitask-downstream",
        check=True,
    )
```

Run:
```bash
python downstreams/scripts/run_v2.0_benchmark.py
```

- [ ] **Step 3: 汇总 V2.0 指标**

Create: `/root/workspace/xuannv/downstreams/scripts/summarize_v2.0.py`

Similar to V1.0 summary, output `/data/xuannv_embedding/experiments/v2.0/v2.0_summary.json`.

---

### Task 12: V2.0 与 V1.0 / AEF 对比报告

**Files:**
- Create: `/data/xuannv_embedding/experiments/v2.0/V2.0_BENCHMARK_REPORT.md`
- Create: `/root/workspace/xuannv/downstreams/scripts/plot_v2.0_comparison.py`

- [ ] **Step 1: 生成三向对比表**

| Task | V1.0 AUC | AEF AUC | V2.0 AUC | V1.0 F1 | AEF F1 | V2.0 F1 |
|---|---|---|---|---|---|---|

- [ ] **Step 2: 判断是否达到目标**

If all 5 tasks have AUC>0.8 and F1>0.6, declare success. Otherwise document remaining gaps and next steps.

---

### Self-Review Checklist

- [ ] Spec coverage: Track A (embedding), Track B (downstream), integration, evaluation all covered.
- [ ] Placeholder scan: no TBD/TODO in executable sections; only in decision template that will be filled after AEF results.
- [ ] Type consistency: config paths and output paths consistent with V1.0 and AEF benchmark plans.
- [ ] Feasibility: ablations use 1-fold to reduce cost before full 5-fold.
