# 建筑工地分割下游模型架构诊断报告（v2）

> 分析对象：stage-2 Harbin 预训练 AEFModel → 低分辨率 embedding → Haidian 建筑工地二分类分割  
> 结果版本：v2（label rasterization 修复后）  
> 报告日期：2026-06-20

## 1. 实验与数据现状

| 项目 | 现状 |
|---|---|
| 下游训练脚本 | `downstreams/scripts/train_task.py` |
| 下游任务代码 | `downstreams/downstreams/tasks/construction_segmentation.py` |
| 下游模型代码 | `downstreams/downstreams/heads/segmentation_head.py` |
| 数据加载 | `downstreams/downstreams/data/embedding_dataset.py` |
| 配置文件 | `downstreams/configs/construction_segmentation.yaml`（继承 `_base_.yaml`）|
| 当前 head | `upernet` |
| 当前 loss | `focal_dice` |
| Embedding 路径 | `/data/xuannv_embedding/embeddings/20260620_haidian_128_stage2_v1_best_stage2_v1/haidian` |
| Embedding 实际 shape | `(64, 128, 128)`，即 **64 维、128×128 空间分辨率**，与 mask 尺寸一致 |
| Embedding 特性 | 经 vMF bottleneck L2 归一化到单位球面，全局均值≈0，std≈0.125 |
| 使用月份 | 仅 `202604`（`month=202604`） |
| 数据集规模 | 62 个 patch，其中 51 个包含前景；平均前景占比 **约 4.0%**（中位数 2.7%，最大 21.2%）|

v2 5-fold 汇总结果（完整标签）：

| 标签比例 | mIoU | F1@0.5 | best-F1 | Precision | Recall |
|---|---|---|---|---|---|
| 1.0 | 0.094 ± 0.042 | 0.169 ± 0.069 | 0.310 ± 0.120 | 0.106 ± 0.043 | 0.537 ± 0.305 |
| 0.5 | 0.063 ± 0.053 | 0.114 ± 0.091 | 0.278 ± 0.087 | 0.070 ± 0.060 | 0.596 ± 0.386 |
| 0.25 | 0.035 ± 0.029 | 0.065 ± 0.054 | 0.222 ± 0.057 | 0.072 ± 0.067 | 0.238 ± 0.383 |
| 0.1 | 0.080 ± 0.041 | 0.145 ± 0.070 | 0.170 ± 0.078 | 0.092 ± 0.055 | 0.637 ± 0.267 |

**关键观察**：
- fold 间差异极大（如 100% 标签时 fold1 recall=0.997，fold0 recall=0.160），存在明显的 **all-foreground / all-background 崩溃现象**。
- `best-F1` 显著高于 `F1@0.5`，说明分数相对排序尚可，但 0.5 阈值并非最优，且概率校准不佳。
- Precision 普遍极低（~0.07–0.10），Recall 普遍偏高，说明模型倾向过预测前景。

---

## 2. 下游架构现状

当前使用的 `UperNetHead`（`downstreams/downstreams/heads/segmentation_head.py:56`）结构如下：

1. 对输入 `(B, 64, 128, 128)` 做 4 组 AdaptiveAvgPool2d（尺度 1、2、3、6）+ 1×1 Conv + BN + ReLU。
2. 将原特征与 4 组池化上采样后的特征拼接（通道数 `64 + 4×16 = 128`）。
3. 经 3×3 Conv + BN + ReLU + Dropout(0.1) 融合。
4. 最终 1×1 Conv 输出 2 类 logits，**保持 128×128 分辨率**。

训练流程：
- 只取前景 logits：`model(emb)[:, 1]`。
- 使用 `FocalDiceLoss`：BCE focal 项 + 全局 Dice。
- 优化器 AdamW + CosineAnnealingLR；早停 patience=10；验证指标为 `val_miou`。

数据侧：
- 仅做水平/垂直翻转增强。
- `scene_embedding.pt` 已预生成但未被任务使用。
- 多月份 embedding 已预生成，但只使用 `202604`。

---

## 3. 架构级问题诊断

### 3.1 类别不平衡处理缺失 / 实现有误（优先级：P0）

- `_base_.yaml` 中配置了 `pos_weight: 10.0`，但 `FocalDiceLoss` 未使用它；`alpha=0.8` 是乘以所有像素的常数，不是按类别加权。
- `_dice_loss` 在 **整个 batch 上求和**，未按图像归一化。小批量中若某张图前景极少，全局 Dice 会被背景主导。
- 最终 1×1 Conv 的 bias 默认初始化为 0。对于 ~4% 前景的二分类任务，这会导致初始 logits 偏向 0.5，网络在极小数据集上容易滑向全前景或全背景局部最优。
- 这与观察到的“低 precision、高 recall / 全前景崩溃”高度吻合。

### 3.2 Decoder 容量与设计不足（优先级：P0）

- `UperNetHead` 过于简化：PSP 模块的池化尺度在 128×128 上得到的全局上下文非常粗，且没有 **atrous 空间金字塔（ASPP）** 来捕获多尺度局部结构。
- 没有从 embedding 到高分辨率特征的 **skip connection**，边界恢复能力弱。
- `UNetHead` 的设计是“先上采样 4× 再插值回原始尺寸”，中间尺寸并无真实高分辨率信息，反而引入转置卷积棋盘伪影；skip 仅来自输入本身，等于没有多级特征。
- `FCNHead` 仅一层 3×3 + 1×1，容量更低。
- 对建筑工地这种“小目标、不规则边界”的任务，现有 head 无法有效精炼空间细节。

### 3.3 缺少高分辨率信息输入（优先级：P0–P1）

- 当前 embedding 仅用 S2 / S1 / Landsat / WorldCover 等低分辨率源生成，未使用 Haidian 已存在的 `highres_optical` 与 `highres_sar`。
- AEFModel 本身已具备 `AvailabilityAwareFusion`（`src/xuannv_embedding/models/highres_fusion.py`）用于融合高分数据，但当前下游未使用对应 embedding。
- 建筑工地边界在高分辨率影像上更清晰；缺失高分分支会显著限制分割精度上限。

### 3.4 仅使用单月份、未利用场景嵌入（优先级：P1）

- 下游只读取 `202604_embedding_map.pt`，丢弃了 `scene_embedding.pt` 与相邻月份 embedding。
- 场景级嵌入可提供全局上下文（如“该区域城市化程度”），用于 FiLM / 特征门控，有助于减少 fold 间波动。
- 多月份集成可提升 embedding 对临时噪声（如云、季节变化）的鲁棒性。

### 3.5 vMF 嵌入与 head 的适配问题（优先级：P1）

- embedding 位于单位球面，每个像素 L2=1；直接接 BN + ReLU 会改变其角结构，且 64 维空间中信息密度有限。
- 当前没有可学习的 embedding adapter；加一个轻量 1×1 投影 + GN 可能让下游 head 更好地学习任务相关特征。

### 3.6 训练/早停策略放大了小数据集不稳定性（优先级：P1）

- 数据集仅 62 patch，验证集 9–10 patch；以 `val_miou` 早停极易受噪声影响。
- 某些 fold 的 `best_epoch=0` 且 recall≈1，说明模型在第一轮就陷入全 foreground，早停未能纠正。
- 训练时 `num_workers=0`、batch_size=8 在小数据上问题不大，但缺少针对小样本的正则化（dropout、label smoothing、强增强）。

### 3.7 固定 0.5 阈值与后处理（优先级：P2）

- 虽然 `best-F1` 较高，但实际部署和汇报使用 `F1@0.5`；0.5 阈值对类别不平衡问题并非最优。
- 这不是模型架构问题，但在小数据实验中会通过“选错 epoch”间接放大架构缺陷。

---

## 4. 改进建议与优先级

| 优先级 | 改动 | 关键实现点 | 预期影响 | 估算工作量 |
|---|---|---|---|---|
| **P0** | **修复损失函数与输出层初始化** | 1) 让 `pos_weight` 生效（如 `BCEWithLogitsLoss(pos_weight=...)` 或 Focal 按类别加权）；2) 改为 **per-image Dice/Tversky** 并归一化；3) 最终 conv bias 初始化为 `-log((1-π)/π)`（π≈0.04）；4) 加入边缘/距离加权。 | 显著降低全前景/全背景崩溃，提升 precision，mIoU 可能提升 0.02–0.05 | 0.5–1 天 |
| **P0** | **升级 Decoder 为 DeepLabV3+ / ASPP 或多级 U-Net** | 在 128×128 上实现 `ASPP(rates=6,12,18)` + 1×1 + 全局池化；decoder 用 3×3 conv 融合 ASPP 输出与投影后的 embedding skip；避免无意义的上采样-下采样。 | 增强多尺度上下文与边界定位，mIoU 可能提升 0.05–0.10 | 1–2 天 |
| **P0/P1** | **使用高分 embedding 分支** | 用 `haidian_128_stage2_infer.yaml`（含 `highres_optical`/`highres_sar`）重新生成 embedding，或在现有 inference 代码中传入 highres 数据；下游 head 输入维度不变。 | 边界精度大幅提升，是突破当前上限的最有效路径之一 | 1–2 天（重新跑 inference）|
| **P1** | **加入场景嵌入做全局调制** | 将 `scene_embedding.pt`（64 维）复制到空间维度，与像素特征拼接或做 FiLM/SE 门控，稳定 fold 间表现。 | 降低 fold 方差，提升 recall/precision 平衡 | 0.5 天 |
| **P1** | **多月份 embedding 集成** | 将 `202603`、`202604`、`202605` 等相邻月份 embedding 作为额外通道输入，或在推理时多月份预测平均。 | 增强时序鲁棒性，可能提升 0.01–0.03 mIoU | 0.5–1 天 |
| **P1** | **增加正则化与增强** | head 中加入 Dropout(0.1–0.3)；对 embedding 做随机 90° 旋转、scale 抖动；使用 label smoothing。 | 缓解 62 patch 过拟合，降低 fold 方差 | 0.5 天 |
| **P2** | **增加轻量 embedding adapter** | 在 head 前加 1×1 Conv（64→128/256）+ GN + ReLU，学习从 vMF 球面到任务空间的投影。 | 可能小幅提升表达能力 | 0.5 天 |
| **P2** | **基线对照与阈值优化** | 跑 `linear_probe` 与 `fcn` 基线；在验证集上选最优阈值再计算 `F1@0.5`；早停改为监控 `val_best_f1` 或平均 AP。 | 明确 head 改进是否真正有效，避免被阈值误导 | 0.5 天 |

---

## 5. 为什么当前指标“卡”在这里？

综合判断，当前 mIoU ~0.09 的核心瓶颈不是 embedding 分辨率（128×128 与 mask 对齐），而是：

1. **Head 太弱 + 损失函数没处理好类别不平衡**：小目标二分类任务对输出层 bias 和 per-image 损失非常敏感，当前实现让模型很容易滑向全 foreground。
2. **没有高分信息**：建筑工地在纯 10m 低分辨率 embedding 中边界模糊，head 本身又无法从粗粒度特征中恢复精细轮廓。
3. **数据太少、验证噪声大**：62 patch 导致 fold 间极不稳定，任何小改动都应配合更鲁棒的训练策略（正则化、更好的早停指标、阈值优化）。

---

## 6. 推荐实验顺序

1. **立即做（1 天内）**：修复 `focal_dice` 的类别加权、加 per-image Dice/Tversky、输出 bias 初始化；重跑 100% 标签 5-fold，观察 fold 崩溃是否消失。
2. **随后做（1–2 天）**：将 `UperNetHead` 替换为带 ASPP + skip 的 DeepLabV3+ 风格 decoder；与 linear probe / FCN 做对照。
3. **并行做（1–2 天）**：用 Haidian 高分数据重新生成 embedding（`highres_optical` + `highres_sar`），比较低分 vs 高分下游结果。
4. **优化阶段（2–3 天）**：加入场景嵌入调制、多月份集成、更强的数据增强与早停策略，进一步压减 fold 方差。

---

## 7. 附录：关键文件速查

- 下游 head：`downstreams/downstreams/heads/segmentation_head.py`
- 下游任务：`downstreams/downstreams/tasks/construction_segmentation.py`
- 损失函数：`downstreams/downstreams/tasks/construction_segmentation.py:21`
- 数据加载：`downstreams/downstreams/data/embedding_dataset.py`
- 训练脚本：`downstreams/scripts/train_task.py`
- 上游模型：`src/xuannv_embedding/models/model.py`
- 高分融合：`src/xuannv_embedding/models/highres_fusion.py`
- vMF bottleneck：`src/xuannv_embedding/models/bottleneck.py`
- v2 汇总 markdown：`/data/xuannv_embedding/outputs/downstream/stage2_haidian_construction_v2_aggregate.md`
