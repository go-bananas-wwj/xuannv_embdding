# P2 Embedding Upgrade Plan（2026-06-27）

## 结论先说

当前 P1B embedding 已经有变化检测信号，但还不是一个足够好的通用地物 embedding。证据是 simple probe：`linear` 和逐像素 `mlp` 在建筑/道路 OSM 任务上明显低于 U-Net，尤其哈尔滨建筑和道路失败较明显。如果 embedding 本身足够好，简单头应该能读出更干净的建筑、道路、土地利用和变化信息。

## 当前问题拆解

### 1. 训练目标偏“重建”，不够“语义可读”

P1B 训练目标包含 S2/S1/Landsat/highres 重建、WorldCover 分类、uniformity、temporal contrast、supervised change alignment。这里重建占比较大，重建会鼓励 embedding 保存光谱/纹理/传感器信息，但不一定把“建筑、道路、水体、土地利用类别”整理成线性可分的通道空间。

现象：U-Net 可以借助空间 decoder 挖出一点信息，但 linear/MLP 读不干净。

### 2. 变化目标只约束“距离”，没有教 embedding 表达“是什么变了”

P1B 的 supervised change alignment 主要是：变化像素两期 embedding 拉远，不变像素拉近。这能提升变化检测 AUC/F1，但它不知道变化类别，也不知道建筑边界、道路结构、水体范围。它回答的是“变没变”，不是“这个像素是什么”。

现象：变化任务有提升，但建筑/道路语义任务 simple probe 弱。

### 3. 高分辨率稀疏监督没有直接进入 embedding 训练

OSM 建筑/道路、水体、土地利用等标签目前主要用于下游测试，没有作为 embedding 训练阶段的密集监督。模型没有被明确要求在 embedding map 中形成“建筑通道”“道路通道”“水体通道”“土地利用结构”。

现象：下游头需要复杂 decoder 和高阈值，说明 embedding 没有把目标变成简单可读特征。

### 4. 尺度和细线结构不足

道路是窄线目标，建筑是块状边界目标，土地利用是区域语义目标；它们需要不同尺度的信息。当前 embedding 输出保持 128x128 是对的，但训练目标没有专门强化边界、线结构、对象尺度一致性。

现象：道路容易泛红，建筑边界粗糙，mIoU 偏低。

### 5. 标签噪声和伪标签质量还没有被建模

OSM、LocateAnything、SAM2 伪标签会有位置偏差、漏标、误标。直接把伪标签当真值训练会伤害 embedding。需要置信度、边界忽略区、噪声鲁棒 loss。

## 外部经验吸收

参考方向：

- SatMAE：多时相、多光谱 MAE 能提升土地覆盖分类和语义分割迁移能力。
- Scale-MAE：遥感任务需要尺度感知，显式学习不同地面尺度关系可提升 SpaceNet 建筑分割迁移。
- Prithvi-EO-2.0：多时相、地理位置/时间编码、大规模 EO 数据和下游专家反馈对通用 EO 任务很关键。
- RingMoE：多模态遥感需要 modality-specific 与 shared representation，避免不同传感器互相冲突。
- Changen2 / semantic change foundation models：变化检测不应只靠距离约束，最好有 change-aware 或 semantic-change 训练信号。

## P2 升级目标

P2 不只是继续刷变化检测，而是把 embedding 升级成“简单头可读”的通用地物表征。

验收目标：

1. 建筑/道路/水体/土地利用：linear 和 MLP probe 明显提升，不只 U-Net 提升。
2. 变化检测：P1B 已有优势不能丢，building_change、farm_change、rubbish、construction_joint 至少不退化。
3. 概率校准改善：下游阈值不能长期集中在 0.95 到 0.97；预测正类比例不能明显失控。
4. 可视化改善：红色预测图应更贴近 GT，FP/FN 明显减少。

## P2 分阶段计划

### P2A：先加“语义原型监督”，让 embedding 变得可读

核心想法：在 embedding 训练阶段加入轻量 training-only probe head，用 OSM/WorldCover/水体/建筑/道路伪标签监督 embedding。这个 head 训练完不一定用于下游，目的是逼 embedding 空间把语义信息摆出来。

训练标签：

- OSM building mask
- OSM road mask
- WorldCover / 土地利用 mask
- 水体 mask（WorldCover water 或 NDWI/OSM water 伪标签）
- 可选：LocateAnything + SAM2 生成的建筑/道路/水体伪 mask

实现方式：

- 在 embedding trainer 里增加 `semantic_probe_heads`。
- 每个任务一个 1x1 或 tiny MLP probe。
- loss 使用 BCE + Dice/Focal，但对伪标签加 confidence mask。
- OSM 边界附近设置 ignore band，减少坐标偏差伤害。
- 训练权重从小到大 warmup，例如 `semantic_probe_weight: 0.02 -> 0.10`。

预期效果：linear/MLP downstream probe 明显提升。

### P2B：加“类别感知的时序变化监督”，让变化不只会拉远

核心想法：变化像素不只是两期 embedding 距离变大，还要知道变化和什么语义相关。例如建筑新增、农田变化、垃圾/施工变化应该在 embedding 中形成可读方向。

实现方式：

- 保留 P1B supervised change alignment。
- 增加 training-only change probe：输入 `[emb_t1, emb_t2, emb_t2 - emb_t1, abs(diff)]`，预测 change masks。
- 对 sparse change tasks 使用 Dice/Tversky + hard negative mining。
- 对稳定区域保留 consistency loss，但不要过强，避免两期 embedding 完全一样。
- 对有语义伪标签的区域增加 semantic temporal consistency：不变区域语义 probe 输出应一致，变化区域允许改变。

预期效果：变化检测不退化，同时 PCA/embedding diff 对时序变化更敏感。

### P2C：加尺度/结构约束，专门修道路和建筑

道路和建筑不能只用普通 mask loss。

道路：

- road mask 加 centerline/skeleton 或 distance transform 辅助目标。
- 增加 clDice/Tversky 类结构损失到 training-only road probe。
- road buffer 分 coarse/fine 两个版本：粗 mask 提供召回，细 centerline 提供结构。

建筑：

- 建筑 mask 加边界 ignore band。
- 小碎片过滤；低置信 OSM 建筑不参与训练。
- 增加 boundary-aware loss 或 distance map 回归，让 embedding 保留边界。

预期效果：道路 FP 泛滥减少，建筑边界更清晰。

### P2D：减少重建目标对语义的压制

当前重建目标较多，可能让 embedding 学成“压缩影像”的空间，而不是“语义可读”的空间。

调整：

- 保留 S2/S1/Landsat 重建作为基础，但降低 highres reconstruction 权重。
- 增加 masked reconstruction，而不是全图重建，让模型学上下文和语义而不是复制纹理。
- 对 highres optical 增加边缘/低频重建分支，参考 Scale-MAE 思路区分低频语义和高频细节。
- 把 `embed_dim` 从 64 提到 96 或 128 做一次 ablation；下游 concat_diff 会变成 288/384，需评估速度。

### P2E：固定验收流程

每轮训练后固定跑：

- sparse change benchmark：building_change、farm_change、rubbish、construction_joint。
- semantic downstream：building_osm、road_osm、water、landcover。
- simple probe：linear + MLP。
- U-Net probe：作为上限参考。
- 红色诊断图：概率、Pred@0.5、Pred@val_thr、GT、TP/FP/FN。
- 报告指标：F1@0.5、F1@val_thr、mIoU、AUPRC、预测正类比例；AUC 只辅助。

## 推荐执行顺序

### Step 1：P2A small sanity

先不动主模型结构，只增加 semantic training-only probe 和标签读取，跑 5 epoch sanity。

目标：确认 loss 正常、语义标签能进入 batch、不会 NPU OOM。

### Step 2：P2A full quick

训练 100 epoch，使用 P1B 配置为底座，加入 semantic probe loss。

优先任务：building_osm、road_osm、worldcover/water。

### Step 3：P2A quick eval

导出 embedding，跑完整 quick evaluation：simple probe + U-Net + sparse change。

如果 linear/MLP 没提升，说明语义监督接线或权重不够；不要进入 P2B。

### Step 4：P2B change-aware upgrade

在 P2A 有效后再加入 change probe，避免同时改太多不知道谁有效。

### Step 5：P2C structure upgrade

最后专门修道路/建筑结构损失和标签清洗。

## 第一轮建议配置方向

基于 P1B：

- `semantic_probe_weight: 0.08`
- `semantic_probe_warmup_epochs: 10`
- `semantic_tasks: building_osm, road_osm, worldcover, water`
- `supervised_change_weight: 0.06` 保持 P1B，不先加大
- `temporal_contrast_weight: 0.02` 保持 P1B
- `highres_recon_weight: 0.25`，从 0.5 降低
- `embed_dim: 64` 第一轮先不改，减少变量

## 风险

1. OSM 噪声可能把 embedding 带偏，所以必须做 ignore band 和 confidence mask。
2. 语义监督过强可能伤害变化检测，所以第一轮不提高 change loss，不改 embedding 维度。
3. WorldCover 分辨率较粗，不能直接要求精细建筑/道路边界，只适合土地利用语义。
4. 如果 simple probe 提升但 U-Net 不提升，需要看阈值和可视化，不盲目判断失败。

## 通过/失败标准

P2A 通过标准：

- 至少 3/4 个 OSM building/road simple probe 的 `F1_best` 提升。
- `harbin_building_osm` linear/MLP 明显高于 P1B simple probe。
- sparse change macro F1/mIoU 不低于 P1B 过多，允许小幅波动。
- 红色诊断图中 Pred@0.5 不再大面积全红。

P2A 失败标准：

- simple probe 无提升，只有 U-Net 提升。
- 变化检测明显退化。
- 预测正类比例继续失控。

如果 P2A 失败，优先检查标签质量和 loss 权重，不立刻扩大模型。
