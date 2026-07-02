# P5H 海淀区专用地表状态嵌入模型训练方案

日期：2026-07-02

## 1. 目标

P5H 的目标不是继续做泛化到多城市的变化检测模型，而是先把海淀区一个区域做精：

- 为海淀区 320 个 patch 生成稳定、可解释、可 few-shot 迁移的像素级 embedding。
- embedding 要表达“这块地是什么”，例如建筑、道路、水体、林地、山地、绿地、农田、商业区、居民区、施工地等。
- 用户只标少量样本后，简单下游头就能扩展到全海淀，例如标几个建筑物样本后训练建筑头，能够识别全区建筑分布。
- 道路提取、建筑提取、水体提取、土地利用分类、绿地/林地/山地识别等任务都要有稳定指标。
- 本轮不使用之前的人工变化检测标签训练主嵌入模型；变化检测标签只可作为后验诊断，不进入主训练。

一句话：P5H 要从“重建型 embedding”转成“海淀月度地表状态 embedding”。

## 2. 当前海淀数据基础

海淀区目前数据比哈尔滨更适合做区域专用模型，因为时序更长、标签体系更完整。

### 2.1 影像数据

当前本地数据盘显示：

| 数据源 | 时间覆盖 | 文件数 | 说明 |
|---|---:|---:|---|
| S2 | 2025-01 到 2026-05，共 17 个月 | 27344 | 主光学时序 |
| S1 | 2025-01 到 2026-05，共 17 个月 | 22960 | SAR 时序，抗云，补结构信息 |
| Landsat | 2025-01 到 2026-05，共 17 个月 | 20603 | 中分辨率长期稳定参考 |
| 高分光学 | 2025-12 到 2026-04，共 5 个月 | 1600 | 每月 320 patch，可作为细节/边界 teacher |
| 高分 SAR | 2025-01 到 2026-05，共 17 个月 | 27953 | 高分结构和纹理补充 |

### 2.2 标签数据

海淀 OSM 弱语义标签覆盖 320 patch：

| 标签 | mask 数 | 正样本 patch | 平均正像素比例 |
|---|---:|---:|---:|
| osm_building | 320 | 290 | 0.0960 |
| road_osm | 320 | 317 | 0.1563 |
| osm_major_road | 320 | 274 | 0.0893 |
| osm_minor_road | 320 | 314 | 0.1060 |
| osm_water | 320 | 256 | 0.0376 |
| osm_green | 320 | 319 | 0.3520 |
| osm_agriculture | 320 | 162 | 0.0663 |
| osm_residential | 320 | 257 | 0.1367 |
| osm_commercial | 320 | 222 | 0.0574 |
| osm_industrial | 320 | 185 | 0.0365 |
| osm_construction | 320 | 220 | 0.0503 |
| osm_rail | 320 | 157 | 0.0176 |
| osm_path_walk | 320 | 291 | 0.0539 |
| osm_playground | 320 | 210 | 0.0119 |

人工施工地标签较稀疏：

- 320 个 patch 中只有 62 个施工地 mask 文件。
- 其中 51 个有施工地正样本，11 个为空标注。
- 这批标签适合做 few-shot 验证和小权重 probe，不适合作为主模型核心监督。

## 3. 之前实验的失败经验

### 3.1 只靠重建不会自动形成强语义

P3B/P3C/P4A 增强了高分重建和困难重建，但下游变化类任务没有明显提升，甚至下降。

典型结果：

| 实验 | construction F1 | building_change F1 | farm_change F1 | rubbish F1 | 说明 |
|---|---:|---:|---:|---:|---|
| P1 full 5-fold | 0.3705 | 0.2524 | 0.1603 | 0.4023 | 采样/困难样本有效，但仍偏任务技巧 |
| P3C monthly state hard recon | 0.2221 | 0.0100 | 0.0025 | 0.1563 | 重建增强后变化类基本失败 |
| P4A monthly highres | 0.1978 | 0.0393 | 0.0025 | 0.1077 | 水体强，但变化/施工弱 |

教训：影像重建主要逼模型学颜色、亮度、纹理、传感器统计，不一定学“地物是什么”。

### 3.2 OSM 弱语义能帮静态任务，但不能自然解决变化敏感性

P4A 在海淀静态任务上表现更好：

| 任务 | F1_best | mIoU | 说明 |
|---|---:|---:|---|
| 海淀建筑 OSM | 0.3871 | 0.2370 | 有一定建筑语义 |
| 海淀道路 OSM | 0.4131 | 0.2596 | 道路分布可读 |
| 海淀水体 OSM | 0.5555 | 0.3796 | 水体可分性较强 |
| 施工地 | 0.1978 | 0.0828 | 仍弱 |

教训：OSM 是有效的区域弱语义，但它主要描述长期静态属性；如果权重和目标设计不好，embedding 会偏“地理先验”，不一定表达月度状态。

### 3.3 变化检测标签不应继续作为主训练目标

之前用变化类标签时，短期指标可能上升，但会让模型过拟合下游定义，削弱通用地物 embedding。

本轮目标是“海淀地物嵌入”，不是“变化检测专用模型”，所以 building_change/farm_change/rubbish 等旧变化标签不进入主训练。

### 3.4 best.pt 不能只按重建 val_loss 选

当前 best checkpoint 按验证总 loss 选择。这个 loss 主要由重建、弱语义、正则项组成，不等价于 few-shot 下游可用性。

P5H 必须新增 embedding 质量验证指标，包括 few-shot probe、原型检索、类内/类间距离、空间一致性。

## 4. 行业最佳实践启发

### 4.1 AEF / AlphaEarth Foundations

AlphaEarth Foundations 的重点是生成像素级 64 维地理 embedding field，并服务不同下游制图任务。它强调用统一 embedding 支撑多任务，而不是为每个任务单独训练模型。对 P5H 的启发是：海淀 embedding 要成为“全区可复用数据层”，few-shot 下游头只做轻量适配。

参考：[AlphaEarth Foundations](https://arxiv.org/html/2507.22291v1)

### 4.2 SatMAE

SatMAE 使用 masked autoencoder 预训练遥感时序/多光谱 Transformer，引入 temporal embedding，并对不同时相独立 mask；多光谱部分使用 spectral positional encoding。对 P5H 的启发是：不要只做普通重建，要显式建模时间和光谱结构。

参考：[SatMAE](https://arxiv.org/abs/2207.08051)

### 4.3 Prithvi-EO-2.0

Prithvi-EO-2.0 使用大规模 HLS 时序数据，加入 temporal 和 location embeddings，并在多类 EO 任务上提升。对 P5H 的启发是：即使只做海淀，也要加入月度编码、位置编码和区域内空间上下文，而不是把 patch 当成无坐标图片。

参考：[Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732)

### 4.4 Clay

Clay 的模型输入影像、位置和时间，输出某地点某时间的 embedding，并使用 MAE 自监督训练。对 P5H 的启发是：embedding 应定义为“某地点在某时间的表示”，不是一张静态图的表示。

参考：[Clay Foundation Model](https://clay-foundation.github.io/model/)

### 4.5 DOFA

DOFA 用 wavelength-conditioned 动态机制统一不同遥感传感器。对 P5H 的启发是：S2、S1、Landsat、高分光学、高分 SAR 不应简单拼接，应显式告诉模型来源、波段、分辨率和观测月份。

参考：[DOFA](https://arxiv.org/abs/2403.15356)

## 5. P5H 总体路线

P5H 采用三阶段训练：

1. P5H-A：海淀长时序自监督地表状态预训练。
2. P5H-B：OSM 弱语义和地物原型约束。
3. P5H-C：few-shot 下游头验证与主动标注闭环。

主模型只使用海淀数据。

不使用旧变化检测标签训练主模型。

## 6. P5H-A：长时序地表状态自监督预训练

### 6.1 输入

使用海淀 2025-01 到 2026-05 共 17 个月：

- S2 月度合成。
- S1 月度合成。
- Landsat 月度合成。
- 高分 SAR 月度合成。
- 高分光学 2025-12 到 2026-04。

每个月输出一个 embedding map：

```text
B, 17, D, 128, 128
```

建议 D 从 96 开始；若显存压力大，先用 64。

### 6.2 训练目标

#### 6.2.1 多源月度重建

每个月 embedding 重建当月观测：

- S2，权重 1.0
- S1，权重 0.8
- Landsat，权重 0.6
- 高分 SAR，权重 0.5
- 高分光学，有月份才监督，权重 0.6

连续影像重建不只用 L1，建议改为：

```text
0.5 * L1 + 0.3 * SSIM/Charbonnier + 0.2 * gradient/edge loss
```

原因：单纯 L1 容易学平均颜色，不重视边界。

#### 6.2.2 光谱指数重建

额外重建：

- NDVI：植被/农田/林地。
- NDWI/MNDWI：水体。
- NDBI：建筑/裸地。
- BSI：裸土/施工地。
- SAR ratio 或 VV/VH 派生指数：结构和粗糙度。

这些目标比 RGB 更语义化，必须进入 P5H。

#### 6.2.3 高分结构 teacher

高分光学/SAR 不只是重建 RGB，还要生成结构监督：

- 边缘图。
- 局部纹理强度。
- 建筑/道路细线结构先验。
- 多尺度梯度。

这会让 embedding 更适合建筑、道路、施工地这类边界敏感任务。

### 6.3 时间一致性与状态变化约束

P4A 的 temporal contrast 只看首尾月，太弱。P5H 改为全时序 pairwise 约束。

#### 稳定区域

如果同一像素相邻月 NDVI/NDWI/NDBI/SAR 变化很小：

```text
embedding(t) 和 embedding(t+1) 应接近
```

#### 疑似变化区域

如果同一像素跨月指数变化显著，且不是云/缺测造成：

```text
embedding(t1) 和 embedding(t2) 应拉开
```

#### 季节变化处理

植被有季节性，不能简单把 NDVI 波动都当变化。建议：

- 对绿地/林地/农田区域允许周期性变化。
- 对建筑/道路/水体区域要求跨月更稳定。
- 使用 OSM 静态类作为稳定区域 prior。

## 7. P5H-B：OSM 弱语义和地物原型约束

### 7.1 OSM 弱语义任务

建议保留这些 OSM 任务：

- building
- major_road
- minor_road
- path_walk
- rail
- water
- green
- agriculture
- residential
- commercial
- industrial
- construction
- playground

但要分权重：

| 类别 | 权重策略 |
|---|---|
| building / road / water | 较高，0.8 到 1.0 |
| green / agriculture / residential | 中等，0.5 到 0.7 |
| commercial / industrial / construction | 中等偏低，0.4 到 0.6 |
| path / rail / playground | 低权重，0.2 到 0.4 |

OSM 是弱标签，不能当精标。建议使用 soft target 和 label smoothing。

### 7.2 原型约束

为了支持 few-shot，embedding 空间必须具备“同类聚集、异类分开”的性质。

加入 prototype loss：

1. 从 OSM 高置信区域采样类别原型。
2. 同类像素 embedding 拉近。
3. 不同类原型拉开。
4. 对混合区域降低权重。

示例：

```text
建筑像素应该靠近建筑原型。
水体像素应该靠近水体原型。
道路像素应该靠近道路原型。
建筑和水体原型之间要有角度间隔。
```

这比单纯 BCE probe 更适合 few-shot。

### 7.3 人工标注如何进入

用户自己标的数据分两类：

1. 高可信 few-shot 原型标注。
2. 小规模验证集标注。

人工 few-shot 标注不要一开始就大权重训练主模型。推荐：

- 第一轮只用于训练下游头和评估 embedding。
- 第二轮如果确实稳定，再作为 prototype anchor 小权重加入。
- 每类优先标 5 到 20 个高质量对象，而不是随便涂大面积。

## 8. P5H-C：few-shot 下游验证

P5H 是否成功，不看重建 loss，而看 few-shot 下游头。

### 8.1 下游头类型

必须按从简单到复杂验证：

1. kNN / prototype classifier。
2. Linear probe，1x1 conv。
3. 小 MLP head。
4. Tiny UNet head。

如果只有 Tiny UNet 能好，说明 embedding 不够好；如果 kNN/linear probe 就好，说明 embedding 空间真的强。

### 8.2 few-shot 标注设置

每个任务设计：

| 任务 | few-shot 标注方式 |
|---|---|
| 建筑提取 | 标 5/10/20 个建筑对象或 patch 内局部 polygon |
| 道路提取 | 标 5/10/20 段主路/支路 |
| 水体提取 | 标 3/5/10 个水体区域 |
| 绿地/林地 | 标 5/10/20 个区域 |
| 农田/裸地/施工地 | 标 5/10/20 个高置信区域 |
| 土地利用分类 | 每类 5/10/20 个点或小 polygon |

### 8.3 评估指标

每个任务输出：

- AUC
- AP
- F1_best
- F1@0.5
- mIoU
- boundary F1
- calibration curve
- 可视化全域图
- 每类误检/漏检分析

### 8.4 成功门槛

P5H 第一阶段目标：

| 任务 | few-shot 下游目标 |
|---|---:|
| 水体 | F1_best >= 0.75，mIoU >= 0.60 |
| 建筑 | F1_best >= 0.55，mIoU >= 0.40 |
| 道路 | F1_best >= 0.50，mIoU >= 0.35 |
| 绿地/林地 | F1_best >= 0.65，mIoU >= 0.50 |
| 土地利用大类 | macro F1 >= 0.55 |
| 施工地 | F1_best >= 0.35，mIoU >= 0.20 |

施工地门槛低一些，因为人工标签少、OSM construction 噪声大。

## 9. 模型结构建议

### 9.1 主干

建议保留当前 AEF/STP 框架，但做以下改造：

- num_months 从 6 改为 17。
- embed_dim 从 64 升到 96；显存不够则保持 64。
- 加 month embedding。
- 加 patch 地理位置 embedding。
- 加 source/modality embedding。
- 对光学波段加 wavelength embedding。
- 高分光学/SAR 不直接拼到所有月份，而是按真实月份作为 teacher。

### 9.2 bottleneck

继续使用 vMF / hypersphere embedding。原因：

- few-shot 原型检索适合余弦距离。
- 类间角度间隔更容易解释。
- 与 prototype loss 匹配。

### 9.3 输出

输出保持：

```text
embedding_map: [B, 17, D, 128, 128]
```

导出时每个月一个：

```text
/embeddings/p5h_haidian/.../haidian_patch_000001/202501_embedding_map.pt
...
/202605_embedding_map.pt
```

## 10. 数据增强与采样

海淀只有 320 个 patch，所以必须强增强：

- source dropout：S2/S1/Landsat 随机缺失。
- month dropout：随机遮挡月份。
- band dropout：随机遮挡光谱 band。
- cloud/noise augmentation：模拟云、阴影、噪声。
- spatial crop：128 内随机 crop/resize，但输出仍对齐 128。
- rotation/flip：道路和建筑任务必须有。
- class-balanced sampling：提高 OSM 稀有类、人工施工地、高分变化区域采样概率。

但是不要把人工施工地标签作为主采样唯一依据，避免过拟合。

## 11. 训练配置建议

### P5H-A

- 训练时长：先 300 epoch sanity，再 1000 epoch 长训。
- batch size：6 卡时每卡 4 到 6，梯度累积 2。
- lr：1e-4 起步，warmup 20 epoch，cosine decay。
- weight decay：0.05。
- AMP：开启。
- gradient checkpointing：开启。

### P5H-B

- 从 P5H-A best 初始化。
- 加 OSM semantic loss 和 prototype loss。
- 训练 300 到 500 epoch。
- OSM loss 逐步 warmup，避免一开始被噪声标签带偏。

### P5H-C

- 冻结主模型。
- 跑 few-shot 下游头。
- 输出全域地图和报告。
- 只在 few-shot 结果稳定后，才考虑小权重人工原型微调。

## 12. 不做什么

本轮明确不做：

- 不使用旧变化检测标签训练主 embedding。
- 不把 P5H 做成哈尔滨/海淀联合模型。
- 不用施工地 51 个正样本强监督主模型。
- 不只看 reconstruction val_loss 选 best。
- 不以复杂 UNet 头的好结果证明 embedding 好。
- 不再只做“加 epoch / 调权重 / 加 hard recon”的小修小补。

## 13. 需要新增的工程模块

### 13.1 数据侧

- 海淀 17 个月 manifest。
- 月度指数生成脚本：NDVI/NDWI/NDBI/BSI/SAR ratio。
- 高分结构 teacher 生成脚本：edge/gradient/texture。
- OSM 标签置信度图生成脚本。
- few-shot annotation registry。

### 13.2 模型侧

- source/modality embedding。
- month embedding。
- location embedding。
- optional wavelength embedding。
- prototype loss。
- pairwise temporal state loss。

### 13.3 测评侧

- few-shot split 生成器。
- prototype/kNN/linear/MLP/tinyUNet 四类头统一测评。
- 全域可视化。
- embedding 空间诊断：PCA、UMAP、类内/类间角距离、原型相似度图。

## 14. 里程碑

### M0：数据审查

目标：确认海淀所有输入和弱标签无坐标错位。

交付：

- 17 个月数据清单。
- OSM 标签全域分布图。
- 高分光学/SAR 可用月份图。
- 人工 few-shot 标注规范。

### M1：P5H-A 自监督预训练

目标：得到月度地表状态 embedding 初版。

交付：

- checkpoint。
- 每月 embedding 导出。
- 重建/指数/结构可视化。
- 稳定/疑似变化 embedding 距离分布。

### M2：P5H-B OSM + prototype 训练

目标：让 embedding 具备可 few-shot 分类的语义空间。

交付：

- checkpoint。
- OSM probe 指标。
- prototype similarity maps。
- 类内/类间角度报告。

### M3：P5H-C few-shot 验收

目标：用少量标注训练简单头，验证全域提取能力。

交付：

- 建筑、道路、水体、绿地、土地利用、施工地评估报告。
- 全域预测图。
- 每类误检漏检图。
- 是否进入下一轮人工标注闭环的建议。

## 15. 最终判断

如果我们只做海淀区，正确路线不是继续追求“大模型泛化”，而是把海淀做成一个高质量地表状态 embedding 场。

P5H 成功的关键不在于重建图像多像，而在于：

- 同类地物 embedding 聚在一起。
- 不同地物 embedding 拉开。
- 少量标注可以形成可靠原型。
- 简单下游头能扩展到全域。
- 高分影像用于教结构，OSM 用于教弱语义，人工标注用于验证和少量原型校正。

这才是面向生产的海淀专用嵌入模型路线。
