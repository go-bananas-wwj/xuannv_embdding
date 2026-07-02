# P5H 海淀月度通用嵌入模型最终简版计划

日期：2026-07-02

## 1. 这次到底要做什么

我们不再围绕旧的变化检测任务做优化，也不把模型训练成某个任务的专用模型。

P5H 要做的是一层“海淀月度地理嵌入数据层”：

- 每个 10m 像素一个 embedding。
- 每个月一套 embedding。
- 输出保持 128x128 空间分辨率。
- 少量 few-shot 标注后，用 kNN、原型分类、1x1 linear head 或小 MLP 就能扩展到全海淀。

简单说：模型要学会“这里是什么地物”，而不是只学会“重建影像”。

## 2. 参考了什么

本计划参考这些方向：

- AEF / AlphaEarth Foundations：先产出通用 embedding field，再用稀疏标签快速制图。
- Learned Embedding Fields / AEFL：多源、多时相、一套 embedding 服务多任务。
- OlmoEarth：多模态、时空遥感数据，强调稳定 latent 表征和 kNN/linear probe 评估。
- SatMAE：显式建模时间和多光谱结构，不同时间独立 mask。
- Prithvi-EO / Clay：把时间、位置和影像一起作为 embedding 条件。

落到我们这里，就是：

> 做海淀-only 的月度 AEF，而不是小号全球基础模型。

## 3. 数据怎么用

只使用海淀区。

影像数据：

- S2：2025-01 到 2026-05，17 个月。
- S1：2025-01 到 2026-05，17 个月。
- Landsat：2025-01 到 2026-05，17 个月。
- 高分 SAR：2025-01 到 2026-05，17 个月。
- 高分光学：2025-12 到 2026-04，5 个月。

标签数据：

- OSM 用作弱语义提示。
- 用户人工 few-shot 标注用于验证和少量原型校正。
- 旧的变化检测标签不进入主模型训练。

## 4. OSM 标签怎么整理

OSM 不当标准答案，只当“语义提示”。

### 高可信地物层

这些和遥感影像对应更直接，权重较高：

- building
- road / major_road / minor_road
- rail
- water
- green
- agriculture
- path_walk
- playground

### 功能区层

这些描述城市功能，边界不一定准，权重中低：

- residential
- commercial
- industrial
- construction

### 活动线索层

这些更像“附近有什么活动”，不要当像素级真值：

- activity_poi

### 派生组合层

把 OSM 变成更稳的结构特征：

- 建成环境强度 = building + road + residential/commercial/industrial
- 交通可达性 = major_road + minor_road + rail + path_walk
- 生态开放空间 = green + water + agriculture + playground
- 开发扰动提示 = construction + 裸地/纹理变化

训练时使用多标签软监督、label smoothing、低权重和置信度 mask。OSM 缺失不能当强负样本。

## 5. 模型输出

建议输出：

```text
embedding_map: [B, 17, D, 128, 128]
```

其中：

- 17 表示 2025-01 到 2026-05 的 17 个月。
- D 建议先用 64；如果稳定，再尝试 96。
- 保留 dense spatial embedding，也导出 patch-level pooled embedding 用于检索。

## 6. 训练目标

### 主目标：月度地表状态

模型每个月只表达这个月的地表状态。

它要知道：

- 哪里像建筑。
- 哪里像道路。
- 哪里像水体。
- 哪里像绿地/林地/山地。
- 哪里像农田。
- 哪里像裸地或施工扰动。

### 自监督目标

使用影像自身训练：

- 重建 S2/S1/Landsat。
- 重建 NDVI、NDWI、NDBI、BSI 等指数。
- 高分影像只在有月份时监督。
- 高分不只重建 RGB，还提取边缘、纹理、结构作为 teacher。

### 时序目标

不是做变化检测，而是让 embedding 更稳：

- 稳定地物跨月 embedding 应该接近。
- 真实地表状态明显不同的月份 embedding 应该拉开。
- 植被季节变化不要被误当成强变化。

### 语义目标

OSM 只做低权重弱语义：

- 帮模型理解建筑、道路、水体、绿地、功能区。
- 不让模型变成 OSM 标签预测器。

### 原型目标

为 few-shot 服务：

- 从高可信 OSM 和少量人工标注中抽取类别原型。
- 同类 embedding 靠近。
- 异类 embedding 拉开。
- 混合边界区域降权。

## 7. 训练阶段

### P5H-A：影像自监督预训练

目标：先学会海淀 17 个月的地表状态。

使用：

- S2/S1/Landsat/高分 SAR。
- 高分光学作为有月份 teacher。
- 指数重建和结构重建。

### P5H-B：OSM 弱语义 + 原型训练

目标：让 embedding 空间更适合 few-shot。

使用：

- OSM 多标签软监督。
- 高可信地物原型。
- 低权重功能区语义。

### P5H-C：冻结 embedding 做 few-shot 验收

目标：验证 embedding 好不好。

必须先跑简单头：

- kNN
- prototype classifier
- 1x1 linear probe
- small MLP

Tiny UNet 只能作为上限参考，不能用它证明 embedding 已经强。

## 8. 验收方式

必须做 spatial block holdout，避免临近 patch 泄漏。

每个 checkpoint 都要看：

- kNN / prototype / linear probe 指标。
- 建筑、道路、水体、绿地、土地利用、施工地的 few-shot 结果。
- 类内距离是否小。
- 类间距离是否大。
- with/without location embedding 对比，防止模型只记位置。
- 全域可视化图。

第一版目标：

| 任务 | 目标 |
|---|---:|
| 水体 | F1_best >= 0.75 |
| 建筑 | F1_best >= 0.55 |
| 道路 | F1_best >= 0.50 |
| 绿地/林地 | F1_best >= 0.65 |
| 土地利用大类 | macro F1 >= 0.55 |
| 施工地 | F1_best >= 0.35 |

## 9. 最大风险

最大风险是模型变成“海淀坐标记忆器”。

因为海淀只有 320 个 patch，OSM 又很多，如果位置编码和 OSM 权重太强，模型可能只是记住“这个地方大概率是什么”，而不是学会“这种影像模式是什么地物”。

解决办法：

- location embedding 做 ablation。
- few-shot 必须空间隔离测试。
- OSM 不做强标签。
- 原型只用高置信纯像素。
- 以 kNN/linear probe 作为主要验收。

## 10. 不做什么

本轮明确不做：

- 不用旧变化检测标签训练主模型。
- 不追求复杂大模型。
- 不只看重建 loss 选 best。
- 不把 OSM 当标准答案。
- 不用复杂下游头掩盖 embedding 本身不足。
- 不继续做“加 epoch、调权重、加 hard recon”的小修小补。

## 11. 最终一句话

P5H 要做的是：

> 海淀区每个月一张通用地理 embedding field。影像负责教真实地表状态，OSM 负责补充弱语义，少量人工标注负责形成 few-shot 原型，最终用简单下游头在全海淀制图。
