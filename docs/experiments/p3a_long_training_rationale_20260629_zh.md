# P3A 长训配置依据

- 日期：2026-06-29
- 配置：`configs/v2_p3a_osm_weak_semantic_long_20260629.yaml`
- 目标：把 P3A 当作主 embedding 底座训练，而不是短周期小实验。

## 为什么要长训

P3A 的目标不是训练一个下游任务头，而是训练“每个月地表状态”的通用 embedding。它同时包含：

- S2/S1/Landsat/WorldCover/高分影像重建；
- temporal contrast 自监督；
- OSM 多类别弱语义 probe；
- 海淀 + 哈尔滨两区联合训练。

按当前数据量估算，744 个 patch、6 卡、每卡 batch size 6，单个 epoch 只有二十来个 batch。即便训练 100 epoch，总优化步数也不算多。对于 embedding 预训练来说，这偏短。

## 行业参考

几个遥感 foundation / pretrain 工作的共同点是：长时间、多模态、自监督预训练，而不是短训。

- Prithvi-EO-2.0：基于 MAE 的多时相地球观测 foundation model，论文中提到 300M/600M 模型训练 400 epochs。
- SatMAE：强调时序和多光谱结构的 masked autoencoder 预训练，通过 temporal/spectral encoding 和独立 mask 学遥感表示。
- CROMA：把 SAR/光学的 contrastive 与 reconstruction 目标结合，说明遥感表示学习通常需要多目标预训练。
- AnySat：强调多数据集、多模态、多分辨率联合训练，符合我们海淀 + 哈尔滨 + 多源数据联合训练的方向。

参考链接：

- Prithvi-EO-2.0: https://arxiv.org/html/2412.02732v3
- SatMAE: https://arxiv.org/abs/2207.08051
- CROMA: https://arxiv.org/abs/2311.00566
- AnySat: https://arxiv.org/abs/2412.14123

## 本次配置调整

从短训 P3A 改成长训 P3A：

| 项目 | 原配置 | 长训配置 |
|---|---:|---:|
| epochs | 80 | 300 |
| warmup_epochs | 5 | 10 |
| semantic_probe_warmup_epochs | 8 | 12 |
| temporal_contrast_warmup_epochs | 8 | 12 |
| eval_every | 5 | 10 |
| save_every | 10 | 20 |
| lr | 5e-5 | 5e-5 |

学习率保持 `5e-5`，因为这是从 P2A best checkpoint 继续训练，并且 OSM 是弱标签，过大学习率可能把已有 embedding 拉偏。

## 训练策略

1. 使用 P2A best checkpoint 初始化。
2. 使用海淀 + 哈尔滨联合 manifest。
3. OSM 只作为弱语义辅助，不进入 supervised change。
4. 使用 6 张 NPU 训练。
5. 训练后按固定 post-training eval 流程导出 embedding、下游测评、和 AEF/P2A 对比。

## 风险控制

- 由于 OSM 弱标签噪声较大，保持 `semantic_probe_weight=0.06`，不让 OSM 主导训练。
- `osm_green`、`osm_water`、`osm_activity_poi` 这类大面覆盖类别已经低权重。
- 每 10 epoch 做一次评估，每 20 epoch 保存 checkpoint。
- 最终优先看 best checkpoint，而不是盲目使用最后一个 checkpoint。
