# P3A 8 小时训练计划

- 日期：2026-06-29
- 配置：`configs/v2_p3a_osm_weak_semantic_8h_20260629.yaml`
- 目标：按约 8 小时预算训练 P3A OSM 弱语义版 embedding。

## 当前速度估算

长训版启动后已经完成前 3 个 epoch：

| epoch | 时间 | train loss |
|---:|---|---:|
| 0 | 15:49:17 | 1.627133 |
| 1 | 15:50:06 | 1.635730 |
| 2 | 15:50:54 | 1.638077 |

稳定阶段约 `48 秒 / epoch`。因此：

- 300 epoch 约 4 小时；
- 600 epoch 约 8 小时；
- 加上评估和 checkpoint 保存，实际可能是 8 到 8.5 小时。

## 8 小时配置

| 项目 | 设置 |
|---|---:|
| epochs | 600 |
| warmup_epochs | 20 |
| semantic_probe_warmup_epochs | 24 |
| temporal_contrast_warmup_epochs | 24 |
| save_every | 25 |
| eval_every | 10 |
| lr | 5e-5 |
| batch_size | 6 / NPU |
| NPU | 0-5 |

## 为什么这样设

P3A 是 embedding 底座训练，不是普通下游小任务。当前数据只有 744 个 patch，6 卡训练时每个 epoch 的优化步数不多，所以需要更长 epoch 来积累足够优化步数。600 epoch 更接近“预训练”节奏，同时不会过度超出 8 小时预算。

## 风险控制

1. 继续从 P2A best checkpoint 初始化。
2. 学习率保持 `5e-5`，避免 OSM 弱标签把已有 embedding 拉偏。
3. OSM 仍然只作为弱语义 probe，不进入 supervised change。
4. 每 10 epoch 评估一次，每 25 epoch 保存一次。
5. 最终优先使用 best checkpoint 做下游测评。
