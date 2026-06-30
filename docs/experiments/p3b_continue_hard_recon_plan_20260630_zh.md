# P3B 继续训练 + 困难重建实验计划（2026-06-30）

## 背景

P3A 8 小时训练的最佳权重出现在最后一轮，说明训练损失和验证损失还没有完全停止改善；但 P3A 下游 quick eval 显示变化检测任务明显退化，尤其是 `building_change` 和 `farm_change`。因此下一步不能只盲目增加 epoch，需要验证“继续训练是否有效”，同时避免 OSM 弱语义把 embedding 进一步推向静态语义。

## 行业经验对应

遥感自监督和地理基础模型常见做法包括长时间预训练、masked reconstruction、跨时相/多光谱重建和多尺度重建。直接套到当前项目里，最稳妥的低风险验证是使用已有 source dropout 做“困难重建”：随机遮掉更多源模态或高分模态，让模型在信息不完整时仍然重建目标，同时保留时序对比约束。

## 实验配置

- 配置：`configs/v2_p3b_continue_hard_recon_20260630.yaml`
- 初始化权重：`/data/xuannv_embedding/outputs/v2_p3a_osm_weak_semantic_8h_20260629/best.pt`
- 输出目录：`/data/xuannv_embedding/outputs/v2_p3b_continue_hard_recon_20260630`
- 训练硬件：NPU 0-5，6 卡 DDP
- 训练轮次：300 epoch
- 学习率：`2e-5`

## 相比 P3A 的关键变化

| 项 | P3A | P3B |
|---|---:|---:|
| epochs | 600 | 300 continuation |
| lr | 5e-5 | 2e-5 |
| highres optical weight | 0.25 | 0.40 |
| highres SAR weight | 0.25 | 0.35 |
| temporal contrast weight | 0.03 | 0.05 |
| semantic probe weight | 0.06 | 0.04 |
| low-res source dropout | 0.10 | 0.15 |
| high-res source dropout | 0.20 | 0.35 |

## 验收方式

训练完成后复用固定流程：

1. 导出 202512-202605 embedding；
2. 跑下游 quick fold-0；
3. 与 AEF 对比；
4. 与 P2A/P3A 对比；
5. 查看可视化，重点看两期 embedding 是否重新出现时序差异。

## 成功标准

- `building_change` 和 `farm_change` AUC 不能继续接近随机，应明显高于 P3A；
- `construction` 和 `rubbish` 不应明显回退；
- macro AUC 需要重新接近或超过 P2A/P1B；
- 可视化上 202512 和 202605 embedding/PCA 不应几乎一样。
