# P2A Semantic Probe Sanity（2026-06-27）

## 目的

P2A 的目标是让 embedding 在训练阶段直接接受建筑/道路语义监督，使建筑、道路等任务可以被简单 linear/MLP head 更容易读出来。本次先跑 5 epoch sanity，只验证训练链路、loss、checkpoint 是否正常。

## 已实现

- 在 `TotalLoss` 中新增 training-only `SemanticProbeLoss`。
- Semantic probe 是逐像素 1x1 MLP，只在训练时使用，训练完成后可丢弃。
- 新增配置字段：`semantic_probe_weight`、`semantic_probe_tasks`、`semantic_probe_pos_weight` 等。
- Trainer 现在会把 criterion 参数也加入优化器，并在 checkpoint 中保存 criterion state。
- 新增 P2A sanity 配置：`configs/v2_p2a_semantic_probe_sanity_20260627.yaml`。

## Sanity 结果

输出目录：

`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_sanity_20260627`

训练设置：单卡 NPU 0，5 epoch，`max_patches=96`，用于快速验证。

日志摘要：

| Epoch | train_loss | val_loss |
|---:|---:|---:|
| 0 | 3.164933 | 3.083831 |
| 1 | 3.136181 | 3.094834 |
| 2 | 3.103643 | 3.054069 |
| 3 | 3.084231 | 3.034169 |
| 4 | 3.097804 | 3.031861 |

Best checkpoint：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_sanity_20260627/best.pt`

Checkpoint 中已确认 semantic metrics 存在：

- `semantic_probe`: 约 `1.76`
- `semantic_probe_weighted`: 约 `0.14`
- `semantic_probe_haidian_building_osm_loss`: 约 `1.69`
- `semantic_probe_haidian_road_osm_loss`: 约 `1.81`

由于 sanity 配置只取 manifest 前 96 个 patch，样本主要来自海淀，哈尔滨 semantic 指标为 0。这只影响 sanity 覆盖，不影响正式配置；正式 P2A 会取消 `max_patches`。

## 全量标签覆盖检查

全量 manifest 共 744 个 patch：海淀 320，哈尔滨 424。

| Region | Task | valid patches | positive patches | positive pixels |
|---|---|---:|---:|---:|
| haidian | building_osm | 320 | 290 | 503142 |
| haidian | road_osm | 320 | 317 | 819511 |
| harbin | building_osm | 424 | 166 | 209990 |
| harbin | road_osm | 424 | 375 | 502262 |

## 下一步

启动 6 卡正式 P2A 训练：

- 配置：`configs/v2_p2a_semantic_probe_full_20260627.yaml`
- 初始化：`/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- 输出：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627`

正式训练完成后，必须导出 embedding 并跑固定评估：sparse change、OSM building/road simple probe、U-Net probe、红色诊断可视化。
