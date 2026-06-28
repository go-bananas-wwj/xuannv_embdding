# P2E change-anchor balance sanity 记录（2026-06-28）

## 目标

P2E 从 P2A 初始化，做一个主改动：把训练重心往变化检测锚点拉回去。

P2D 说明 semantic hard-negative 对 OSM simple probe 有一点帮助，但变化检测没有恢复到 P2A。P2E 因此不继续加 hard-negative，而是：

- 加强 `supervised_change_alignment_loss`。
- 把 `construction` 加入 supervised change tasks。
- 降低 semantic probe 权重。
- 关闭 semantic hard-negative。

通俗地说，这轮让模型学建筑/道路时，不能忘记“两期影像哪里真的变了”。

## 运行配置

- 配置文件：`configs/v2_p2e_change_anchor_balance_sanity_20260628.yaml`
- 初始化权重：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- 输出目录：`/data/xuannv_embedding/outputs/v2_p2e_change_anchor_balance_sanity_20260628`
- 设备：单卡 NPU 0
- epoch：5

## 关键改动

相对 P2A/P2D：

- `supervised_change_weight`: `0.06 -> 0.10`
- `supervised_change_tasks`: 新增 `construction`
- `supervised_change_pos_margin`: `0.35 -> 0.40`
- `supervised_change_neg_margin`: `0.05 -> 0.04`
- `supervised_change_pos_weight`: `1.5 -> 2.0`
- `supervised_change_neg_weight`: `0.5 -> 0.35`
- `semantic_probe_weight`: `0.08 -> 0.06`
- `semantic_probe_hard_negative_weight`: `0.25 -> 0.0`

## 训练结果

| epoch | train loss | val loss |
| --- | ---: | ---: |
| 0 | 1.686796 | 1.649417 |
| 1 | 1.667642 | 1.678188 |
| 2 | 1.697862 | 1.702625 |
| 3 | 1.660765 | 1.682049 |
| 4 | 1.635867 | 1.665859 |

最佳 checkpoint：

- `/data/xuannv_embedding/outputs/v2_p2e_change_anchor_balance_sanity_20260628/best.pt`
- best epoch：4
- best val loss：1.665859

## 观察

- 训练链路完整跑完，没有 NaN/OOM/崩溃。
- 新增 `construction` 标签被正常加载，metrics 中有 `supervised_change_construction_*`。
- 关闭 semantic hard-negative 后，`semantic_probe_*_hard_negative_weight` 为 0，符合预期。
- 采样权重生效：`positive_weighted=701/744`，mean weight 为 `4.611`。相比 P2D 更偏向变化标签，OSM 采样权重被压低。
- 单卡 sanity 明显慢于 P2D，正式训练必须用 6 卡。

## 结论

P2E sanity 通过。下一步启动 6 卡正式训练：

- 使用 NPU 0-5。
- 从 P2A best checkpoint 初始化。
- 使用 `configs/v2_p2e_change_anchor_balance_full_20260628.yaml`。
- 训练完成后复用固定流程：导出 embedding、变化检测测评、OSM linear/MLP/U-Net 测评、红色诊断可视化、生成报告。
