# P2D 延迟 hard-negative sanity 记录（2026-06-28）

## 目标

P2D 用 P2A 作为初始化底座，保留 P2C 的 semantic hard-negative 思路，但把 hard-negative 权重延迟 warmup，避免一开始就强行压背景高分点，破坏变化检测和跨城市泛化。

本次 sanity 只验证三件事：

1. 新增 `semantic_probe_hard_negative_warmup_epochs` 训练链路可以正常运行。
2. 从 P2A checkpoint 初始化后，5 epoch 小训练不崩溃。
3. loss 曲线没有明显异常，之后可以进入 6 卡正式训练。

## 运行配置

- 配置文件：`configs/v2_p2d_delayed_hardneg_sanity_20260628.yaml`
- 初始化权重：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- 输出目录：`/data/xuannv_embedding/outputs/v2_p2d_delayed_hardneg_sanity_20260628`
- 设备：单卡 NPU 0
- epoch：5
- hard-negative ratio：0.03
- hard-negative target weight：0.25
- hard-negative warmup：2 epoch

## 训练结果

| epoch | train loss | val loss |
| --- | ---: | ---: |
| 0 | 1.836366 | 1.550039 |
| 1 | 1.755355 | 1.644902 |
| 2 | 1.739403 | 1.576680 |
| 3 | 1.711676 | 1.525706 |
| 4 | 1.711963 | 1.536592 |

最佳 checkpoint：

- `/data/xuannv_embedding/outputs/v2_p2d_delayed_hardneg_sanity_20260628/best.pt`
- best epoch：4
- best val loss：1.536592

## 观察

- 训练可以完整跑完，说明 P2D 新增的 warmup 代码链路可用。
- train loss 整体下降，val loss 在 1.52-1.64 间波动，5 epoch sanity 没有发现阻塞问题。
- 日志中出现一次 NPU AMP 的 non-finite check/unscale warning，但训练继续完成且 loss 没有发散。正式训练时继续观察是否反复出现。

## 结论

P2D sanity 通过。下一步进入 6 卡正式训练：

- 使用 NPU 0-5。
- 从 P2A best checkpoint 初始化。
- 使用 `configs/v2_p2d_delayed_hardneg_full_20260628.yaml`。
- 训练完成后按固定 post-training scale 流程导出 embedding、跑变化检测和 OSM 下游任务、生成可视化与对比报告。
