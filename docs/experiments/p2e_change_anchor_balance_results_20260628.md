# P2E change-anchor balance 结果报告（2026-06-28）

## 这一轮改了什么

P2E 从 P2A best checkpoint 初始化，只做一个主改动：把训练重心往变化检测锚点拉回去。

具体来说：

- 加强 `supervised_change_alignment_loss`。
- 把 `construction` 加入 supervised change tasks。
- 降低 semantic probe 权重。
- 关闭 P2D 的 semantic hard-negative。

通俗地说，P2D 是“更努力学建筑/道路背景别误检”，P2E 是“学建筑/道路时不要忘记两期影像哪里真的变了”。

## 关键路径

- 配置：`configs/v2_p2e_change_anchor_balance_full_20260628.yaml`
- 初始化权重：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- P2E checkpoint：`/data/xuannv_embedding/outputs/v2_p2e_change_anchor_balance_full_20260628/best.pt`
- P2E embedding：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260628_v2_p2e_change_anchor_balance_full_20260628_best_p2e_change_anchor_balance_full_best_20260628`
- 变化检测测评：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2e_change_anchor_balance_full_quick_20260628_055741`
- OSM 下游测评：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2e_change_anchor_balance_full_quick_20260628_055741`
- 变化检测可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2e_change_anchor_balance_full_quick_20260628_055741/visualizations`
- OSM linear 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2e_change_anchor_balance_full_quick_20260628_055741/diagnostics_linear`
- OSM MLP 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2e_change_anchor_balance_full_quick_20260628_055741/diagnostics_mlp`
- OSM U-Net 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2e_change_anchor_balance_full_quick_20260628_055741/diagnostics_unet`

## 训练状态

- 设备：NPU 0-5，6 卡 DDP
- epoch：100
- final val loss：1.505789
- embedding 导出数量：8928 个 `.pt`

训练 loss/val loss 比 P2D 更低，但下游指标说明“训练 loss 更低”不等于“embedding 更好”。

## OSM simple probe 对比（F1_best）

| version | head | haidian building | haidian road | harbin building | harbin road | avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P1B | linear | 0.3057 | 0.3176 | 0.1373 | 0.1887 | 0.2373 |
| P1B | MLP | 0.2938 | 0.3020 | 0.2860 | 0.3183 | 0.3000 |
| P2A | linear | 0.2894 | 0.2953 | 0.1752 | 0.2254 | 0.2463 |
| P2A | MLP | 0.2897 | 0.3024 | 0.4044 | 0.3648 | 0.3403 |
| P2A | U-Net | 0.4225 | 0.4491 | 0.6121 | 0.5469 | 0.5077 |
| P2C | linear | 0.3118 | 0.3221 | 0.1500 | 0.2322 | 0.2540 |
| P2C | MLP | 0.2901 | 0.3243 | 0.3792 | 0.3492 | 0.3357 |
| P2C | U-Net | 0.4323 | 0.4402 | 0.6314 | 0.5215 | 0.5063 |
| P2D | linear | 0.3128 | 0.3216 | 0.1509 | 0.2333 | 0.2546 |
| P2D | MLP | 0.2911 | 0.3243 | 0.3841 | 0.3551 | 0.3386 |
| P2D | U-Net | 0.4395 | 0.4524 | 0.6294 | 0.5381 | 0.5148 |
| P2E | linear | 0.3132 | 0.3234 | 0.1462 | 0.2283 | 0.2528 |
| P2E | MLP | 0.2916 | 0.3259 | 0.3589 | 0.3559 | 0.3331 |
| P2E | U-Net | 0.4300 | 0.4698 | 0.6043 | 0.5318 | 0.5090 |

结论：

- P2E 的 Haidian road 变好，尤其 U-Net 到 0.4698。
- 但 Harbin building 明显掉，MLP 从 P2A 的 0.4044 降到 0.3589。
- P2E 的 simple probe 平均低于 P2D，不能算 embedding 变好。

## P2E OSM 详细指标

| head | task | F1@0.5 | F1@val_threshold | F1_best | mIoU | AUPRC | pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | haidian_building_osm | 0.1601 | 0.3080 | 0.3132 | 0.1820 | 0.2008 | 0.2090 |
| linear | haidian_road_osm | 0.2736 | 0.3199 | 0.3234 | 0.1904 | 0.1905 | 0.6572 |
| linear | harbin_building_osm | 0.0448 | 0.1436 | 0.1462 | 0.0773 | 0.0674 | 0.3539 |
| linear | harbin_road_osm | 0.1360 | 0.2277 | 0.2283 | 0.1285 | 0.1366 | 0.2426 |
| MLP | haidian_building_osm | 0.1588 | 0.2899 | 0.2916 | 0.1695 | 0.1902 | 0.2168 |
| MLP | haidian_road_osm | 0.2736 | 0.3215 | 0.3259 | 0.1916 | 0.2353 | 0.3449 |
| MLP | harbin_building_osm | 0.2299 | 0.3587 | 0.3589 | 0.2185 | 0.2879 | 0.0590 |
| MLP | harbin_road_osm | 0.2539 | 0.3528 | 0.3559 | 0.2142 | 0.2789 | 0.1494 |
| U-Net | haidian_building_osm | 0.2265 | 0.4266 | 0.4300 | 0.2712 | 0.3846 | 0.1167 |
| U-Net | haidian_road_osm | 0.2736 | 0.4676 | 0.4698 | 0.3052 | 0.4672 | 0.1833 |
| U-Net | harbin_building_osm | 0.3329 | 0.5988 | 0.6043 | 0.4273 | 0.6303 | 0.0612 |
| U-Net | harbin_road_osm | 0.2127 | 0.5259 | 0.5318 | 0.3567 | 0.5111 | 0.1058 |

## 变化检测对比（F1_best）

| version | construction | building_change | farm_change | rubbish | construction_joint | avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1B | 0.3696 | 0.2732 | 0.0534 | 0.5359 | 0.3420 | 0.3148 |
| P2A | 0.2978 | 0.3229 | 0.1421 | 0.5971 | 0.3371 | 0.3394 |
| P2C | 0.2650 | 0.3033 | 0.1599 | 0.5581 | 0.3608 | 0.3294 |
| P2D | 0.2659 | 0.3028 | 0.1594 | 0.5597 | 0.3591 | 0.3294 |
| P2E | 0.2656 | 0.3141 | 0.0563 | 0.5332 | 0.4596 | 0.3258 |

结论：

- P2E 把 `construction_joint` 从 P2D 的 0.3591 大幅提升到 0.4596。
- `building_change` 也比 P2D 好，从 0.3028 到 0.3141。
- 但是 `farm_change` 从 P2D 的 0.1594 掉到 0.0563，几乎退回 P1B 水平。
- `rubbish` 从 P2A/P2D 下降到 0.5332。
- 平均变化检测不如 P2A/P2D，因此不能晋升。

## P2E 变化检测详细指标

| task | F1@0.5 | F1@val_threshold | F1_best | mIoU | AUPRC | pred+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construction | 0.0000 | 0.2644 | 0.2656 | 0.1523 | 0.1943 | 0.0804 |
| building_change | 0.2113 | 0.1689 | 0.3141 | 0.0922 | 0.1950 | 0.0130 |
| farm_change | 0.0324 | 0.0000 | 0.0563 | 0.0000 | 0.0154 | 0.0002 |
| rubbish | 0.2532 | 0.4847 | 0.5332 | 0.3198 | 0.5707 | 0.0013 |
| construction_joint | 0.1772 | 0.3964 | 0.4596 | 0.2472 | 0.4482 | 0.0541 |

## 和 AEF 对比

P2E quick fold0 的变化检测 macro 指标仍高于 AEF 5-fold 均值：

| metric | P2E | AEF |
| --- | ---: | ---: |
| AUC ROC | 0.9378 | 0.7848 |
| F1_best | 0.3258 | 0.1305 |
| F1@0.5 | 0.1348 | 0.0846 |
| mIoU | 0.1623 | 0.0491 |

但这不能掩盖 P2E 相对 P2A/P2D 的退化，尤其 farm_change 和 simple probe。

## 可视化观察

已生成：

- 变化检测可视化：20 张 strip。
- OSM linear/MLP/U-Net 诊断图：各 24 张。

主要问题：

- `farm_change` 的 pred+ 只有 0.0002，说明模型几乎不预测农田变化正类。
- `construction` 的 F1@0.5 为 0，仍严重依赖阈值校准。
- `haidian_road_osm` linear pred+ 为 0.6572，背景误检风险仍很高。
- P2E 加强变化锚点后，部分变化任务提升，但简单语义 probe 受到挤压，尤其 Harbin building。

## 是否晋升 baseline

不晋升。

理由：

1. simple probe 平均没有超过 P2D/P2A。
2. 变化检测平均低于 P2A，也低于 P2D。
3. farm_change 明显退化，不满足“变化检测明显退化不能算成功”。

当前 baseline 仍建议保持 P2A。

## 下一轮建议：P2F task-balanced change anchor

P2E 证明“加强变化锚点”这个方向不是完全错的，因为 `construction_joint` 和 `building_change` 提升了。但全局加大 change anchor 会伤害 `farm_change` 和 Harbin building simple probe。

下一轮不要继续整体加权，而是做 task-balanced change anchor：

1. 从 P2A 初始化。
2. supervised change 总权重不要再升高，回到 0.06-0.08。
3. 单独给 `construction_joint` 较高权重。
4. 降低对 `farm_change` 的负样本压制，避免它几乎不预测正类。
5. semantic probe 回到 P2A 的 0.08，保住 Harbin building/road 的 MLP 可读性。
6. 不启用 semantic hard-negative。

通俗地说，P2E 是“所有变化任务一起加力”，结果有的任务被照顾到了，有的被压坏了。P2F 要改成“按任务单独调力气”，不能一刀切。
