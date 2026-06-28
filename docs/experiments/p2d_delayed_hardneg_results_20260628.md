# P2D 延迟 hard-negative 结果报告（2026-06-28）

## 这一轮改了什么

P2D 从 P2A best checkpoint 初始化，保留 P2C 的 semantic hard-negative 思路，但给 hard-negative 加了延迟 warmup：

- P2C：一开始就压低“背景里模型最容易误判成建筑/道路”的像素。
- P2D：先让模型稳定学习原来的 embedding，再逐步加 hard-negative 压力。

通俗地说，P2D 不是更用力地打模型，而是晚一点再打，避免一上来把变化检测和跨城市泛化打乱。

## 关键路径

- 配置：`configs/v2_p2d_delayed_hardneg_full_20260628.yaml`
- 初始化权重：`/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627/best.pt`
- P2D checkpoint：`/data/xuannv_embedding/outputs/v2_p2d_delayed_hardneg_full_20260628/best.pt`
- P2D embedding：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260628_v2_p2d_delayed_hardneg_full_20260628_best_p2d_delayed_hardneg_full_best_20260628`
- 变化检测测评：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2d_delayed_hardneg_full_quick_20260628_025857`
- OSM 下游测评：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2d_delayed_hardneg_full_quick_20260628_025857`
- 变化检测可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2d_delayed_hardneg_full_quick_20260628_025857/visualizations`
- OSM linear 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2d_delayed_hardneg_full_quick_20260628_025857/diagnostics_linear`
- OSM MLP 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2d_delayed_hardneg_full_quick_20260628_025857/diagnostics_mlp`
- OSM U-Net 可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p2d_delayed_hardneg_full_quick_20260628_025857/diagnostics_unet`

## 训练状态

- 设备：NPU 0-5，6 卡 DDP
- epoch：100
- best checkpoint：epoch 99
- final val loss：1.536761
- embedding 导出数量：8928 个 `.pt`

训练过程中第一次启动失败过一次，原因是 `PYTHONPATH` 覆盖了 CANN/TBE 路径，导致 `ModuleNotFoundError: No module named 'tbe'`。修正后通过 `source /usr/local/Ascend/cann-9.0.0/set_env.sh` 并追加项目路径，训练正常完成。

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

结论：

- P2D 比 P2C 略好，尤其 U-Net 平均从 0.5063 到 0.5148。
- linear 平均从 P2C 的 0.2540 小幅到 0.2546，是正向但幅度很小。
- P2D 仍没有恢复 P2A 的 Harbin MLP 表现，说明跨城市简单可读性还没有真正解决。

## P2D OSM 详细指标

| head | task | F1@0.5 | F1@val_threshold | F1_best | mIoU | AUPRC | pred+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | haidian_building_osm | 0.1599 | 0.3061 | 0.3128 | 0.1807 | 0.2031 | 0.1918 |
| linear | haidian_road_osm | 0.2736 | 0.3189 | 0.3216 | 0.1897 | 0.1914 | 0.6465 |
| linear | harbin_building_osm | 0.0438 | 0.1505 | 0.1509 | 0.0814 | 0.0709 | 0.3031 |
| linear | harbin_road_osm | 0.1355 | 0.2312 | 0.2333 | 0.1307 | 0.1436 | 0.2906 |
| MLP | haidian_building_osm | 0.1588 | 0.2880 | 0.2911 | 0.1682 | 0.1919 | 0.2088 |
| MLP | haidian_road_osm | 0.2736 | 0.3217 | 0.3243 | 0.1917 | 0.2326 | 0.3687 |
| MLP | harbin_building_osm | 0.2333 | 0.3832 | 0.3841 | 0.2370 | 0.3210 | 0.0563 |
| MLP | harbin_road_osm | 0.2512 | 0.3526 | 0.3551 | 0.2140 | 0.2845 | 0.1482 |
| U-Net | haidian_building_osm | 0.2257 | 0.4276 | 0.4395 | 0.2719 | 0.4047 | 0.0936 |
| U-Net | haidian_road_osm | 0.2736 | 0.4490 | 0.4524 | 0.2895 | 0.4495 | 0.1751 |
| U-Net | harbin_building_osm | 0.3463 | 0.6286 | 0.6294 | 0.4584 | 0.6526 | 0.0519 |
| U-Net | harbin_road_osm | 0.2083 | 0.5369 | 0.5381 | 0.3670 | 0.5237 | 0.0921 |

## 变化检测对比（F1_best）

| version | construction | building_change | farm_change | rubbish | construction_joint | avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1B | 0.3696 | 0.2732 | 0.0534 | 0.5359 | 0.3420 | 0.3148 |
| P2A | 0.2978 | 0.3229 | 0.1421 | 0.5971 | 0.3371 | 0.3394 |
| P2B | 0.2576 | 0.3081 | 0.0131 | 0.5677 | 0.3477 | 0.2988 |
| P2C | 0.2650 | 0.3033 | 0.1599 | 0.5581 | 0.3608 | 0.3294 |
| P2D | 0.2659 | 0.3028 | 0.1594 | 0.5597 | 0.3591 | 0.3294 |

结论：

- P2D 和 P2C 变化检测几乎持平。
- P2D 仍明显弱于 P2A 的平均变化检测 0.3394。
- 因为目标要求“变化检测明显退化不能算成功”，P2D 不能晋升为新的总 baseline。

## P2D 变化检测详细指标

| task | F1@0.5 | F1@val_threshold | F1_best | mIoU | AUPRC | pred+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| construction | 0.0000 | 0.2653 | 0.2659 | 0.1529 | 0.1830 | 0.0754 |
| building_change | 0.1932 | 0.1778 | 0.3028 | 0.0976 | 0.1582 | 0.0100 |
| farm_change | 0.1470 | 0.0000 | 0.1594 | 0.0000 | 0.0492 | 0.0004 |
| rubbish | 0.2449 | 0.4235 | 0.5597 | 0.2686 | 0.5900 | 0.0020 |
| construction_joint | 0.1237 | 0.3545 | 0.3591 | 0.2154 | 0.3204 | 0.0479 |

## 可视化观察

已生成两类可视化：

- 变化检测 strip：高分影像、embedding/PCA、预测、阈值结果、GT、TP/FP/FN。
- OSM 诊断图：每个任务按 linear、MLP、U-Net 分开输出，预测用红色展示，背景白色，右侧带 GT。

重点问题：

- linear/MLP 仍存在“阈值敏感”：F1@0.5 明显低于 F1_best，说明概率校准仍不好。
- 道路的 linear pred+ 偏高，尤其 haidian road linear 预测正类比例 0.6465，仍有大片背景被当成道路的风险。
- farm_change 的 mIoU 为 0，pred+ 只有 0.0004，说明这个任务仍接近“几乎不预测正类”，不能被高 AUC 掩盖。

## 是否晋升 baseline

不晋升。

原因：

1. P2D 的 OSM simple probe 比 P2C 有小幅提升，但幅度很小。
2. P2D 的变化检测没有恢复到 P2A 水平。
3. 变化检测中的 F1@0.5、pred+、mIoU 暴露了校准和正类预测不足问题。

当前推荐：

- 总 baseline 继续使用 P2A。
- P2D 作为“延迟 hard-negative 有小幅帮助，但不足以解决核心矛盾”的参考实验。

## 下一轮建议：P2E change-anchor semantic balance

下一轮不要继续单纯调 hard-negative。P2D 说明 hard-negative 能稍微帮简单语义头，但会把变化检测压住。

建议 P2E 只做一个主改动：加入 change-anchor balance。

通俗解释：

- 现在模型学语义时，会更关注“哪里像建筑/道路”。
- 但变化检测需要它同时记住“两期影像哪里真的变了”。
- P2E 要做的是：训练时给变化区域一个保护锚点，让语义学习不能把变化敏感性冲掉。

具体做法：

1. 从 P2A 初始化，而不是从 P2D 初始化。
2. 保留 P2A 的 semantic probe。
3. hard-negative 只保留很小权重或暂时关掉。
4. 对有变化标签的 patch，提高 change loss 权重。
5. 在 embedding 层增加一个轻量 change-consistency/prototype 约束：变化区两期 embedding 差异要更明显，非变化区两期 embedding 要更稳定。
6. 继续用同一套 quick eval 判断：只有 simple probe 和变化检测同时提升，才允许晋升 baseline。
