# P2A vs P2A-Long 海淀区下游任务对比（2026-06-28）

本文档只比较海淀区任务，用来回答：如果后续主要服务海淀区，P2A 原版和 P2A-Long 哪个 embedding 更适合作为底座。

## 评测范围

- P2A：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2a_semantic_probe_full_quick_20260627_185700`
- P2A-Long：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p2a_long_lr5e5_full_quick_20260628_090032`
- OSM 下游任务只取海淀：`haidian_building_osm`、`haidian_road_osm`
- 施工任务只取海淀：`construction`
- `construction_joint` 使用已有预测结果，按 `haidian_` patch 前缀过滤，只在海淀子集上重新计算指标。

不纳入 `building_change`、`farm_change`、`rubbish`，因为这些任务当前主要是哈尔滨标签，不能作为海淀专测。

## 评测方法

1. 两个模型使用各自已导出的 embedding 和已完成的 quick fold-0 下游结果。
2. OSM 建筑/道路分别比较 Linear、MLP、U-Net 三类下游头。
3. `construction` 使用当前 benchmark head 的 fold-0 指标。
4. `construction_joint` 从 `fold_0/predictions/*_prob.tif` 中筛选 `haidian_*` 样本，使用 `/data/xuannv_embedding/processed/construction_joint_v2/masks` 中对应 GT mask 重新计算 F1、mIoU、Precision、Recall、AP、AUC。
5. 胜负以 `F1_best` 为主，差值绝对值小于 0.005 视为基本持平。

`construction_joint` 海淀子集重算结果保存于：

`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/haidian_only_region_metrics_20260628`

## 总览

| 任务组 | P2A F1 | P2A-Long F1 | 差值 | 判断 |
|---|---:|---:|---:|---|
| 施工类平均（construction + construction_joint 海淀子集） | 0.3330 | 0.3070 | -0.0261 | P2A 更好 |
| OSM Linear 平均 | 0.2923 | 0.3074 | +0.0151 | P2A-Long 更好 |
| OSM MLP 平均 | 0.2960 | 0.3053 | +0.0093 | P2A-Long 略好 |
| OSM U-Net 平均 | 0.4358 | 0.4150 | -0.0208 | P2A 更好 |
| 海淀任务整体平均 | 0.3393 | 0.3337 | -0.0056 | P2A 略好 |

## 详细指标

| 任务 | 下游头 | P2A F1 | P2A-Long F1 | 差值 | 胜者 | P2A mIoU | Long mIoU | P2A P | Long P | P2A R | Long R | P2A AP | Long AP | P2A AUC | Long AUC |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 海淀施工地 construction | benchmark-head | 0.2978 | 0.2600 | -0.0378 | P2A | 0.1593 | 0.1490 | 0.2474 | 0.2071 | 0.3091 | 0.3469 | 0.2207 | 0.1709 | 0.8744 | 0.8336 |
| 海淀 construction_joint 子集 | benchmark-head | 0.3683 | 0.3540 | -0.0143 | P2A | 0.2256 | 0.2150 | 0.3555 | 0.3709 | 0.3819 | 0.3384 | 0.3878 | 0.3626 | 0.9267 | 0.9175 |
| 海淀建筑 OSM | linear | 0.2894 | 0.2997 | +0.0103 | P2A-Long | 0.1667 | 0.1718 | 0.2023 | 0.2094 | 0.4860 | 0.4892 | 0.1787 | 0.1875 | 0.7506 | 0.7657 |
| 海淀道路 OSM | linear | 0.2953 | 0.3152 | +0.0199 | P2A-Long | 0.1715 | 0.1851 | 0.1770 | 0.1935 | 0.8470 | 0.8100 | 0.1744 | 0.1883 | 0.5675 | 0.6020 |
| 海淀建筑 OSM | mlp | 0.2897 | 0.2918 | +0.0021 | Tie | 0.1688 | 0.1673 | 0.1997 | 0.2055 | 0.5222 | 0.4736 | 0.1859 | 0.1907 | 0.7529 | 0.7628 |
| 海淀道路 OSM | mlp | 0.3024 | 0.3188 | +0.0164 | P2A-Long | 0.1741 | 0.1869 | 0.2155 | 0.2288 | 0.4754 | 0.5051 | 0.2177 | 0.2285 | 0.6037 | 0.6272 |
| 海淀建筑 OSM | unet | 0.4225 | 0.3938 | -0.0286 | P2A | 0.2611 | 0.2345 | 0.3722 | 0.3320 | 0.4666 | 0.4441 | 0.3700 | 0.3239 | 0.8695 | 0.8568 |
| 海淀道路 OSM | unet | 0.4491 | 0.4362 | -0.0129 | P2A | 0.2886 | 0.2754 | 0.3974 | 0.4030 | 0.5130 | 0.4652 | 0.4352 | 0.4235 | 0.7811 | 0.7664 |

## 结论

如果只看海淀区，结论不是简单的一边倒：

- P2A-Long 在简单下游头上更好，尤其是 OSM Linear 和 MLP 的海淀建筑/道路任务。
- P2A 在施工类任务上更好，包括海淀 `construction` 和 `construction_joint` 海淀子集。
- P2A 在 U-Net 强下游头上也更好，建筑和道路两个海淀 OSM 任务都领先。
- 海淀任务整体平均 F1 为 P2A `0.3393`，P2A-Long `0.3337`，P2A 略高。

因此，当前建议是：

1. 如果目标是汇报“embedding 本身被简单模型读出来的能力”，可以展示 P2A-Long 在海淀 Linear/MLP OSM 上的提升。
2. 如果目标是实际下游效果、施工地检测、以及更强分割头效果，仍推荐 P2A 原版作为底座。
3. 下一轮升级不建议继续单纯加训练轮次，而应保留 P2A 的空间语义能力，同时吸收 P2A-Long 在简单头线性可分性上的收益。

## 可视化

P2A 与 P2A-Long 的海淀 OSM 对比图已生成在：

`/data/xuannv_embedding/experiments/v2_202512_202605/comparisons/p2a_vs_p2a_long_20260628`

重点查看：

- `haidian_building_osm_unet_p2a_vs_p2a_long.png`
- `haidian_road_osm_unet_p2a_vs_p2a_long.png`

