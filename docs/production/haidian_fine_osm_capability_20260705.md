# 海淀生产版细粒度 OSM 能力评测记录

日期：2026-07-05

## 目的

在建筑、道路、水体、施工地之外，继续从 OSM 中补充更细的弱标签，评测海淀生产版 embedding 是否能支持更丰富的下游地物制图任务。

## 标签来源

本轮对本地缺失的 OSM 标注使用在线 OSM 数据补充缓存，再统一栅格化为 128x128 mask。

新增评测类别：

- park
- garden
- pitch
- sports
- school
- university
- hospital
- parking
- forest
- grass
- retail
- research_gov

未纳入本轮稳定评测：

- cemetery：正样本 patch 只有 22 个，过于稀疏，暂不作为公平对比结论。

OSM 缓存与标签输出：

- cache：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_osm_fine_labels/cache`
- mask/split：`/data/xuannv_embedding/processed/haidian/labels`

## 评测设置

对比模型：

- Xuannv Haidian v1：P10C epoch800 生产版 embedding
- AEF annual 2025：官方年度 embedding

公平设置：

- 同一批 OSM mask
- 同一 fold0 划分
- 同一 downstream head：linear、MLP
- 同一 shot：10、50、full
- 同一训练轮数：50 epoch
- 同一指标：F1、AP、mIoU

结果目录：

- suite：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_fold0`
- 对比表：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_fold0/paired_comparison.csv`
- 可视化：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full`

## 总体结果

共 72 组成对比较。

- F1 胜出：38/72
- AP 胜出：45/72
- mIoU 胜出：38/72

按类别平均，Xuannv 更强的类别：

- pitch：F1 +0.0363，AP +0.0335
- grass：F1 +0.0197，AP +0.0509
- university：F1 +0.0185，AP +0.0157
- research_gov：F1 +0.0161，AP +0.0065
- sports：F1 +0.0156，AP +0.0048
- parking：F1 +0.0076，AP +0.0043

AEF 更强的类别：

- park：F1 -0.0333，AP -0.0253
- forest：F1 -0.0238，AP -0.0494
- hospital：F1 -0.0179，AP -0.0054
- retail：F1 -0.0175，AP -0.0056
- school：F1 -0.0103，AP -0.0003
- garden：F1 -0.0053，AP -0.0017

## MLP full-shot 重点结论

这一档代表给下游头较充分的弱标签样本后，embedding 的可线性/浅层可分性。

Xuannv 胜出：

- pitch：F1 0.3528 vs AEF 0.2854
- sports：F1 0.1707 vs AEF 0.1135
- university：F1 0.2300 vs AEF 0.1732
- grass：F1 0.0598 vs AEF 0.0149
- research_gov：F1 0.1050 vs AEF 0.0704
- forest：F1 0.7942 vs AEF 0.7694
- school：F1 0.1756 vs AEF 0.1686

AEF 胜出：

- park：F1 0.4676 vs AEF 0.4782
- hospital：F1 0.0797 vs AEF 0.1011
- parking：F1 0.0474 vs AEF 0.0749
- garden：F1 0.0240 vs AEF 0.0632
- retail：F1 0.0777 vs AEF 0.1353

## 解读

这轮结果说明，海淀生产版 embedding 对运动场、校园/高校、科研政务、草地等更细粒度语义已经有可用表达，特别是在 MLP full-shot 下表现明显。

短板主要集中在 park、forest、retail、hospital、garden 这些边界定义更依赖 OSM 面状语义、且视觉外观内部差异较大的类别。后续如果要继续提升，应优先检查这些类别的 OSM 合并规则、标签噪声、类别边界，以及下游阈值选择。

