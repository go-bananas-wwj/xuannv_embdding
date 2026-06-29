# P3A OSM 弱语义标签生成结果

- 生成时间：2026-06-29
- 数据摘要：`/data/xuannv_embedding/processed/osm_weak_semantic_summary_20260629.json`
- 输出位置：`/data/xuannv_embedding/processed/{region}/labels/{task}/masks/*.tif`
- 范围：海淀 + 哈尔滨新区联合训练区域
- 用途：作为主 embedding 训练的弱语义辅助目标，不作为强真值；OSM 空白不作为强负样本。

## 生成类别

`osm_activity_poi`, `osm_agriculture`, `osm_building`, `osm_commercial`, `osm_construction`, `osm_green`, `osm_industrial`, `osm_major_road`, `osm_minor_road`, `osm_path_walk`, `osm_rail`, `osm_residential`, `osm_water`。

## 覆盖率摘要

| 区域 | 弱语义任务 | patch 数 | 有正样本 patch | 有正样本比例 | 平均正像素比例 | P95 正像素比例 | 最大正像素比例 |
|---|---|---:|---:|---:|---:|---:|---:|
| haidian | osm_activity_poi | 320 | 303 | 94.69% | 26.68% | 72.51% | 96.72% |
| haidian | osm_agriculture | 320 | 162 | 50.62% | 6.63% | 34.20% | 49.13% |
| haidian | osm_building | 320 | 290 | 90.62% | 9.60% | 33.09% | 42.88% |
| haidian | osm_commercial | 320 | 222 | 69.38% | 5.74% | 21.94% | 77.45% |
| haidian | osm_construction | 320 | 220 | 68.75% | 5.03% | 22.68% | 45.03% |
| haidian | osm_green | 320 | 319 | 99.69% | 35.20% | 100.00% | 100.00% |
| haidian | osm_industrial | 320 | 185 | 57.81% | 3.65% | 16.02% | 59.61% |
| haidian | osm_major_road | 320 | 274 | 85.62% | 8.93% | 18.99% | 26.20% |
| haidian | osm_minor_road | 320 | 314 | 98.12% | 10.60% | 20.56% | 27.58% |
| haidian | osm_path_walk | 320 | 291 | 90.94% | 5.39% | 14.97% | 26.95% |
| haidian | osm_rail | 320 | 157 | 49.06% | 1.76% | 6.98% | 19.88% |
| haidian | osm_residential | 320 | 257 | 80.31% | 13.67% | 39.76% | 100.00% |
| haidian | osm_water | 320 | 256 | 80.00% | 3.76% | 12.34% | 54.75% |
| harbin | osm_activity_poi | 424 | 187 | 44.10% | 6.88% | 39.17% | 80.09% |
| harbin | osm_agriculture | 424 | 45 | 10.61% | 0.72% | 3.49% | 41.05% |
| harbin | osm_building | 424 | 166 | 39.15% | 3.02% | 14.94% | 66.77% |
| harbin | osm_commercial | 424 | 116 | 27.36% | 1.58% | 9.92% | 36.24% |
| harbin | osm_construction | 424 | 126 | 29.72% | 2.36% | 14.99% | 38.64% |
| harbin | osm_green | 424 | 206 | 48.58% | 9.42% | 49.11% | 100.00% |
| harbin | osm_industrial | 424 | 146 | 34.43% | 4.25% | 26.88% | 49.43% |
| harbin | osm_major_road | 424 | 331 | 78.07% | 5.76% | 19.30% | 38.73% |
| harbin | osm_minor_road | 424 | 315 | 74.29% | 2.35% | 8.66% | 17.17% |
| harbin | osm_path_walk | 424 | 103 | 24.29% | 0.78% | 5.86% | 18.30% |
| harbin | osm_rail | 424 | 116 | 27.36% | 0.80% | 4.45% | 14.63% |
| harbin | osm_residential | 424 | 267 | 62.97% | 9.30% | 39.17% | 77.76% |
| harbin | osm_water | 424 | 273 | 64.39% | 19.83% | 91.37% | 100.00% |

## 训练使用建议

1. 优先接入：`osm_building`、`osm_major_road`、`osm_minor_road`、`osm_water`、`osm_residential`、`osm_commercial`、`osm_industrial`、`osm_agriculture`。
2. 低权重接入：`osm_activity_poi`、`osm_green`、`osm_path_walk`、`osm_rail`、`osm_construction`。其中 `osm_green` 和 `osm_water` 局部覆盖可能很大，训练时需要较低权重，避免模型被单一大面目标主导。
3. 不接入下游人工标注到主 embedding 训练，变化检测、建筑/道路/土地利用等任务只用于训练后测评。
4. 这一步解决的是“区域大概语义分布”，不是月度变化监督。P3A 主线仍然应以月度状态重建和跨月指数变化重建为核心。

## 下一步

运行训练数据审查，确认联合 manifest、S2/S1/Landsat/highres/WorldCover/OSM 弱标签在两个区域的样本数、通道、空间对齐和覆盖率没有异常，然后再启动新版 P3A 训练。
