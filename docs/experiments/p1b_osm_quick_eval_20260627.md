# P1B OSM 建筑/道路 Quick Downstream 结果 - 2026-06-27

本次实验使用 P1B embedding 冻结不变，只训练下游二分类分割头，验证 OSM 自动标签能否替代部分人工标注，用于建筑物和道路提取能力展示。

## 输入版本

- P1B embedding：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- 训练配置：`downstreams/configs/v2_acceptance_quick_concat_diff.yaml`
- 输入模式：`concat_diff = [202512 embedding, 202605 embedding, abs(diff)]`
- 训练方式：fold-0 quick benchmark，保存预测概率图。
- 输出目录：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_quick_20260627`

## 标签来源

| Region | Task | Label root | Label kind |
|---|---|---|---|
| haidian | building_osm | `/data/xuannv_embedding/processed/haidian/labels/building_osm` | `osm_gt` |
| haidian | road_osm | `/data/xuannv_embedding/processed/haidian/labels/road_osm` | `osm_gt` |
| harbin | building_osm | `/data/xuannv_embedding/processed/harbin/labels/building_osm` | `osm_gt` |
| harbin | road_osm | `/data/xuannv_embedding/processed/harbin/labels/road_osm` | `osm_gt` |

## Quick Fold-0 结果

| Task | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | Val threshold | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| haidian_building_osm | 0.8336 | 0.3590 | 0.2053 | 0.3525 | 0.2140 | 0.9743 | 25 |
| haidian_road_osm | 0.7219 | 0.3869 | 0.2736 | 0.3864 | 0.2394 | 0.9600 | 27 |
| harbin_building_osm | 0.9515 | 0.4283 | 0.2446 | 0.4129 | 0.2601 | 0.9528 | 25 |
| harbin_road_osm | 0.8423 | 0.3654 | 0.1802 | 0.3644 | 0.2228 | 0.9717 | 29 |

## 结论

- OSM 自动标签路线可行，四个任务都训练出了非平凡结果。
- 建筑物提取表现尤其值得继续推进：哈尔滨 `building_osm` 达到 AUC `0.9515`、F1_best `0.4283`、mIoU `0.2601`。
- 道路提取也有可学习信号，但普通 mIoU/F1 对道路细线不够友好，下一步应增加 relaxed F1 / skeleton recall。
- 所有任务的最佳工作阈值都很高，约 `0.95-0.97`，说明下游概率校准仍需要固化。
- `osm_gt` 不是人工精标，汇报时要明确：这是自动标签/可追溯弱监督，用来减少人工标注成本。

## 下一步

1. 为 OSM 建筑/道路生成可视化 contact sheet：高分影像、OSM GT、P1B 预测、embedding PCA。
2. 对四个任务跑完整 5-fold，优先 `harbin_building_osm` 和 `haidian_building_osm`。
3. 增加道路专用指标：relaxed F1、skeleton recall、buffer IoU。
4. 若 OSM 建筑或道路漏标明显，接入 LocateAnything bbox/point + SAM2 mask 伪标签补召回。
5. 把 OSM 建筑/道路加入 P1B 汇报包能力矩阵。
