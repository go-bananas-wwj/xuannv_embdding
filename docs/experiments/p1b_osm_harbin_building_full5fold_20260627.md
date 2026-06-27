# P1B OSM 哈尔滨建筑提取 Full 5-Fold - 2026-06-27

本实验使用 OSM `building=*` 自动标签作为 `osm_gt`，冻结 P1B embedding，只训练轻量下游二分类分割头，验证无需人工标注时的建筑提取能力。

## 输入版本

- P1B embedding：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- Label root：`/data/xuannv_embedding/processed/harbin/labels/building_osm`
- Label kind：`osm_gt`
- 训练配置：`downstreams/configs/v2_acceptance_quick_concat_diff.yaml`
- 输入模式：`concat_diff = [202512 embedding, 202605 embedding, abs(diff)]`
- 输出目录：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_full5fold_20260627/harbin_building_osm`
- 可视化目录：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_full5fold_20260627/harbin_building_osm_contact_sheets`

## Full 5-Fold 指标

| Metric | Mean | Fold values |
|---|---:|---|
| AUC | 0.9470 | 0.9515, 0.9426, 0.9450, 0.9510, 0.9449 |
| F1_best | 0.3775 | 0.4283, 0.3845, 0.2849, 0.4581, 0.3317 |
| F1@0.5 | 0.2074 | 0.2446, 0.1889, 0.1422, 0.2896, 0.1716 |
| F1@val_thr | 0.3612 | 0.4129, 0.3437, 0.2838, 0.4535, 0.3123 |
| mIoU | 0.2223 | 0.2601, 0.2075, 0.1654, 0.2933, 0.1850 |

## 结论

- 结果稳定成立：5 折 AUC 都在 `0.94+`，说明 P1B embedding 对建筑区域有很强排序/区分能力。
- F1_best 均值 `0.3775`，mIoU 均值 `0.2223`，可以作为“OSM 自动标签 + P1B embedding 建筑提取”的正式汇报数字。
- 默认阈值 `0.5` 明显偏低，验证集阈值后 F1 从 `0.2074` 提升到 `0.3612`，后续汇报和产品化必须带 per-task calibration。
- fold 2 和 fold 4 较弱，说明 OSM 标签空间分布/漏标可能有 fold 差异；需要用 contact sheet 抽检标签质量。

## 可视化

已生成 6 张 contact sheet，每张包含：高分影像、P1B embedding PCA、预测概率、OSM GT、预测叠加、GT 叠加。

索引：`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_full5fold_20260627/harbin_building_osm_contact_sheets/index.md`

## 下一步

1. 将 `harbin_building_osm` 加入 P1B 汇报包能力矩阵。
2. 对 `haidian_building_osm` 跑完整 5-fold，形成双区域建筑提取结果。
3. 对道路任务补 relaxed F1 / skeleton recall 后再跑完整 5-fold。
4. 抽检 OSM 漏标区域，如果漏标明显，再接 LocateAnything bbox/point + SAM2 mask 伪标签补召回。
