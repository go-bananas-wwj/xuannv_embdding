# P3A OSM 弱语义 8 小时训练版下游快速测评（2026-06-30）

## 评测对象

- 模型权重：`/data/xuannv_embedding/outputs/v2_p3a_osm_weak_semantic_8h_20260629/best.pt`
- Embedding 导出：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260630_v2_p3a_osm_weak_semantic_8h_20260629_best_p3a_osm_weak_semantic_8h_best_quick`
- 下游测评：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p3a_osm_weak_semantic_8h_best_quick_20260630_011912`
- 评测方式：fold 0 quick，`concat_diff`，月份为 `202512` 和 `202605`

## 结论

P3A 这版不建议作为当前主力底座替换 P2A/P1B。它在 `construction` 和 `rubbish` 上有一定表现，macro F1_best 和 mIoU 表面上超过 AEF；但 `building_change`、`farm_change` 两个变化任务严重退化，AUC 也明显低于 AEF，说明长时间 OSM 弱语义训练削弱了变化敏感性。

## 指标

| 任务 | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | 主要判断 |
|---|---:|---:|---:|---:|---:|---|
| construction | 0.8473 | 0.2227 | 0.1429 | 0.1657 | 0.0903 | F1/mIoU 高于 AEF，但 AUC 略低 |
| building_change | 0.5251 | 0.0391 | 0.0058 | 0.0030 | 0.0015 | 变化检测明显退化 |
| farm_change | 0.5734 | 0.0031 | 0.0000 | 0.0001 | 0.0001 | 基本失效 |
| rubbish | 0.9081 | 0.3044 | 0.0655 | 0.1953 | 0.1082 | 明显优于 AEF，但仍依赖阈值校准 |
| construction_joint | 0.8428 | 0.2180 | 0.0581 | 0.2167 | 0.1215 | 明显低于 AEF |

## 与 AEF 的 Macro 对比

| 指标 | P3A | AEF | 差值 | 胜出方 |
|---|---:|---:|---:|---|
| AUC | 0.7393 | 0.7848 | -0.0454 | AEF |
| F1_best | 0.1575 | 0.1305 | +0.0270 | P3A |
| F1@0.5 | 0.0545 | 0.0846 | -0.0302 | AEF |
| mIoU | 0.0643 | 0.0491 | +0.0153 | P3A |

## 风险解读

- `building_change` 和 `farm_change` 的 AUC 只有 0.5251 和 0.5734，接近随机排序，说明 embedding 对真实时序变化的可分性不足。
- `building_change` 的 recall 很高但 precision 极低，预测大面积铺开，误报严重。
- `farm_change` 几乎没有有效召回，说明模型对耕地类变化没有保住之前版本的能力。
- P3A 的 OSM 弱语义训练更偏静态语义，可能强化了“哪里像建筑、道路、水体”等静态分布，但没有同步约束“同一地点跨月份状态变化”的表达。

## 输出文件

- AEF 对比：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p3a_osm_weak_semantic_8h_best_quick_20260630_011912/comparison_vs_aef.md`
- V1 验收对比：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p3a_osm_weak_semantic_8h_best_quick_20260630_011912/comparison_vs_v1.0.md`
- 阈值校准：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p3a_osm_weak_semantic_8h_best_quick_20260630_011912/threshold_calibration.md`
- 可视化索引：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p3a_osm_weak_semantic_8h_best_quick_20260630_011912/visualizations/index.md`

## 下一步建议

1. 不直接采用 P3A 作为新底座。
2. 保留 P3A 的 OSM 静态语义收益，但下一轮需要加回变化敏感约束，尤其是同 patch 跨月 embedding 差异、稳定区一致性、变化区可分性。
3. 优先对比 P2A/P1B/P3A 的可视化，确认 P3A 是否把两期 embedding 拉得过近，导致变化任务失效。
