# 海淀 V1 Embedding 空间诊断

## 背景

下游任务中出现较多假正例。一个合理怀疑是：64 维 embedding 中不同地物类别没有充分分开，甚至出现表征坍缩。本诊断使用海淀 OSM 弱标签抽样，对玄女海淀 V1 与 AEF official annual embedding 做同口径比较。

参考依据：

- AEF 论文明确将每个像素表示为 64 维、单位球面上的 embedding，并引入 batch uniformity objective 抑制坍缩。
- Google Earth Engine 的 AEF 教程也说明 AEF 用 64 个数压缩多源时序信息。
- 表征学习常用 alignment/uniformity、有效维度、类内/类间距离来判断特征是否有信息量和是否坍缩。

## 诊断命令

```bash
python scripts/report/diagnose_embedding_space.py \
  --samples-per-class 30000 \
  --background-samples 30000
```

输出目录：

```text
/data/xuannv_embedding/experiments/production/haidian_v1_embedding_space_diagnostics_20260706/
```

关键产物：

- `embedding_space_diagnostics.md`
- `embedding_space_summary.csv`
- `xuannv_centroid_auc.csv`
- `aef_centroid_auc.csv`
- `figures/summary_metrics.png`
- `figures/xuannv_pca_scatter.png`
- `figures/aef_pca_scatter.png`
- `figures/xuannv_centroid_distance.png`
- `figures/aef_centroid_distance.png`

## 核心结果

| 模型 | 有效维度 | Top10 方差占比 | 类间/类内距离比 | 类中心 AUC | 负样本靠近正类比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 玄女海淀 V1 | 9.7 | 0.933 | 0.71 | 0.723 | 0.631 |
| AEF annual 2025 | 14.8 | 0.812 | 0.47 | 0.683 | 0.721 |

解释：

- **没有发现完全坍缩**：玄女 embedding 范数为 1，死维度为 0，PCA 不是单点。
- **存在维度利用不足**：玄女有效维度约 9.7，Top10 主成分解释 93.3% 方差，说明 64 维里真正起主要作用的维度偏少。
- **类别边界仍然重叠**：道路、教育、运动场等类别的负样本靠近正类中心比例偏高，容易造成下游假正例。
- **AEF 更分散但未必在 OSM 弱标签上类中心 AUC 更高**：这说明 OSM 标签噪声、类别重叠和年度/月度时间差会影响这个诊断，不能只用单一指标判断模型优劣。

## 对假正例的解释

当前更像是“半坍缩/低有效维 + 语义边界不够锐”，不是所有类别坍缩成一个点。下游头看到的 embedding 中，很多背景像素与道路、教育、运动场等正类中心相似度较高，因此阈值稍微低一点就会出现大面积假正例。

## 下一步建议

1. **加强 batch uniformity / vMF uniformity**  
   目标是让 64 维更充分地铺开，降低 Top10 方差占比，提高有效维度。

2. **加入类别原型间隔约束，但只用弱语义 OSM**  
   不把下游精标标签喂给主模型，只用 OSM 合并后的弱语义类别做 prototype separation，让道路、建筑、水体、绿地等中心距离更远。

3. **加入 hard negative mining**  
   重点处理“背景很像正类”的像素，例如道路附近硬化地面、建筑屋顶与操场、校园建筑与普通建筑。

4. **训练后固定做 embedding-space health check**  
   每轮模型除了下游 F1，也必须报告有效维度、Top10 方差占比、类中心 AUC、负样本靠近正类比例，避免只看下游头掩盖 embedding 本身问题。

