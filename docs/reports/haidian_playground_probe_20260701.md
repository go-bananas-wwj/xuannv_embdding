# 海淀区操场/运动场识别下游头实验

## 实验目的

使用 OSM 弱标签提取海淀区操场/运动场样本，并基于 P2A 版 embedding 训练一个轻量下游头，初步验证当前 embedding 对“操场/运动场”类别的可分性。

## 数据与标签

- 区域：海淀区，320 个 128x128 patch。
- embedding：`/data/xuannv_embedding/modelscope_upload/haidian/v1/api_ready/data/haidian/embeddings/v1/202605`
- OSM 弱标签：`/data/xuannv_embedding/processed/haidian/labels/osm_playground/masks`
- 标签口径：`leisure=pitch/track/stadium/sports_centre/playground`，以及 `landuse=recreation_ground`。
- 标签覆盖：320 个 patch 中 210 个 patch 有正样本，平均正样本像素比例 1.19%。

## 训练设置

- 下游头：像素级 linear probe，`StandardScaler + SGDClassifier(log_loss)`。
- 划分：按 patch 做 80/20 划分，训练 256 个 patch，测试 64 个 patch。
- 训练采样：正样本像素 45,238，负样本像素 219,904。
- 标签性质：OSM 弱标签，不是人工精标 GT，结果只用于快速诊断和展示潜力。

## 测评结果

| 指标 | 数值 |
|---|---:|
| ROC-AUC | 0.8287 |
| AP | 0.0725 |
| Best F1 | 0.1504 |
| Precision | 0.1277 |
| Recall | 0.1828 |
| IoU | 0.0813 |
| 最优阈值 | 0.9260 |

## 结论

当前 P2A embedding 对操场/运动场有一定排序能力，ROC-AUC 达到 0.829，说明“更像操场”的区域在特征空间中并非完全不可分。但精确分割指标仍偏低，AP、F1、IoU 都不高，说明只靠一个简单线性头还难以稳定落到具体边界。

主要原因可能有三点：

1. 操场/运动场在 10m 等效分辨率下目标较小，且形状细碎。
2. OSM 标签本身是弱标签，存在漏标、边界粗糙、点位缓冲误差。
3. 该类别强依赖上下文和形状结构，单像素 linear probe 不容易表达“跑道/球场/校园场景”这类空间模式。

## 输出位置

实验目录：

`/data/xuannv_embedding/experiments/haidian_playground_probe_20260701/month_202605`

关键文件：

- `playground_linear_probe.joblib`
- `summary.json`
- `haidian_playground_osm_gt_full_domain.png`
- `haidian_playground_pred_full_domain.png`
- `haidian_playground_probability_full_domain.png`
- `haidian_playground_probe_samples.png`
- `archives/month_202605_summary.zip`

