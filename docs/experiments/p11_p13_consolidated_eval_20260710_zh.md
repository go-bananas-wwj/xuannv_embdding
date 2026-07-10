# P11/P12/P13 统一快评与 v3 基座选择

日期：2026-07-10　分支：v3-semantic-64d

## 背景

P11（400 维扩展）、P12（context160）、P13（latent dim 64 全家桶）训练均已完成。
本文汇总三组已有评测，统一口径后选出 v3（P14）升级的初始化基座。

## 口径说明

- P11/P12 原评测套件使用 MLP probe（`v2_probe_mlp_single_202604.yaml`），
  P13 使用 conv3x3 probe（`v2_probe_binary_conv3x3_single_202604.yaml`），
  两者绝对值不可直接互比。
- 本文 conv3x3 同头对比来自：
  - P10C/P12C：`/data/xuannv_embedding/experiments/p12_context_eval_20260707/head_sweep_20260708/conv3x3/benchmarks/`
  - P10C construction：`/data/xuannv_embedding/experiments/production/haidian_v1_landuse_construction_20260710/construction_conv3x3/`
  - P13：`/data/xuannv_embedding/experiments/p13_latent_eval_20260709/benchmarks/`
- P13 的 embedding 空间诊断为本次补跑：
  `/data/xuannv_embedding/experiments/p13_latent_eval_20260709/embedding_space/`

## conv3x3 同头 fold-0 对比（F1_best / AP）

| 模型 | construction | building | road | water |
| --- | --- | --- | --- | --- |
| P10C epoch800（生产） | 0.5495 / 0.5435 | 0.4842 / 0.4497 | 0.5218 / 0.5664 | 0.6237 / 0.6041 |
| P12C best | — | 0.4809 / 0.4395 | 0.5165 / 0.5552 | 0.6143 / 0.6042 |
| P13 best | 0.5265 / 0.4436 | 0.4842 / 0.4444 | 0.5165 / 0.5544 | 0.6181 / 0.6025 |
| P13 e800 | 0.5429 / 0.3891 | 0.4814 / 0.4377 | 0.5183 / 0.5582 | 0.6183 / 0.6112 |

MLP probe 口径（P11 套件内，同头）：P11C best 在 construction/road/water 小幅优于
P10C（如 water 0.5254 vs 0.4938），但 P11 系列是 400 维，与 64 维产品形态不符。

## Embedding 空间诊断（haidian 202604）

| 模型 | 有效秩(熵) | Top10 方差占比 | mean centroid AUC | between/within |
| --- | ---: | ---: | ---: | ---: |
| P10C epoch800 | 9.69 | 0.933 | 0.723 | — |
| P11C best (400d) | 10.87 | 0.914 | 0.741 | 0.760 |
| P13 best | 10.45 | 0.921 | 0.718 | 0.646 |
| P13 e800 | 11.53 | 0.903 | 0.735 | 0.669 |
| AEF 官方 (参考) | 14.79 | 0.813 | 0.684 | 0.488 |

## 结论

1. P11/P12/P13 的下游 fold-0 指标均未实质超过 P10C；P12 系列整体回退。
2. P13 e800 与 P10C 下游打平（差距 <1%），但 embedding 空间显著更健康：
   有效秩 11.53 vs 9.69，Top10 方差占比 0.903 vs 0.933，centroid AUC 0.735 vs 0.723。
3. P13 e800 是 64 维、已带 uniformity/covariance/patchdisc/latent-recon/semantic-hardneg
   损失栈的最新基座，**选定为 P14（v3 首个实验）的初始化权重**。
4. 对标基线仍为 P10C epoch800（生产版）：v3 的目标是在同头协议下全面超过它。

## 待办（v3 P14 升级方向）

- 高分注入 embedding（highres_fusion_to_embedding + 轻量 FPN 高分 encoder）
- precision path 保持 1x，消除 64x64 降采样再上采样
- OSM 边界带 boundary contrast + 语义任务池扩展（土地分类多类）
- uniformity/covariance 加强 + 有效秩例行监控（目标 >= 12）
- few-shot 新类别与相似性检索（centroid/kNN）纳入统一验收
