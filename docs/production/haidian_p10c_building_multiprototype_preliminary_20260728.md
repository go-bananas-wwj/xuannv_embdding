# 海淀 P10C 建筑物多原型稀疏制图预实验

## 目的

定位 P10C 在原始 3 多边形 PU+Query 建筑物检索中假正例偏多的问题。该预实验不重训
embedding 底座，只改读取器，并与 AEF、传统多源特征使用完全相同的标注、划分和协议。

## 两套协议不可混用

| 协议 | 原始 P10C PU+Query | 本预实验 |
| --- | --- | --- |
| 前景表示 | 3 个建筑多边形平均为一个原型 | 保留 3 个原型，对任一原型相似即可 |
| 测试时 Query | 开启，每个测试 patch 可自适应 | 关闭，防止高分硬化地自我强化 |
| 阈值 | 支持多边形与可靠背景 | 仅由隔离验证集选择 F1 阈值 |
| 用途 | 交互式 PU+Query 原始基线 | 建筑物误报诊断与严格读取器预实验 |

原始结果保留在
[`haidian_pu_query_3polygon_comparison_20260726.md`](haidian_pu_query_3polygon_comparison_20260726.md)，
不能被本页数值替代或合并平均。

## 固定设置

- 特征：玄女 P10C epoch800 的 2026-04 月度嵌入、AEF 官方 2025 年度嵌入、2026-04 传统 42 维多源特征。
- 标注：同一批 3 个建筑 OSM 连通多边形，面积分别为 20、51、45 个 10 m 像素。
- 划分：fold 0，训练 231 个 patch、验证 25 个 patch、独立测试 64 个 patch。
- 公平性：三个特征源共用支持多边形、验证/测试 patch、PU 背景挖掘参数和多原型规则。
- 阈值：只能读取验证标签选取；测试标签只用于最终指标计算。

这是一项**同一读取器下的特征对比**，不是时间完全匹配或上游训练完全对等的主基准：
玄女为海淀 OSM 弱语义辅助训练的 2026-04 月度嵌入，AEF 为官方 2025 年度嵌入，传统基线为
2026-04 多源观测。评测标签同样来自 OSM，因此结果应表述为 OSM 辅助稀疏制图诊断，而非独立
人工真值上的严格归纳泛化结论。

## 结果

| 特征 | Precision | Recall | F1 | IoU | AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 玄女 P10C 月度嵌入 | 0.214 | 0.545 | **0.307** | **0.182** | **0.775** | 0.225 |
| AEF 年度嵌入 | **0.236** | 0.419 | 0.302 | 0.178 | 0.760 | **0.229** |
| 传统多源特征 | 0.191 | 0.434 | 0.265 | 0.153 | 0.717 | 0.180 |

相对原始 3 多边形 PU+Query 的玄女建筑 F1 `0.256`，本协议达到 `0.307`，主要来自
Precision 从 `0.150` 提升到 `0.214`。这支持“原始 Query 的自增强放大了建筑假正例”的诊断，
但目前仅覆盖一个 fold、一个随机种子，不能作为论文主结论。

## 可复现命令

```bash
PYTHONPATH=$PWD python scripts/eval/run_pu_query_sparse_eval.py \
  --tasks building \
  --prototype-mode max \
  --query-mode disabled \
  --threshold-mode validation_f1 \
  --output-root <data-output-root>/haidian_p10c_pu_query_building_multiprototype_valcal
```

结果记录了特征来源、支持/验证/测试 patch、支持阈值、验证阈值和 PU 超参数：
`<data-output-root>/haidian_p10c_pu_query_building_multiprototype_valcal/results.json`。

## 下一步

在不改动协议的前提下，扩展到 5 fold 与多个随机种子；同时保留原始 PU+Query 作为交互式检索
实验，以分开回答“测试时局部查询”和“固定稀疏标注读取器”两个问题。
