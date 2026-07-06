# 玄女海淀 V1 传统机器学习对比评测

本轮评测用于回答一个更宽的问题：玄女 embedding 是否只是比 AEF 好，还是也能超过传统遥感特征加传统机器学习方法。

## 输出目录

```text
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_valselected_fulltest/
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_visuals/
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_report/
```

主要报告：

```text
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_report/haidian_v1_traditional_ml_benchmark_report.md
```

## 评测设置

- 区域：海淀区 320 个 patch。
- 月份：2026-04。
- 玄女模型：P10C epoch800 生产版 embedding。
- 任务：建筑、道路、水体、公园绿地、学校高校、运动场地。
- 标签：OSM 弱标签。
- 标签量：5-shot、10-shot、50-shot。
- 玄女特征：64 维 embedding。
- 传统特征：S2 光谱+指数、S2+S1+Landsat+指数。
- 传统/下游模型：Random Forest、ExtraTrees、Logistic Regression、HistGradientBoosting。
- 阈值：在验证集选择 F1-best 阈值，再在测试集汇报。
- 模型选择：同一 task/shot 内按验证集 `val_f1_best` 选玄女最佳模型和传统最佳 baseline，再汇报对应测试指标。
- 测试口径：完整验证/测试 patch 的全量像素指标；训练阶段按 patch 抽样像素以控制传统模型训练成本。

## 关键结果

在 6 个任务和 3 种 shot 设置下，玄女 embedding 的验证集选型结果在 17/18 个 task-shot 组合上超过最佳传统遥感 baseline。唯一例外是 `sports_pitch` 的 50-shot F1，传统多源 HGB 高 2.0%；但玄女在该项 AP 高 0.4%、AUC 高 2.3%。

50-shot 下的核心结果：

| 任务 | 玄女 F1 | 最佳传统 F1 | F1 相对提升 | 玄女 AP | 最佳传统 AP | AP 相对提升 |
|---|---:|---:|---:|---:|---:|---:|
| 建筑 | 0.4651 | 0.4016 | +15.8% | 0.4082 | 0.3372 | +21.1% |
| 道路 | 0.5187 | 0.4371 | +18.7% | 0.5758 | 0.4620 | +24.6% |
| 水体 | 0.6344 | 0.5604 | +13.2% | 0.6356 | 0.5663 | +12.2% |
| 公园绿地 | 0.7949 | 0.7448 | +6.7% | 0.8783 | 0.8223 | +6.8% |
| 学校高校 | 0.2652 | 0.2217 | +19.6% | 0.2248 | 0.2031 | +10.7% |
| 运动场地 | 0.3210 | 0.3276 | -2.0% | 0.2380 | 0.2370 | +0.4% |

标签效率结果：

| 对比 | 结论 |
|---|---|
| 玄女 5-shot vs 传统 50-shot | 建筑、道路、水体已经超过传统 50-shot；教育、公园绿地、运动场仍低于传统 50-shot。 |
| 玄女 10-shot vs 传统 50-shot | 建筑、道路、水体、公园绿地超过传统 50-shot；教育基本持平；运动场仍不足。 |

## 生成脚本

传统 ML benchmark：

```bash
python scripts/eval/run_traditional_ml_benchmark.py \
  --embedding-root /data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604 \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_valselected_fulltest \
  --tasks building road water park_green education sports_pitch \
  --feature-sets xuannv_embedding s2_indices s2_s1_landsat_indices \
  --models rf extratrees logistic hgb \
  --shots 5 10 50 \
  --fold 0 \
  --max-pixels-per-patch 1024
```

报告和 320 patch 拼图：

```bash
python scripts/report/build_traditional_ml_benchmark_report.py \
  --metrics-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_valselected_fulltest \
  --visual-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_visuals \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_report
```

## 注意

- 本轮是 Phase B 修正版，修复了 Phase A 中测试集抽样和测试集 oracle F1 的问题。
- 本轮不包含 full-shot、XGBoost、SVM 和 U-Net；XGBoost 未安装，使用 sklearn 内置 HGB 作为增强 boosting baseline。
- kNN 像素级全量预测会产生很大的距离矩阵，本轮没有放入主表；后续更适合单独做 ROI 检索评测。
- 指标使用 patch 内抽样验证/测试，以保证传统 ML 评测速度可控；所有方法使用同一抽样协议，因此横向比较公平。
