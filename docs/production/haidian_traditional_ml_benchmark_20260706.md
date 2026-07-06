# 玄女海淀 V1 传统机器学习对比评测

本轮评测用于回答一个更宽的问题：玄女 embedding 是否只是比 AEF 好，还是也能超过传统遥感特征加传统机器学习方法。

## 输出目录

```text
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_core/
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_visuals/
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_report/
```

主要报告：

```text
/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_report/haidian_v1_traditional_ml_benchmark_report.md
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
- 传统模型：Random Forest、Logistic Regression。
- 阈值：在验证集选择 F1-best 阈值，再在测试集汇报。

## 关键结果

在 6 个任务和 3 种 shot 设置下，玄女 embedding 的最佳结果全部超过最佳传统遥感 baseline。

50-shot 下的核心结果：

| 任务 | 玄女 F1 | 最佳传统 F1 | F1 提升 | 玄女 AP | 最佳传统 AP | AP 提升 |
|---|---:|---:|---:|---:|---:|---:|
| 建筑 | 0.7172 | 0.6877 | +0.0295 | 0.7386 | 0.6788 | +0.0597 |
| 道路 | 0.7384 | 0.7007 | +0.0377 | 0.8121 | 0.7523 | +0.0598 |
| 水体 | 0.6689 | 0.6315 | +0.0374 | 0.7637 | 0.7244 | +0.0393 |
| 公园绿地 | 0.7922 | 0.7531 | +0.0392 | 0.8413 | 0.8117 | +0.0297 |
| 学校高校 | 0.5397 | 0.5132 | +0.0265 | 0.5480 | 0.5121 | +0.0360 |
| 运动场地 | 0.5330 | 0.5132 | +0.0198 | 0.5626 | 0.5356 | +0.0271 |

## 生成脚本

传统 ML benchmark：

```bash
python scripts/eval/run_traditional_ml_benchmark.py \
  --embedding-root /data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604 \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_core \
  --tasks building road water park_green education sports_pitch \
  --feature-sets xuannv_embedding s2_indices s2_s1_landsat_indices \
  --models rf logistic \
  --shots 5 10 50 \
  --fold 0 \
  --max-pixels-per-patch 1024 \
  --max-eval-pixels-per-patch 2048
```

报告和 320 patch 拼图：

```bash
python scripts/report/build_traditional_ml_benchmark_report.py \
  --metrics-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_core \
  --visual-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseA_visuals \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_report
```

## 注意

- 本轮是 Phase A 核心版，不包含 full-shot、XGBoost、SVM 和 U-Net。
- kNN 像素级全量预测会产生很大的距离矩阵，本轮没有放入主表；后续更适合单独做 ROI 检索评测。
- 指标使用 patch 内抽样验证/测试，以保证传统 ML 评测速度可控；所有方法使用同一抽样协议，因此横向比较公平。
