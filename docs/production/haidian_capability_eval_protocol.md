# 海淀生产版 Embedding 能力评测协议

## 目标

本协议用于公平评估 `haidian_embedding_v1` 与 AEF 年度 embedding 的下游能力。核心问题不是某个下游头是否足够强，而是 frozen embedding 在少量标签、简单下游头和统一阈值规则下，能否快速生成可靠地图。

## 公平性约束

- 两个模型使用同一个下游训练脚本：`scripts/eval/train_aef_downstream_probe.py`。
- 两个模型使用同一套标签、同一套 `split_5fold.json`、同一 fold。
- 两个模型使用同样的 head：`linear` 或 `mlp`。
- 两个模型使用同样的优化器、学习率、epoch、采样像素数、正负采样比例。
- 阈值选择统一为：在 validation set 上选择最佳 F1 阈值，再固定该阈值评估 test set。
- 唯一允许变化的是 embedding 来源：
  - `xuannv_haidian_v1`：月度 embedding，默认使用 `202604`。
  - `aef_annual_2025`：AEF 年度 embedding，本地文件名使用 `202512`。

## Phase 1：稀疏标签快速制图

Phase 1 先跑 fold0 快速矩阵，用于判断趋势。

任务：

- building
- road
- water
- construction

下游头：

- `linear`：最严格的线性可分性测试。
- `mlp`：两层小 MLP，测试简单非线性可读性。

稀疏标签设置：

- `shot_5`
- `shot_10`
- `shot_20`
- `shot_50`
- `shot_full`

这里的 shot 表示从训练 split 中选取多少个含正样本的训练 patch；负样本 patch 默认选相同数量。validation/test split 不变。

输出目录：

`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_fold0_sparse_mapping`

启动命令：

```bash
python scripts/eval/run_embedding_capability_suite.py \
  --xuannv-root /data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/embeddings/20260705_haidian_embedding_v1_p10c_epoch800_production_epoch_80_p10c_epoch800_capability \
  --aef-root /data/xuannv_embedding/embeddings/aef_official_2025_annual \
  --output-root /data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_fold0_sparse_mapping \
  --tasks building road water construction \
  --heads linear mlp \
  --shots 5 10 20 50 full \
  --folds 0 \
  --epochs 50 \
  --eval-every 5 \
  --max-pixels-per-patch 4096 \
  --devices npu:0 npu:1 npu:2 npu:3 npu:4 npu:5 \
  --workers 6
```

## Phase 2：全量 5-fold 验证

Phase 1 完成后，选择最有代表性的设置扩展到 5-fold。默认优先：

- 每个任务最强的 head。
- few-shot 曲线上最能体现差异的 shot。
- `shot_full` 作为上限对照。

## Phase 3：Embedding 检索

后续增加 query-by-example 检索：

- 用户指定一个区域作为 query。
- 对 query 区域 embedding 求平均。
- 在全海淀 embedding 中找 top-k 相似区域。
- 用 OSM 标签或人工检查统计 top-k 同类比例。

## Phase 4：月度变化检测

使用已有训练窗口内月份，例如 `202604` 与 `202605`：

- 计算同一像素或区域的 embedding cosine/L2 距离。
- 生成变化热力图。
- 排序输出 top-k 变化区域。
- 与高分影像人工核查。

2026 年 6 月泛化暂不执行，按当前用户要求后置。
