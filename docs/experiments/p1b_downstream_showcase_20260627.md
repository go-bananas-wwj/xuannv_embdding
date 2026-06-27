# P1B 下游任务汇报包 - 2026-06-27

本文档记录使用 P1B embedding 生成的下游任务汇报资产包，方便后续汇报、复现和继续迭代。

## 固定模型版本

- 模型版本：P1B sparse sampler + hard-negative supervised loss
- 配置：`configs/v2_p1_sparse_sampler_hardneg_20260627_090500.yaml`
- Checkpoint：`/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- Embedding：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- Benchmark：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751`

## 汇报包位置

- 汇报包目录：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627`
- 中文说明：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/README.md`
- 指标表：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/tables/p1b_vs_aef_metrics.csv`
- Macro 对比图：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/charts/macro_metrics_vs_aef.png`
- 分任务 F1_best 对比图：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/charts/task_f1_best_vs_aef.png`
- 分任务 mIoU 对比图：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/charts/task_miou_vs_aef.png`
- 每类任务代表可视化：`/data/xuannv_embedding/experiments/v2_202512_202605/presentation/p1b_downstream_showcase_20260627/visuals/`

## 下游任务范围

本次汇报包覆盖当前已有标注和完整 5 折测评的 5 类任务：

| 任务 | 中文说明 | 是否适合汇报重点展示 |
|---|---|---|
| `construction` | 施工地识别 | 是 |
| `building_change` | 建筑变化检测 | 是 |
| `farm_change` | 耕地变化检测 | 可展示，但说明稀疏和不稳定 |
| `rubbish` | 疑似垃圾/堆场检测 | 是，提升最明显 |
| `construction_joint` | 施工地联合检测 | 可展示，但需说明部分指标仍低于 AEF |

## 核心指标

P1B 完整 5 折 macro 指标整体超过 AEF：

| 指标 | P1B | AEF |
|---|---:|---:|
| AUC | 0.9452 | 0.7848 |
| F1_best | 0.3007 | 0.1305 |
| F1@0.5 | 0.1157 | 0.0846 |
| mIoU | 0.1360 | 0.0491 |

分任务上，`construction`、`building_change`、`farm_change`、`rubbish` 均在主要指标上超过 AEF；`construction_joint` 的 AUC 超过 AEF，但 F1 和 mIoU 仍需要后续优化。

## 可视化内容

每张代表图从左到右依次包含：

1. 变化前高分辨率影像；
2. 变化后高分辨率影像；
3. 变化前 embedding PCA；
4. 变化后 embedding PCA；
5. embedding 差分 PCA；
6. 模型预测概率图；
7. 真实 GT 标签。

本次每类任务选取 2 张代表图，共 10 张。图像尺寸检查通过：指标图为 1584x864 或 2070x900，任务条带图约为 3114x492。

## 复现命令

```bash
python scripts/report/build_p1b_presentation_package.py \
  --run-name p1b_downstream_showcase_20260627
```

该脚本会读取已有 P1B full 5-fold benchmark 和 visualizations，然后生成汇报目录、指标表、对比图、精选可视化和中文 README。

## 后续建议

- 汇报时重点讲：P1B embedding 在稀疏变化类任务上明显优于 AEF，尤其是 `rubbish` 和 `building_change`。
- 展示图建议优先使用 `rubbish`、`building_change`、`construction` 三类。
- `construction_joint` 可以作为诚实说明：模型已有排序能力，但默认阈值和下游 head 还需要进一步校准。
- 下一轮如果要加强汇报材料，可以把 WorldCover 土地覆盖、土地利用分类、道路/建筑提取这些新增下游任务接入同一个汇报包脚本。
