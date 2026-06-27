# V2 实验版本与效果总结 - 2026-06-27

这份文档用于快速回顾最近几轮 V2 embedding 实验：每一版改了什么、效果如何、当前应该选哪一个 checkpoint 作为后续升级底座，以及下一轮最值得优先解决的问题。

## 一句话结论

当前推荐把 **P1B full 5-fold** 作为新的升级底座。

它相比 Round2 P0 在整体 macro 指标上继续提升，尤其是 `rubbish` 和 `building_change` 这类稀疏变化任务提升明显。不过 `construction_joint` 相比 P0 有回退，所以后续 P2 不建议先继续大改采样器，而应该优先做阈值校准、每类任务 operating point、以及下游 head/loss 的针对性修复。

## 版本时间线

| 版本 | 主要改动 | Checkpoint / Embedding | 测评方式 | 结论 |
|---|---|---|---|---|
| Round2 P0 | 修完 P0 问题后的监督式变化对齐 baseline | `/data/xuannv_embedding/outputs/v2_round2_supervised_change_20260627_050500/best.pt` | 完整 5 折 | 很强的 baseline；macro 指标超过 AEF，但稀疏任务和阈值校准仍偏弱。 |
| P1A | 只做标签和 fold 诊断，不改模型 | 无新模型 | 诊断分析 | 确认 Harbin 的部分 fold 正样本极少，任务本身确实有稀疏和不稳定问题。 |
| P1B quick | 加入 positive-aware sampler 和 task-aware hard-negative supervised loss | `/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt` | fold-0 快速测评 | 结果很有希望，macro 超过 AEF，`rubbish` 明显提升；但 fold0 的 `farm_change` 太稀疏，不能作为最终判断。 |
| P1B full | 使用同一个 P1B checkpoint 做完整验证 | 同上 | 完整 5 折 | 当前最推荐的升级 baseline；macro 超过 AEF，并且稀疏目标任务整体变好。 |

## Macro 总体指标

| 版本 | 测评方式 | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | 对比 AEF |
|---|---|---:|---:|---:|---:|---:|---|
| AEF | 参考基线 | 0.7848 | 0.1305 | 0.0846 | n/a | 0.0491 | 官方对比 baseline。 |
| Round2 P0 | 完整 5 折 | 0.9397 | 0.2487 | 0.0967 | 0.2084 | 0.1220 | macro 全部超过 AEF。 |
| P1B quick | fold 0 | 0.9301 | 0.3148 | 0.1286 | n/a | 0.1481 | 只能作为快速检查，表现很好但 fold 敏感。 |
| P1B full | 完整 5 折 | 0.9452 | 0.3007 | 0.1157 | n/a | 0.1360 | 当前最好的完整验证 macro 结果。 |

说明：`F1@val_thr` 是用验证集学习出来的阈值后的 F1。它在 AEF 对比表里不一定都有，但对实际落地很重要，因为现在模型在默认 `0.5` 阈值下校准还不好。

## 完整 5 折分任务结果

### Round2 P0

| 任务 | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU |
|---|---:|---:|---:|---:|---:|
| construction | 0.8943 | 0.3272 | 0.1574 | 0.2975 | 0.1755 |
| building_change | 0.9548 | 0.1586 | 0.0607 | 0.1241 | 0.0662 |
| farm_change | 0.9318 | 0.1711 | 0.0625 | 0.1142 | 0.0616 |
| rubbish | 0.9644 | 0.1461 | 0.0430 | 0.1101 | 0.0593 |
| construction_joint | 0.9530 | 0.4408 | 0.1599 | 0.3962 | 0.2476 |
| macro | 0.9397 | 0.2487 | 0.0967 | 0.2084 | 0.1220 |

### P1B full 5-fold

| 任务 | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | 对比 AEF |
|---|---:|---:|---:|---:|---:|---|
| construction | 0.9098 | 0.3705 | 0.1585 | 0.3476 | 0.2119 | 全部指标超过 AEF。 |
| building_change | 0.9620 | 0.2524 | 0.0794 | 0.1161 | 0.0627 | 全部指标超过 AEF，但 mIoU 仍然不高。 |
| farm_change | 0.9450 | 0.1603 | 0.0543 | 0.1032 | 0.0575 | 超过 AEF，但部分指标略低于 P0。 |
| rubbish | 0.9954 | 0.4023 | 0.1825 | 0.2825 | 0.1710 | 提升最大，是 P1B 最明显的收益点。 |
| construction_joint | 0.9139 | 0.3180 | 0.1038 | 0.2983 | 0.1770 | AUC 超过 AEF，但 F1 指标仍输给 AEF，mIoU 也略低于 AEF。 |
| macro | 0.9452 | 0.3007 | 0.1157 | n/a | 0.1360 | macro 对比指标全部超过 AEF。 |

## 主要提升

- P1B 对稀疏变化任务整体有效，尤其是 `rubbish` 和 `building_change`。
- Macro F1_best 从 Round2 P0 的 `0.2487` 提升到 P1B full 的 `0.3007`。
- Macro mIoU 从 Round2 P0 的 `0.1220` 提升到 P1B full 的 `0.1360`。
- AUC 仍然很强，说明 embedding 排序能力不错，也就是模型大体能把正负样本区分开。
- P1A 的标签诊断是有价值的：稀疏 fold 不平衡是真的存在，后面需要显式处理。

## 仍然存在的问题

- `construction_joint` 从 P0 到 P1B 出现回退，mIoU 从 `0.2476` 降到 `0.1770`。P1B 的 mIoU 几乎追平 AEF，但 F1_best 和 F1@0.5 仍然输给 AEF。
- `farm_change` 仍然不稳定。P1B full 虽然超过 AEF，但并没有在所有指标上明显超过 P0。
- 默认阈值 `0.5` 仍然不好用。P1B full 使用验证集阈值后，F1 有明显提升：
  - construction：`+0.1890`
  - building_change：`+0.0367`
  - farm_change：`+0.0488`
  - rubbish：`+0.1000`
  - construction_joint：`+0.1945`

这说明模型不是完全没学到，而是“分数怎么转成最终预测图”这一步还没有校准好。

## 当前推荐底座

后续如果继续做稀疏变化检测能力升级，建议以 P1B full 作为当前 baseline：

- 配置：`configs/v2_p1_sparse_sampler_hardneg_20260627_090500.yaml`
- 权重：`/data/xuannv_embedding/outputs/v2_p1_sparse_sampler_hardneg_20260627_090500/best.pt`
- Embedding 目录：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260627_v2_p1_sparse_sampler_hardneg_20260627_090500_best_v2_p1_sparse_sampler_hardneg_20260627_090500`
- 完整测评目录：`/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/v2_p1_sparse_sampler_hardneg_full5fold_20260627_122751`

但要注意：如果当前产品最优先的是 `construction_joint`，Round2 P0 仍然是一个很强的对照版本，因为它在 `construction_joint` 上明显更好。

## 下一步建议

下一轮 P2 不建议先做“大而全”的采样器变化。更推荐先把输出分数和下游任务校准做好：

1. 把 calibration report 和 calibrated inference artifact 固化进训练后测评流程，不只是生成分析表。
2. 为每个任务保存验证集学到的最佳阈值，尤其是 `construction_joint` 和 `rubbish`。
3. 针对 `construction_joint` 单独测试下游 head 和 loss，而不是马上重新训练 embedding。
4. 补充更丰富的下游能力测试：先加 WorldCover 土地覆盖分割、patch-level 土地利用分类；道路提取、静态建筑物提取等任务等标签准备好后再加入。

## 参考报告

- Round2 P0 完整 5 折：`docs/experiments/v2_round2_p0_full5fold_20260627.md`
- P1 标签和 fold 诊断：`docs/experiments/v2_p1_label_fold_diagnostics_20260627.md`
- P1B 快速测评：`docs/experiments/v2_p1_hardneg_quick_eval_20260627.md`
- P1B 完整 5 折：`docs/experiments/v2_p1_hardneg_full5fold_20260627.md`
