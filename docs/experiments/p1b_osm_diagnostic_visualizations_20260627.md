# P1B OSM 下游任务诊断可视化记录（2026-06-27）

## 背景

用户指出：OSM 建筑物、道路等任务中，模型可能把大量背景预测成前景，导致 ROC-AUC 看起来偏高，但真实分割效果并不好。这个判断是对的，尤其在建筑、道路、水体这类稀疏目标任务里，AUC 只能说明像素排序能力，不能说明固定阈值后的可用分割质量。

## 本次修正

更新脚本：`scripts/report/visualize_osm_downstream_outputs.py`

新增诊断内容：

- 每个任务默认输出 8 张诊断图，不只挑“看起来好”的样例。
- 样例选择混合三类：GT 阳性面积较大、验证阈值下 F1 较差、误检 FP 较多。
- 每张图横向展示：高分辨率影像、P1B embedding PCA、概率图、`Pred >= 0.50`、`Pred >= val_threshold`、OSM GT、TP/FP/FN 错误图、GT overlay。
- 错误图颜色：绿色 TP，橙色 FP，红色 FN，浅灰 TN。
- `index.md` 和 `metadata.json` 中记录每张图的 GT 正类比例、预测正类比例、F1、FP、FN。

输出目录：

`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_quick_20260627/osm_diagnostic_sheets`

索引文件：

`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_osm_quick_20260627/osm_diagnostic_sheets/index.md`

## Quick 任务指标复查

| 任务 | AUC | F1@0.5 | F1@val_thr | mIoU | val_thr |
|---|---:|---:|---:|---:|---:|
| haidian_building_osm | 0.8336 | 0.2053 | 0.3525 | 0.2140 | 0.9743 |
| haidian_road_osm | 0.7219 | 0.2736 | 0.3864 | 0.2394 | 0.9600 |
| harbin_building_osm | 0.9515 | 0.2446 | 0.4129 | 0.2601 | 0.9528 |
| harbin_road_osm | 0.8423 | 0.1802 | 0.3644 | 0.2228 | 0.9717 |

## 诊断结论

1. 当前 OSM 下游头存在明显概率校准问题：4 个任务的验证阈值都在 0.95 到 0.97 之间，说明 `0.5` 不是可用阈值。
2. `F1@0.5` 明显低于 `F1@val_thr`，说明模型常把大量区域打成高概率，阈值稍低时前景会泛滥。
3. 对稀疏目标任务，后续汇报不能把 ROC-AUC 作为主指标。主指标应改为：`F1@0.5`、`F1@val_thr`、`mIoU`、预测正类比例、TP/FP/FN 可视化。
4. 后续训练/测评流程应固定生成诊断图，避免只用 AUC 或只展示好看的样例。

## 后续建议

- 下游头训练加入更强的类别不平衡处理，例如 Dice/Focal、Tversky、正负样本采样控制。
- 增加概率校准评估，例如 ECE、reliability diagram、固定正类比例阈值。
- 对道路/建筑任务分别处理：道路应强化细线结构连续性，建筑应强化块状边界和面积一致性。
- 汇报时同时放 AUC 与阈值后结果，明确说明 AUC 只是辅助排序指标。
