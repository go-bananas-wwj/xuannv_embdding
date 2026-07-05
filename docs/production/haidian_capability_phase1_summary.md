# 海淀生产版 Embedding Phase 1 下游能力总结

## 评测设置

- 模型对比：`xuannv_haidian_v1` vs `aef_annual_2025`
- 区域：海淀区
- 月份：玄女使用 `202604` 月度 embedding；AEF 使用本地 2025 年度 embedding 文件
- 任务：building、road、water、construction
- 下游头：`linear`、`mlp`
- 稀疏标签：5、10、20、50、full 个正样本训练 patch
- Fold：fold0 快速矩阵
- 公平性：同一训练脚本、同一 split、同一 head、同一 epoch、同一阈值选择方式；只替换 embedding

这里的 `80 个 job` 指 80 次公平下游小实验，不是 80 个 embedding 模型：

`4 个任务 × 2 种 head × 5 种 shot × 2 个 embedding × 1 个 fold = 80`

这里的 `5-shot / 10-shot / 20-shot / 50-shot` 指训练下游头时，只从训练 split 中抽取 5/10/20/50 个“含目标正样本”的训练 patch，再配相同数量的负样本 patch。validation 和 test split 不变。因此它模拟的是“只标很少区域，然后快速制图”的场景。

输出目录：

`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_fold0_sparse_mapping`

可视化目录：

`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full`

可视化设置：`MLP + full-shot + fold0`。每张图上面一行是玄女，下面一行是 AEF；列依次为高分影像、embedding PCA、预测概率、预测 mask、GT。

## 总体结论

本轮共 80 个 job，全部完成且无失败。

| 指标 | 玄女胜出次数 |
|---|---:|
| F1@validation-threshold | 34 / 40 |
| AP | 32 / 40 |
| mIoU | 34 / 40 |

总体看，海淀生产版 embedding 在稀疏标签快速制图上已经具备很强能力，尤其是 construction 和 road。building 稳定小幅超过 AEF；water 在少样本和 linear 下表现较好，但 full-shot MLP 仍略低于 AEF。

## 分任务平均效果

| 任务 | 玄女平均 F1 | AEF 平均 F1 | F1 差值 | 玄女平均 AP | AEF 平均 AP | AP 差值 |
|---|---:|---:|---:|---:|---:|---:|
| building | 0.3821 | 0.3605 | +0.0216 | 0.3140 | 0.2761 | +0.0379 |
| road | 0.4292 | 0.3761 | +0.0531 | 0.3887 | 0.3044 | +0.0843 |
| water | 0.4866 | 0.4103 | +0.0763 | 0.4451 | 0.3933 | +0.0518 |
| construction | 0.3131 | 0.1465 | +0.1666 | 0.2342 | 0.1253 | +0.1090 |

## Building

建筑提取是稳定小胜。少样本 5-shot 时 AEF 略好，但从 10-shot 开始玄女基本持续超过 AEF，MLP full-shot 提升最明显。

| Head | Shot | 玄女 F1 | AEF F1 | 差值 | 玄女 AP | AEF AP |
|---|---:|---:|---:|---:|---:|---:|
| linear | 5 | 0.3013 | 0.3372 | -0.0359 | 0.2069 | 0.2402 |
| linear | full | 0.3850 | 0.3776 | +0.0075 | 0.3199 | 0.2971 |
| mlp | 5 | 0.3535 | 0.3652 | -0.0118 | 0.2876 | 0.2811 |
| mlp | full | 0.4921 | 0.4522 | +0.0399 | 0.4575 | 0.3851 |

结论：建筑语义已经能被简单 MLP 读出来，说明 embedding 对建筑地物有可用表达；但 5-shot 下还不如 AEF 稳，说明极少样本建筑泛化还可以继续优化。

代表性可视化：

- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/building/building_mlp_shot_full_patch_000126_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/building/building_mlp_shot_full_patch_000066_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/building/building_mlp_shot_full_patch_000109_compare.png`

## Road

道路提取是本轮最有价值的证据之一。除 linear 5-shot 外，玄女几乎全面超过 AEF，MLP 在 10/20/50-shot 的优势明显。

| Head | Shot | 玄女 F1 | AEF F1 | 差值 | 玄女 AP | AEF AP |
|---|---:|---:|---:|---:|---:|---:|
| linear | 5 | 0.2730 | 0.3257 | -0.0527 | 0.1089 | 0.1849 |
| linear | full | 0.4983 | 0.3985 | +0.0999 | 0.5209 | 0.3837 |
| mlp | 10 | 0.4648 | 0.3585 | +0.1063 | 0.4761 | 0.2785 |
| mlp | full | 0.5249 | 0.4924 | +0.0326 | 0.5806 | 0.5326 |

结论：道路结构在玄女 embedding 中比较清楚，少量标签加 MLP 就能明显释放能力。下一步应重点做道路全域图和空间连续性检查。

代表性可视化：

- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/road/road_mlp_shot_full_patch_000096_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/road/road_mlp_shot_full_patch_000245_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/road/road_mlp_shot_full_patch_000079_compare.png`

## Water

水体任务比较微妙。玄女在少样本 linear 下明显强于 AEF，但 full-shot MLP 下 AEF 略强。

| Head | Shot | 玄女 F1 | AEF F1 | 差值 | 玄女 AP | AEF AP |
|---|---:|---:|---:|---:|---:|---:|
| linear | 5 | 0.3956 | 0.0725 | +0.3231 | 0.2931 | 0.0467 |
| linear | full | 0.4893 | 0.5225 | -0.0332 | 0.5076 | 0.4951 |
| mlp | 20 | 0.5916 | 0.5650 | +0.0267 | 0.5586 | 0.5451 |
| mlp | full | 0.6414 | 0.6513 | -0.0098 | 0.6160 | 0.6677 |

结论：玄女 embedding 对水体有信号，尤其少样本下不弱；但 AEF 的年度稳定表征在 full-shot 水体上仍略好。这个任务要结合可视化判断边界、岛屿和阴影误检。

代表性可视化：

- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/water/water_mlp_shot_full_patch_000107_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/water/water_mlp_shot_full_patch_000109_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/water/water_mlp_shot_full_patch_000022_compare.png`

## Construction

施工地是玄女最强优势任务。linear full-shot 中，玄女 F1 为 0.3633，AEF 只有 0.0269；MLP 下玄女也持续领先。

| Head | Shot | 玄女 F1 | AEF F1 | 差值 | 玄女 AP | AEF AP |
|---|---:|---:|---:|---:|---:|---:|
| linear | 5 | 0.0768 | 0.0019 | +0.0749 | 0.0221 | 0.0444 |
| linear | 50 | 0.3646 | 0.0269 | +0.3376 | 0.2525 | 0.0487 |
| linear | full | 0.3633 | 0.0269 | +0.3364 | 0.2521 | 0.0487 |
| mlp | full | 0.4269 | 0.3710 | +0.0559 | 0.4008 | 0.3398 |

结论：施工地更接近月度状态任务，玄女月度 embedding 明显比 AEF 年度 embedding 更适合。这是我们区别于年度 AEF 的重要卖点。

代表性可视化：

- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/construction/construction_mlp_shot_full_patch_000154_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/construction/construction_mlp_shot_full_patch_000002_compare.png`
- `/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase1_visualizations_mlp_full/construction/construction_mlp_shot_full_patch_000134_compare.png`

注意：施工地总体指标玄女明显更强，但单 patch 波动较大。可视化中存在 AEF 更好的失败案例，说明该任务需要做 5-fold 和错误案例复核，不能只看 fold0 均值。

## 下游嵌入能力判断

本轮结果说明：

1. 玄女 embedding 不是只能靠复杂 decoder 才有效，linear 和小 MLP 已经能读出多类地物。
2. 在稀疏标签快速制图上，玄女整体强于 AEF，尤其 construction、road。
3. 玄女的月度状态表达对施工地这类动态地物更敏感，这是相对年度 embedding 的关键差异化。
4. 水体仍是需要重点分析的任务，AEF full-shot MLP 仍略强，可能来自年度多观测平均带来的稳定边界。
5. 当前只是 fold0 快速矩阵，下一步应对代表性设置做 5-fold，验证结果不是单 fold 偶然。

## 下一步建议

1. 生成 4 个任务的全域 320 patch 可视化，重点看 road、construction、水体边界。
2. 选择每个任务最佳 head 和关键 shot，扩展到 5-fold。
3. 增加 embedding 检索评测，展示 query-by-example 找相似地物。
4. 对 construction 做月度变化热力图，突出月度 embedding 相比年度 AEF 的优势。
