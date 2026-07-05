# 海淀生产版细粒度 OSM 下游能力详细分析

日期：2026-07-05

## 1. 结论先行

本轮目标是回答一个问题：海淀生产版 Xuannv embedding 除了建筑、道路、水体、施工地之外，能不能支持更细粒度的 OSM 地物任务，例如球场、公园、学校、医院、停车场、科研政务用地等。

结论是：**能支持，而且在不少细粒度类别上已经超过 AEF；但在 park、retail、hospital、garden 这类 OSM 语义边界复杂的类别上仍然落后。**

总体 72 组成对比较结果：

| 指标 | Xuannv 胜出 | 平均差值 Xuannv - AEF | 解读 |
| --- | ---: | ---: | --- |
| F1 | 38/72 | +0.0005 | 整体接近，略高一点 |
| AUC | 39/72 | +0.0228 | 排序能力更好，说明概率图里经常已经把目标排到前面 |
| AP | 45/72 | +0.0023 | 稀疏目标的整体检索能力略优 |
| mIoU | 38/72 | -0.0011 | 落地二值分割边界仍有波动 |

重点看 `MLP + full-shot`，Xuannv 明显更强的类别：

- `pitch`：F1 0.3528 vs AEF 0.2854，AUC 0.8995 vs 0.8923
- `sports`：F1 0.1707 vs AEF 0.1135，AUC 0.8375 vs 0.8319
- `university`：F1 0.2300 vs AEF 0.1732，AUC 0.8238 vs 0.8108
- `grass`：F1 0.0598 vs AEF 0.0149，AUC 0.7442 vs 0.6829
- `research_gov`：F1 0.1050 vs AEF 0.0704，AUC 0.7404 vs 0.7307
- `forest`：F1 0.7942 vs AEF 0.7694，AUC 0.9487 vs 0.9410
- `school`：F1 0.1756 vs AEF 0.1686，AUC 0.7902 vs 0.7787

这说明当前海淀生产版 embedding 对**校园、高校、运动场、科研政务、草地、森林**等语义已经有比较明显的可分性。

## 2. 评测设置

对比对象：

- Xuannv Haidian v1：P10C epoch800 海淀生产版 embedding
- AEF annual 2025：AEF 官方年度 embedding

评测公平性：

- 使用同一批 OSM 弱标签 mask
- 使用同一 fold0 划分
- 使用同一 downstream head：`linear`、`MLP`
- 使用同一 shot 设置：`10`、`50`、`full`
- 使用同一训练轮数：50 epoch
- 使用同一套指标：F1、AUC、AP、mIoU

标签补充方式：

- 对本地没有的细粒度标注，使用在线 OSM 数据补充缓存。
- 然后把 OSM 面、线、点要素统一栅格化到 patch 的 128x128 mask。
- 本轮纳入：`park`、`garden`、`pitch`、`sports`、`school`、`university`、`hospital`、`parking`、`forest`、`grass`、`retail`、`research_gov`。
- `cemetery` 只有 22 个正样本 patch，太稀疏，暂不作为稳定结论。

产物路径：

- 指标目录：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_fold0`
- 配对指标表：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_fold0/paired_comparison.csv`
- 可视化目录：`/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full`

## 3. 按类别平均指标

这里把 `linear/MLP`、`10/50/full-shot` 六种设置都平均到同一个类别下，观察每一类任务的总体趋势。

| task | Xuannv F1 | AEF F1 | ΔF1 | Xuannv AUC | AEF AUC | ΔAUC | Xuannv AP | AEF AP | ΔAP | Xuannv mIoU | AEF mIoU | ΔmIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pitch | 0.1339 | 0.0976 | 0.0363 | 0.6766 | 0.6786 | -0.0020 | 0.0933 | 0.0598 | 0.0335 | 0.0769 | 0.0544 | 0.0225 |
| grass | 0.0501 | 0.0303 | 0.0197 | 0.7015 | 0.6549 | 0.0466 | 0.0746 | 0.0237 | 0.0509 | 0.0258 | 0.0155 | 0.0102 |
| university | 0.1185 | 0.1000 | 0.0185 | 0.6559 | 0.5987 | 0.0571 | 0.0694 | 0.0537 | 0.0157 | 0.0644 | 0.0531 | 0.0113 |
| research_gov | 0.0709 | 0.0547 | 0.0161 | 0.5952 | 0.5547 | 0.0405 | 0.0376 | 0.0310 | 0.0065 | 0.0369 | 0.0282 | 0.0087 |
| sports | 0.0573 | 0.0417 | 0.0156 | 0.6899 | 0.6739 | 0.0160 | 0.0243 | 0.0194 | 0.0048 | 0.0307 | 0.0218 | 0.0089 |
| parking | 0.0448 | 0.0371 | 0.0076 | 0.6949 | 0.6206 | 0.0743 | 0.0246 | 0.0203 | 0.0043 | 0.0229 | 0.0190 | 0.0039 |
| garden | 0.0134 | 0.0187 | -0.0053 | 0.6176 | 0.5592 | 0.0584 | 0.0066 | 0.0083 | -0.0017 | 0.0067 | 0.0096 | -0.0028 |
| school | 0.0849 | 0.0952 | -0.0103 | 0.6596 | 0.6639 | -0.0043 | 0.0581 | 0.0583 | -0.0003 | 0.0452 | 0.0504 | -0.0052 |
| retail | 0.0328 | 0.0503 | -0.0175 | 0.7227 | 0.6554 | 0.0673 | 0.0196 | 0.0252 | -0.0056 | 0.0168 | 0.0264 | -0.0095 |
| hospital | 0.0321 | 0.0500 | -0.0179 | 0.6817 | 0.6545 | 0.0271 | 0.0158 | 0.0212 | -0.0054 | 0.0165 | 0.0259 | -0.0095 |
| forest | 0.7359 | 0.7597 | -0.0238 | 0.9177 | 0.9340 | -0.0163 | 0.8159 | 0.8654 | -0.0494 | 0.5833 | 0.6130 | -0.0297 |
| park | 0.2977 | 0.3310 | -0.0333 | 0.6684 | 0.7596 | -0.0911 | 0.2394 | 0.2647 | -0.0253 | 0.1814 | 0.2034 | -0.0220 |

观察：

- `pitch` 是本轮最稳定的优势类别。F1、AP、mIoU 都明显领先，说明模型对球场/运动场这类高纹理、规则形状目标比较敏感。
- `grass`、`university`、`research_gov`、`parking` 的 AUC 提升比较明显，说明 Xuannv 的概率排序更好，即目标区域在概率图中往往排得更靠前。
- `retail`、`hospital` 虽然 F1 低于 AEF，但 AUC 高于 AEF。这说明不是完全没识别到，而是阈值、类别边界、OSM 标注噪声会影响最终二值结果。
- `park` 是当前最明显短板，F1/AUC/AP/mIoU 都低于 AEF。公园在 OSM 里经常包含水体、林地、道路、建筑、运动设施等混合区域，语义本身比单一地物更复杂。

## 4. MLP full-shot 重点结果

`MLP + full-shot` 更接近“有较多弱标签时，下游头能从 embedding 里挖出多少语义”的情况。

| task | Xuannv F1 | AEF F1 | ΔF1 | Xuannv AUC | AEF AUC | ΔAUC | Xuannv AP | AEF AP | ΔAP | Xuannv mIoU | AEF mIoU | ΔmIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pitch | 0.3528 | 0.2854 | 0.0674 | 0.8995 | 0.8923 | 0.0072 | 0.2803 | 0.2272 | 0.0531 | 0.2142 | 0.1664 | 0.0478 |
| sports | 0.1707 | 0.1135 | 0.0572 | 0.8375 | 0.8319 | 0.0056 | 0.0699 | 0.0632 | 0.0067 | 0.0933 | 0.0601 | 0.0332 |
| university | 0.2300 | 0.1732 | 0.0568 | 0.8238 | 0.8108 | 0.0130 | 0.1468 | 0.1067 | 0.0401 | 0.1300 | 0.0948 | 0.0351 |
| grass | 0.0598 | 0.0149 | 0.0449 | 0.7442 | 0.6829 | 0.0613 | 0.1176 | 0.0166 | 0.1010 | 0.0308 | 0.0075 | 0.0233 |
| research_gov | 0.1050 | 0.0704 | 0.0346 | 0.7404 | 0.7307 | 0.0097 | 0.0588 | 0.0488 | 0.0099 | 0.0554 | 0.0365 | 0.0189 |
| forest | 0.7942 | 0.7694 | 0.0248 | 0.9487 | 0.9410 | 0.0077 | 0.8831 | 0.8831 | 0.0000 | 0.6587 | 0.6252 | 0.0335 |
| school | 0.1756 | 0.1686 | 0.0070 | 0.7902 | 0.7787 | 0.0115 | 0.1218 | 0.1216 | 0.0002 | 0.0963 | 0.0921 | 0.0042 |
| park | 0.4676 | 0.4782 | -0.0106 | 0.8763 | 0.8828 | -0.0064 | 0.4722 | 0.4920 | -0.0198 | 0.3051 | 0.3142 | -0.0091 |
| hospital | 0.0797 | 0.1011 | -0.0214 | 0.8200 | 0.8216 | -0.0016 | 0.0339 | 0.0414 | -0.0076 | 0.0415 | 0.0532 | -0.0117 |
| parking | 0.0474 | 0.0749 | -0.0275 | 0.7503 | 0.7686 | -0.0183 | 0.0313 | 0.0351 | -0.0038 | 0.0243 | 0.0389 | -0.0146 |
| garden | 0.0240 | 0.0632 | -0.0392 | 0.7740 | 0.8505 | -0.0765 | 0.0119 | 0.0258 | -0.0139 | 0.0121 | 0.0326 | -0.0205 |
| retail | 0.0777 | 0.1353 | -0.0576 | 0.8365 | 0.8728 | -0.0363 | 0.0404 | 0.0624 | -0.0219 | 0.0404 | 0.0725 | -0.0321 |

重点解释：

- 在 MLP full-shot 下，Xuannv 对 `pitch/sports/university/grass/research_gov/forest/school` 全部超过 AEF，说明当前 embedding 不只是能做传统地物，也能支持更细的城市功能语义。
- `forest` 在类别均值上略低，但 MLP full-shot 下反超，说明只要下游头有足够样本，Xuannv 的森林语义是可以被挖出来的。
- `retail/garden/hospital/park` 仍然是短板。它们更像“功能区域”而不是纯视觉地物，内部可能混合建筑、道路、树木、空地、水体，单靠当前 embedding 还不够稳定。

## 5. Shot 与 Head 分析

| shot | head | ΔF1 | ΔAUC | ΔAP | ΔmIoU |
| --- | --- | --- | --- | --- | --- |
| 10 | linear | -0.0055 | 0.0463 | -0.0070 | -0.0049 |
| 10 | mlp | -0.0060 | 0.0365 | -0.0073 | -0.0057 |
| 50 | linear | -0.0108 | 0.0244 | -0.0064 | -0.0096 |
| 50 | mlp | 0.0209 | 0.0184 | 0.0274 | 0.0109 |
| full | linear | -0.0071 | 0.0130 | -0.0046 | -0.0064 |
| full | mlp | 0.0114 | -0.0019 | 0.0120 | 0.0090 |

这个表很关键：

- 10-shot 和 50-shot 的 AUC 大多是正的，说明在少量标签时，Xuannv embedding 的排序能力不错。
- 50-shot MLP 是最亮眼的组合，F1、AUC、AP、mIoU 都为正，说明给一点点样本后，浅层非线性头能明显挖出语义。
- full-shot MLP 的 F1/AP/mIoU 仍然为正，但 AUC 略低，说明某些类别在充分训练后边界拟合更好，但概率排序并不总是全面压过 AEF。

## 6. 可视化对比

下面每张图的组织方式与本轮评测一致：上面是 Xuannv，下面是 AEF；包含影像、GT、预测概率和阈值后结果。这里优先展示 Xuannv 表现较强的类别，同时附少量短板类别作为诊断。

### 6.1 Pitch

`pitch` 是本轮最强类别之一，规则形状明显，Xuannv 的预测更容易贴住目标区域。

![pitch patch 000030](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/pitch/pitch_mlp_shot_full_patch_000030_compare.png)

![pitch patch 000126](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/pitch/pitch_mlp_shot_full_patch_000126_compare.png)

### 6.2 University

`university` 代表校园类复合场景，里面包含建筑、道路、操场、绿地。Xuannv 在 full-shot MLP 下 F1、AUC、AP、mIoU 均高于 AEF。

![university patch 000030](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/university/university_mlp_shot_full_patch_000030_compare.png)

![university patch 000126](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/university/university_mlp_shot_full_patch_000126_compare.png)

### 6.3 Sports

`sports` 和 `pitch` 类似，但范围更杂一些。Xuannv 在 MLP full-shot 下 F1 提升明显。

![sports patch 000025](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/sports/sports_mlp_shot_full_patch_000025_compare.png)

![sports patch 000110](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/sports/sports_mlp_shot_full_patch_000110_compare.png)

### 6.4 Research / Government

`research_gov` 是海淀很重要的功能区语义。这个类别不是单一地物，而是功能区域，因此能超过 AEF 说明 embedding 里已经有一定城市功能表达。

![research_gov patch 000120](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/research_gov/research_gov_mlp_shot_full_patch_000120_compare.png)

![research_gov patch 000206](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/research_gov/research_gov_mlp_shot_full_patch_000206_compare.png)

### 6.5 Grass

`grass` 的 F1 绝对值不高，但 Xuannv 的 AP 和 AUC 提升很明显。这个类别通常很碎，容易受 OSM 标注粒度影响。

![grass patch 000058](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/grass/grass_mlp_shot_full_patch_000058_compare.png)

![grass patch 000203](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/grass/grass_mlp_shot_full_patch_000203_compare.png)

### 6.6 Forest

`forest` 是大面积自然地物。类别均值上 AEF 略优，但 MLP full-shot 下 Xuannv 反超，说明模型里已有可用森林语义。

![forest patch 000084](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/forest/forest_mlp_shot_full_patch_000084_compare.png)

![forest patch 000232](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/forest/forest_mlp_shot_full_patch_000232_compare.png)

### 6.7 短板类别示例

下面这些类别不是重点展开对象，但需要保留诊断价值。

`park`：公园是混合区域，不是单一地物，AEF 当前略强。

![park patch 000088](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/park/park_mlp_shot_full_patch_000088_compare.png)

`retail`：零售/商业区域的 OSM 定义依赖功能边界，视觉上可能和普通建筑、道路混在一起。

![retail patch 000009](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/retail/retail_mlp_shot_full_patch_000009_compare.png)

`hospital`：医院区域也常常是功能区，不是单一视觉对象，边界容易和周边建筑混淆。

![hospital patch 000113](/data/xuannv_embedding/experiments/haidian_production_capability_suite_20260705/phase3_fine_osm_visualizations_mlp_full/hospital/hospital_mlp_shot_full_patch_000113_compare.png)

## 7. 为什么 AUC 高但 F1 不一定高

本轮有几个类别出现了一个很重要的现象：**Xuannv 的 AUC 高于 AEF，但 F1 或 mIoU 低于 AEF。**

这通常表示：

- 模型已经把目标区域排到了较高概率位置；
- 但概率校准、阈值选择、边界细节还没完全做好；
- OSM 弱标签本身可能不是像素级真值，而是带噪声的功能区边界；
- 小目标和碎片目标会让 F1/mIoU 对阈值非常敏感。

所以后续不能只看固定阈值 F1。对于弱标签制图，建议同时看：

- AUC：有没有把目标排到前面；
- AP：稀疏目标检索能力；
- best-F1 或验证集阈值 F1：最终制图可用性；
- 可视化：是否出现大面积误检、漏检、边界错位。

## 8. 本轮证明了什么

这轮结果对生产版 embedding 有三个积极信号。

第一，**不是只能做建筑、道路、水体这类低层地物**。在 `pitch/sports/university/research_gov` 上的结果说明，embedding 已经承载了一部分城市功能语义。

第二，**少量标签下有潜力**。AUC 在 10-shot、50-shot 下整体偏正，说明即使用少量标注，概率排序也常常是可用的。

第三，**浅层 MLP 可以挖出比 linear 更多的信息**。50-shot MLP 的平均 ΔF1、ΔAUC、ΔAP、ΔmIoU 全部为正，说明 embedding 里存在一些非线性可分的语义结构。

## 9. 当前不足

这里简要记录，不展开太多。

- `park/garden/retail/hospital` 仍弱于 AEF，主要原因可能是功能区标签和视觉地物不完全一致。
- 一些稀疏类别的 F1 绝对值仍低，需要更多正样本、更干净的 OSM 合并规则，或者更合理的阈值策略。
- OSM 弱标签不能等价于人工精标，边界天然会有噪声；对这些类别评价时不能只盯像素级 F1。

## 10. 下一步建议

优先做三件事：

1. 对 `park/garden/retail/hospital` 做 OSM 规则清洗，把明显过粗、过大的面剔除或拆成更合理的子类。
2. 下游评测报告里固定展示 AUC、AP、best-F1、验证阈值 F1，不再只看一个 F1。
3. 对 Xuannv AUC 高但 F1 低的类别，单独做阈值校准和概率归一化实验，判断是不是“识别到了但阈值没切好”。

整体判断：**海淀生产版 embedding 已经具备扩展到多类 OSM 弱标签制图的能力；下一步的主要瓶颈不是模型完全没有语义，而是弱标签噪声、类别定义、阈值校准和功能区边界问题。**

