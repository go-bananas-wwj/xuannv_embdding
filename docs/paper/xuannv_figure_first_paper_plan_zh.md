# 玄女海淀论文 Figure-first 写作计划

> **已弃用（2026-07-16）：** 本文件保留为历史设计记录，其中部分 claim、任务划分和模型描述已被后续代码审计否定。JRS 稿件只以 `docs/paper/jrs_2026/` 下的框架、注册评测协议和 claim-evidence matrix 为准；本文件不得作为正文事实或正式实验协议引用。

日期：2026-07-07

## 0. 写作原则

本文先按图组织，再补正文。参考几类公开写作建议：

- ML 论文应围绕 1-3 个清晰 claim 组织 narrative，而不是堆技术细节。[Highly Opinionated Advice on How to Write ML Papers](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
- 每张图必须先定义 message，再选择最简单准确的图形类型。[How To Make Figures For Scientific Papers](https://annaclemens.com/blog/figure-graph-data-vizualisation-plot-scientific-paper/)
- 读者最好只看图也能理解论文主线；正文不能把关键结论只藏在图里。[Heuristics for Scientific Writing](https://www.approximatelycorrect.com/2018/01/29/heuristics-technical-scientific-writing-machine-learning-perspective/)
- 对图像类论文，图应尽量自解释、面向读者而非作者自己。[Creating clear and informative image-based figures for scientific papers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8041175/)
- 科学论文大纲里应提前列出 visuals、findings 和每张图服务的论点。[How to Write Your First Research Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3178846/)

因此本文主张收敛为：

> 玄女海淀是一个面向城市少样本制图的月度多源遥感 dense embedding。它不直接主张全球通用 foundation model，而是证明：月度、多源、高分细节注入和弱语义训练可以让少量标注更快读出城市地物分布。

## 1. 论文核心 claims

**Claim 1：月度 dense embedding 比年度 embedding 更适合表达城市状态。**

对应证据：施工地、水体、道路等任务；月度原图、embedding PCA、预测结果；与 AEF annual embedding 对比。

**Claim 2：玄女 embedding 支持少样本快速制图。**

对应证据：5-shot/10-shot/50-shot 下游任务，Linear/MLP probe，320 patch 全域制图，AEF/传统 ML 对比。

**Claim 3：P10/P11/P12 训练改进逐步解决表示退化和 patch 边界割裂。**

对应证据：有效维度、Top-k 方差、全域 PCA 边界连续性、P10/P11/P12 统一评测。

## 2. 主图设计

### Figure 1. 任务动机与系统总览

**要证明什么**

城市制图不是一次性年度问题，而是月度更新问题；玄女将多源时序观测压缩成每月 dense embedding，再接轻量下游头完成快速制图。

**怎么画**

一张横向流程图，分 4 个区域：

1. 多源输入：S2、S1、Landsat、高分光学、高分 SAR、OSM 弱语义。
2. 月度 embedding 训练：masked reconstruction、高分重建、vMF bottleneck、uniformity/VCReg、weak semantic probe。
3. 输出：202512-202605 每个月 `64D x 128 x 128` embedding。
4. 下游使用：少量 patch 标注 -> Linear/MLP head -> 全域 320 patch 制图。

**需要实验吗**

不需要新实验。需要从代码和配置抽取事实：

- P10C/P12 配置。
- 数据源列表。
- patch size、month list、embedding dim。

**正文承接**

Introduction 最后一段和 Method 开头围绕 Figure 1 写：我们不是训练一个端到端任务模型，而是训练一个月度地理表示，再用轻量头快速适配任务。

### Figure 2. 数据、patch 网格与训练样本质量

**要证明什么**

训练数据不是随便拼出来的，而是经过云雾 mask、OSM 弱语义合并、高分数据对齐和质量筛选；P12 进一步使用 context crop 解决边界上下文不足。

**怎么画**

2 行 4 列：

- 第 1 行：随机 2 个 patch 的 202512-202605 多源输入小图，展示 S2/Landsat/S1/高分光学。
- 第 2 行：同一 patch 的 cloud/pixel mask、OSM landcover 弱标签、P12 160 输入与中心 128 裁剪示意。

右侧放一个小地图：海淀 320 patch 地理分布。

**对应实验/数据**

- 复用已有训练前数据可视化脚本或新写 `scripts/paper/figure2_data_quality.py`。
- 数据来自 `/data/xuannv_embedding/processed/haidian`。
- 使用 `configs/regions/haidian_patches.json` 画 patch 网格。

**正文承接**

Data section 说明：

- 为什么不用 WorldCover 作为主弱标签，而用 OSM 合并地物类别。
- 为什么云雾和像素 mask 重要。
- 为什么 P12 不改变 patch 地理范围，只借邻居上下文再裁回中心。

### Figure 3. 表示空间质量：PCA、有效维度和类分离

**要证明什么**

embedding 的好坏不只看下游指标，还要看空间连续性、维度利用率和语义类别可分性。

**怎么画**

三列：

1. 全域 320 patch 地理分布 PCA：
   - AEF annual
   - P10C
   - P11C
   - P12B/P12C
2. 有效维度条形图：
   - effective rank
   - top10 variance explained
3. 类间/类内分离：
   - 类中心 cosine distance heatmap
   - centroid AUC 或 nearest-centroid AUC。

**对应实验**

- 使用统一导出的 202604 embedding。
- AEF 使用同一区域 annual embedding，但图注明确它是 2025 annual baseline。
- P12 等训练完成后加入。

**需要补做**

- P10C、P11C、P12B、P12C、AEF 的统一 embedding-space diagnostics。
- 当前已有 P11 diagnostics，但论文图必须统一脚本、统一采样。

**正文承接**

Representation Analysis 章节。正文要主动说明：

- PCA 只是定性图，不是主指标。
- P10C 的 64 维利用率不足，是 P11/P12 改进动机。
- P12 的目标之一是减少 patch 边界颜色跳变。

### Figure 4. 少样本快速制图主结果

**要证明什么**

玄女 embedding 的核心价值是：只标少量 patch，就能在全域快速制图。

**怎么画**

主图用 3 个任务，每个任务一行：

- 建筑 / 道路 / 水体，或道路 / 水体 / 运动场。
- 每行包含：
  1. 训练用 5-shot 或 10-shot patch 位置。
  2. 原始高分光学底图。
  3. GT/OSM 弱标签。
  4. 玄女预测概率。
  5. 玄女二值预测。
  6. AEF 二值预测。

旁边加一张 shot curve：

- x 轴：5/10/50 shot。
- y 轴：F1 或 mIoU。
- 曲线：Xuannv Linear、Xuannv MLP、AEF MLP、Raw-feature ML。

**对应实验**

- 正式 5-fold few-shot。
- 任务优先选效果稳定、视觉直观的类别：
  - road
  - water
  - pitch/sports
  - park/green 可作为补充。
- 不建议主图用 building，因为假正例容易多；building 放主表即可。

**正文承接**

Main Results 第一节。先讲少样本能力，不要先讲模型架构细节。

### Figure 5. 多任务 benchmark 主表图

**要证明什么**

玄女不是只在单个好看的任务上有效，而是在多类城市制图任务上有稳定收益。

**怎么画**

一张论文风格的 grouped bar/heatmap：

- 行：任务中文+英文。
  - 建筑 Building
  - 道路 Road
  - 水体 Water
  - 施工地 Construction
  - 运动场 Sports/Pitch
  - 绿地 Green/Park
- 列：方法。
  - AEF + Linear
  - AEF + MLP
  - Xuannv + Linear
  - Xuannv + MLP
  - Traditional ML
  - Raw-feature strong head
- 指标：主图用 F1，副图或右侧小图用 AUC/AP。
- 结果必须写 mean ± std。

**对应实验**

- 统一 5-fold。
- 同 split、同标签、同阈值选择。
- AEF 和玄女下游头必须完全一致。

**正文承接**

Main Results 第二节。强调：

- 玄女 simple head 的收益代表 embedding 可读性。
- 空间增强头是应用上限，不作为 embedding 本体的唯一证据。

### Figure 6. P10/P11/P12 消融：为什么每一步升级有必要

**要证明什么**

模型改进不是靠堆实验，而是针对明确问题逐步解决：

- P10C：海淀专用月度 embedding。
- P11：提高有效维度和语义分离。
- P12：加入上下文，缓解 patch 边界割裂。

**怎么画**

四个 panel：

1. P10/P11/P12 全域 PCA 对比。
2. effective rank / top10 variance 对比。
3. 下游 F1 对比，任务分组。
4. patch 边界连续性指标：
   - 邻接 patch 边界两侧 embedding cosine gap。
   - 或 PCA 色差边界分数。

**对应实验**

- P10C epoch800。
- P11C best/e600。
- P12B/P12C epoch200/400。
- 所有模型统一导出 202604。

**需要补做**

- 边界连续性定量指标脚本。
- P12 epoch200 和 epoch400 下游统一评测。

**正文承接**

Ablation Study 章节。不要写“我们做了很多轮实验”，而要写：

- 问题 1：维度坍缩。
- 问题 2：patch 边界割裂。
- 问题 3：高分细节不足。
- 对应方法和证据。

### Figure 7. 定性案例：原图、GT、玄女、AEF、误差

**要证明什么**

定性案例用来解释模型在哪里好、哪里失败，而不是只挑最好看的图。

**怎么画**

每个任务选 2 个 patch：

- 原始高分光学。
- GT/OSM 标签。
- Xuannv probability。
- Xuannv prediction。
- AEF prediction。
- Error map：
  - TP 绿色
  - FP 红色
  - FN 蓝色

任务建议：

- road：展示线状结构连续性。
- water：展示边界和岛屿细节。
- pitch/sports：展示少样本语义检索/制图。
- construction：展示月度状态。

**对应实验**

- 从统一 5-fold 结果里自动选：
  - 1 个高 F1 patch。
  - 1 个典型失败 patch。
- 避免人工只挑好图。

**正文承接**

Qualitative Analysis。明确说失败案例：

- 建筑假正例。
- OSM 边界不准。
- 云雾/阴影影响。
- 年度 AEF 对月度状态不足。

### Figure 8. ROI 相似度检索

**要证明什么**

embedding 不只是能接 segmentation head，还能做交互式检索：用户点一个小区域，全域找相似地物。

**怎么画**

三组 query：

- 操场/运动场。
- 水体。
- 大型道路/立交。

每组：

1. Query patch 原图，框出 query ROI。
2. 全域 similarity heatmap。
3. Top-10 retrieved patches 小图。
4. top-k precision 曲线。

**对应实验**

- 使用 ROI 内 embedding 均值作为 query vector。
- 全域每个像素/小区域做 cosine similarity。
- 需要人工或 OSM 弱标签评估 top-k precision。

**正文承接**

Applications。这个图很适合展示“embedding 的产品潜力”，但如果没有人工评估，只能作为 qualitative demo；论文主指标需要 top-k precision。

### Figure 9. 月度状态与变化敏感性

**要证明什么**

月度 embedding 的优势在于表达同一区域不同月份的地表状态，而年度 embedding 只能给一个平均表征。

**怎么画**

选施工地/水体/裸地变化案例：

- 202512 原图。
- 202604 原图。
- 202512 embedding PCA。
- 202604 embedding PCA。
- embedding delta heatmap。
- 下游变化/施工地预测。

对比 AEF：

- AEF annual embedding 作为单期表示，无法直接区分月度状态。

**对应实验**

- 导出多个月份 embedding。
- 对施工地或水体季节变化做定性和少量定量。
- 如果没有可靠变化 GT，放应用示例，不放主表。

**正文承接**

Discussion 或 Applications。这里可以自然解释为什么我们做 monthly embedding，而不是只复刻 AEF annual embedding。

## 3. 表格设计

### Table 1. 数据集与训练配置

内容：

- 区域、patch 数、月份、数据源、分辨率、输入通道。
- 训练配置：epoch、batch、NPU、loss weights。
- P10/P11/P12 差异。

### Table 2. 主 benchmark 数字

内容：

- 任务 x 方法。
- F1、AUC、AP、mIoU。
- mean ± std。

### Table 3. 消融实验

内容：

- 去掉 highres。
- 去掉 OSM weak semantic probe。
- 去掉 hard masking。
- 128 vs context160。
- P10/P11/P12。

### Table 4. 局限与失败类别

内容：

- 哪些类别不稳定。
- OSM 噪声类型。
- 可能原因。
- 是否进入主 claim。

## 4. 正文如何根据图来补

**Introduction**

围绕 Figure 1 和 Figure 4 写。先讲问题：城市制图需要少样本、月度更新。再讲我们的解法：月度 dense embedding。

**Method**

围绕 Figure 1、Figure 2 写。不要堆公式，先讲输入、输出、训练目标、P12 context crop。

**Experiments**

围绕 Figure 4、Figure 5 写。先定义统一协议，再讲指标，最后放主结果。

**Ablation**

围绕 Figure 6 写。每个改动必须对应一个可量化问题。

**Analysis**

围绕 Figure 3、Figure 7 写。解释为什么模型有效，也解释为什么有假正例。

**Applications**

围绕 Figure 8、Figure 9 写。展示检索和月度状态能力，但不要让 demo 替代主指标。

**Limitations**

根据 Figure 7 和 Table 4 写。主动承认：

- OSM 弱标签同源。
- 单一区域。
- AEF 年度 vs 玄女月度。
- 部分类别不稳定。
- 需要跨区域和人工精标。

## 5. 补实验执行顺序

1. 等 P12B/P12C 跑完 400 epoch。
2. 对 P10C、P11C、P12B、P12C 统一导出 202604 embedding。
3. 先生成 Figure 3 和 Figure 6：
   - PCA 全域图。
   - effective rank。
   - boundary gap。
4. 用统一协议跑 Figure 5 主 benchmark：
   - AEF 和玄女同 head、同 split。
   - 5-fold mean/std。
5. 跑 Figure 4 few-shot：
   - 5/10/50 shot。
   - 展示训练 patch 和全域预测。
6. 生成 Figure 7 定性图：
   - 自动选高分和失败案例。
7. 生成 Figure 8 ROI 检索。
8. 生成 Figure 9 月度变化案例。
9. 根据图写正文初稿。

## 6. 对抗性审查清单

写完每张图后，用下面问题自查：

1. 这张图支撑哪个 claim？
2. 这张图有没有用不同协议的结果拼接？
3. AEF 是否使用同 split、同 head、同阈值？
4. 图中是否只展示了赢的类别？
5. 是否有 mean/std 或置信区间？
6. OSM 弱标签是否被误写成真值？
7. PCA 是否被过度解释？
8. 是否存在训练标签和评估标签同源风险？
9. 图注是否能让读者不看正文也理解图意？
10. 失败案例有没有展示？

## 7. 当前建议的主图取舍

正文主图控制在 7 张以内：

1. Figure 1：系统总览。
2. Figure 2：数据和 context crop。
3. Figure 3：表示空间质量。
4. Figure 4：few-shot 全域制图。
5. Figure 5：多任务 benchmark。
6. Figure 6：P10/P11/P12 消融。
7. Figure 7：定性案例与失败分析。

附录图：

- ROI 检索。
- 月度变化。
- 更多类别全域图。
- 训练曲线。
- 数据 QA。
- OSM 标签噪声案例。

这样主线最清楚：先说明我们做什么，再证明 embedding 本身有结构，再证明少样本制图有效，最后解释升级和失败。
