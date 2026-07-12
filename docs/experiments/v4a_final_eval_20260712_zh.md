# V4A 最终评测报告（从零 + DINOv3 蒸馏，epoch 800，2026-07-12）

- 实验：`v4a_distill_scratch_haidian_harbin_202512_202605_20260711`
  （从零初始化 + DINOv3-SAT493M dense 蒸馏 cos 0.5 / Gram 0.2 + 双区域 744 patch + v3 语义组件）
- 检查点：`/data/xuannv_embedding/outputs/v4a_distill_scratch_haidian_harbin_202512_202605_20260711/epoch_800.pt`
- 评测产物：`/data/xuannv_embedding/experiments/v4a_eval_20260712/`
- 训练：val_loss 3.36 → 0.684 全程单调下降无平台，收敛形态显著优于 P14A（2.27 平台）。

## 1. 总体结论：**蒸馏带来首个大幅突破（construction +0.05 F1 / +0.07 AP），
但类中心分离度退化导致 few-shot/检索恶化——方向已验证，配方需迭代（V4B）**

## 2. conv3x3 probe fold-0（F1_best / AP）

| 模型 | construction | building | road | water |
| --- | --- | --- | --- | --- |
| P10C e800（生产） | 0.5495 / 0.5435 | **0.4842** / **0.4497** | **0.5218** / 0.5664 | **0.6237** / 0.6041 |
| P14A e800（v3 最优） | 0.5292 / 0.3543 | 0.4830 / 0.4385 | 0.5203 / 0.5604 | 0.6253 / 0.6239 |
| **V4A e800** | **0.5988** / **0.6105** | 0.4651 / 0.4150 | 0.5183 / **0.5758** | 0.5962 / 0.5979 |

- **construction F1 0.5988（+0.0493）、AP 0.6105（+0.0670）**：项目史上首次 0.05 量级突破，
  且 AP 提升 12%。这正是蒸馏的预期效果——DINOv3 对施工场地这类复杂纹理场景的
  语义理解远超我们的小模型自学能力。road AP 0.5758 也超基线。
- building/water 回退 0.02~0.03：从零+双区域训练摊薄了海淀特定拟合。

## 3. Embedding 空间诊断

| 模型 | 有效秩 | Top10 方差 | centroid AUC | between/within |
| --- | ---: | ---: | ---: | ---: |
| P10C | 9.69 | 0.933 | 0.723 | — |
| P14A | 12.03 | 0.895 | 0.7494 | 0.736 |
| **V4A** | **14.38** | **0.854** | 0.6321 ❌ | 0.379 ❌ |
| AEF 官方 | 14.79 | 0.813 | 0.684 | 0.488 |

- **有效秩 14.38 逼近 AEF（14.79）**，维度利用率彻底解决；
- 但 centroid AUC / between-within 大幅退化，embedding 分布形态趋向 AEF 的
  "均匀通用"型——**类中心可分性被蒸馏+uniformity 挤压**，这直接解释了
  few-shot（1/18 胜）与检索（三任务全低）的恶化。

## 4. few-shot 与检索

- few-shot 6 任务×3 shots 仅 building-50shot 一项胜 P10C；education/sports_pitch
  等小样本类退化最重（-0.09~-0.17）——这些正是依赖类中心分离的任务。
- 检索：building 0.3180 / road 0.3970 / water 0.5554，均低于 P10C。

## 5. 归因

1. **蒸馏被验证是大杠杆**：construction 是四任务中最难的（弱标签、复杂纹理），
   +0.067 AP 的提升量级是 P11-P14 所有微调的 10 倍以上。
2. **失衡在于"通用 vs 判别"**：DINOv3 教师特征是任务无关的通用表示，加上
   uniformity 0.16，把 64 维空间推向均匀分布；v3 的原型对比（0.12）力度不足以
   在其上重建类中心结构。P14A 证明原型对比能提升 cAUC 至 0.7494，V4A 中它被
   蒸馏信号淹没。
3. building/water 回退与双区域训练相关（哈尔滨占 57% 样本），海淀任务权重被稀释。

## 6. V4B 配方（下一步，已明确）

保留蒸馏主杠杆，重建判别结构：
1. **蒸馏权重余弦衰减**：0.5 → 0.1（前 400 epoch 灌知识，后 400 epoch 让位判别目标）；
2. **原型对比 0.12 → 0.25 + warmup 200**（在蒸馏底座上重建类中心）；
3. **uniformity 0.16 → 0.10**（有效秩已 14+，不再需要强 uniformity）；
4. 海淀 patch 采样权重 ×1.5（对冲双区域稀释）；
5. 其余保持 V4A（数据、从零、Gram anchoring 不变）。

预期：保住 construction 突破 + 有效秩 ≥13，把 cAUC 拉回 0.74+，few-shot/检索转正。

## 7. 产物索引

- conv3x3：`{root}/benchmarks/v4a_e800/`；诊断：`{root}/diagnostics/v4a_e800/`
- few-shot：`{root}/strong_downstream_v4a_e800_{s0,s1,s2}/`；检索：`{root}/vector_retrieval_v4a_e800/`
- `{root}` = `/data/xuannv_embedding/experiments/v4a_eval_20260712`
- 注意：v2 manifest patch 目录带 `haidian_` 前缀，已在 embedding 目录内建符号链接兼容评测工具。
