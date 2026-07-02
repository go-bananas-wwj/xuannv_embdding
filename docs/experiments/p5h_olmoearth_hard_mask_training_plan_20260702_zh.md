# P5H 海淀月度地物嵌入训练方案：参考 OlmoEarth 的困难重建版

日期：2026-07-02

## 结论先讲

下一版不要继续堆很多 loss。主线应该改成：

1. 用 S2、S1、Landsat、高分影像做月度地表状态建模。
2. 训练时随机遮住一部分输入源、月份、空间块，让模型学会从剩余信息恢复被遮住的内容。
3. OSM/WorldCover 只做低权重弱语义 target，不做强监督，不让模型偷看标签。
4. 先不加主线 location embedding，最多做一个 ablation。
5. AEF 蒸馏先不进主实验，只作为后续可选对照。

一句话：先把模型训练成“看影像能懂地物”，再谈蒸馏、位置编码和复杂任务。

## 调研依据

- AlphaEarth Foundations：公开材料强调 annual satellite embedding field、空间/时间/观测上下文融合、64 维像素级地理嵌入，但没有公开说明必须使用可学习的绝对 location embedding。参考：https://arxiv.org/abs/2507.22291
- Google Satellite Embedding V1：年度 10 m、64 维 embedding，适合做 AEF 基线和可选 teacher，不适合直接替代我们的月度目标。参考：https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- OlmoEarth：开源代码显示它用 Sentinel-2、Sentinel-1、Landsat 加 OSM、WorldCover、SRTM 等 derived maps，核心是 masked self-supervised / latent MIM / patch discrimination，并内置 kNN、linear probe 下游验证。参考：https://github.com/allenai/olmoearth_pretrain
- Prithvi / Clay 等地理基础模型共同经验：masked reconstruction、时间信息、多模态输入、冻结 embedding 的 linear/kNN probe 验证，比直接堆下游监督更稳。

## OlmoEarth 值得学的点

我对照了 OlmoEarth 开源训练脚本和我们当前代码，最值得学习的是这几件事。

| 方向 | OlmoEarth 做法 | 我们当前做法 | 建议 |
|---|---|---|---|
| 困难重建 | `random_time_with_decode`、`modality_cross_random`，随机遮时间、空间、模态 | 只有 `source_dropout_probs`，粒度较粗 | 增加系统化 masking |
| OSM/地图使用 | `only_decode_modalities`，OSM/WorldCover 等只作为 decode target | OSM 通过 semantic probe 进入训练，任务很多 | 改成弱语义 target，降低权重 |
| 损失设计 | 主 loss 是 patch discrimination / latent MIM，少量 InfoNCE | 重建、uniformity、temporal contrast、semantic probe 多路并行 | 收束到 2-3 个核心损失 |
| 下游验证 | 训练中周期性跑 kNN/linear probe | 训练后另跑，best 主要看 val loss | 把快速 probe 接入训练后流程 |
| 缺失模态 | masking 里显式处理 missing token/mask | 有 source mask，但困难场景不足 | 增强缺失模态训练 |

## 我们代码目前不够好的地方

1. 困难重建不够“真困难”。

当前 `prepare_batch` 里有 `source_dropout_probs`，可以随机丢掉某个 source，但它更像输入 dropout。OlmoEarth 的困难点更完整：可能遮一个模态、一个时间段、一组空间 patch，或者只让模型用影像去预测 OSM/WorldCover 这类地图 target。

2. OSM probe 任务太多，容易把训练拉散。

P4A/P3A 里 OSM probe 同时包含 building、road、water、residential、commercial、industrial、agriculture、green、poi、rail、construction 等很多任务。每个都有自己的噪声和正负样本偏差。它们全进训练，会让 embedding 变成“讨好一堆噪声标签”，而不是学通用地表状态。

3. 当前重建太像“看见什么复原什么”。

如果输入和 target 都完整，模型可能学到光谱/纹理复制，而不一定形成清晰语义。我们需要让模型经常在缺 S1、缺 S2、缺某个月、缺高分图的情况下仍然恢复目标。

4. `highres_fusion_to_embedding: false` 是对的，但高分信息没有被充分用好。

高分影像应该主要作为 target 或辅助重建信号，而不是直接融进 embedding 输入。否则模型可能只在有高分月份好，没有高分月份弱。

5. best checkpoint 选择还不够贴近目标。

现在 best 多看 validation loss。下一版应该同时看冻结 embedding 的 few-shot/linear probe，尤其 building、road、water、green、land-use、construction。

## 困难重建怎么做

我建议分三类 mask。

### 1. 模态遮挡

训练时随机遮掉某些输入源：

- 遮 S2，让模型用 S1/Landsat/历史月份恢复 S2。
- 遮 S1，让模型用 S2/Landsat 恢复 S1。
- 遮 Landsat，让模型用 S2/S1 恢复 Landsat。
- 遮高分影像输入，高分只作为 target，不作为稳定输入依赖。

建议概率：

| source | 遮挡概率 |
|---|---:|
| S2 | 0.15 |
| S1 | 0.20 |
| Landsat | 0.25 |
| highres_optical | 0.60 |
| highres_sar | 0.50 |

### 2. 时间遮挡

随机遮住 1-2 个月，让模型从前后月份和其他模态恢复该月。

这对我们特别重要，因为目标是月度 embedding。模型必须知道“每个月自己是什么状态”，不能把所有月份压成一样。

### 3. 空间块遮挡

随机遮住 128x128 内的一些小块，例如 16x16 或 32x32 token block。

作用是让模型学上下文关系：道路是连续线，建筑是块状，水体/绿地有空间纹理，而不是只记单点光谱。

## 新训练目标

主实验只保留三个损失：

1. `masked_recon_loss`
   被遮住的 S2/S1/Landsat/高分目标重建。连续模态用 L1 或 SmoothL1。

2. `weak_map_loss`
   预测 OSM/WorldCover 弱语义图。建议只先用 building、major_road、water、green、agriculture、construction 六类。低权重，缺失不当负样本。

3. `uniformity_loss`
   小权重保留，防止 embedding collapse。权重不超过 0.005-0.01。

暂时关闭：

- supervised change loss
- temporal endpoint separation
- 强 temporal contrast
- AEF distillation
- 主线 location embedding

## 模型结构建议

不大改架构，先小步吸收 OlmoEarth 经验：

1. 保持输出 `embedding_map: [B, T, D, 128, 128]`。
2. `embed_dim` 先用 64，成功后再试 96。
3. 保留 month embedding / timestamp encoding。
4. 增加 modality embedding 或 source-aware token，让模型知道输入来自哪个传感器。
5. 不加绝对 patch id / 可学习 location embedding。
6. 高分影像不直接作为必需输入，优先作为被预测 target。

## 代码改造计划

### P5H-0：先做 mask 框架

新增一个训练时 masking 模块：

- 文件建议：`src/xuannv_embedding/training/masking.py`
- 输入：prepared batch
- 输出：masked model input、原始 target、mask metadata
- 支持：
  - modality dropout
  - month dropout
  - spatial block dropout
  - target-only weak maps

### P5H-1：改 `prepare_batch`

让 target 在遮挡前构造，模型输入在遮挡后构造。

目标是：模型看不到被遮住的源，但 loss 还能用原始 target 监督它恢复。

### P5H-2：改 loss

新增或改造：

- `masked_reconstruction_loss`
- `weak_map_loss`
- `mask_stats` 日志

训练日志必须记录：

- 每个 source 被遮比例
- 每个月被遮比例
- 有效 target 像素数量
- weak map 正样本比例

### P5H-3：配置一份自包含 YAML

新配置建议：

`configs/v2_p5h_haidian_olmo_masked_mim_20260702.yaml`

原则：

- 只训练海淀区。
- 所有字段写全，不使用 `_base_`。
- 训练 8 小时起步，6 NPU。
- eval 每 10 epoch，save 每 20 epoch。

### P5H-4：训练后固定流程

训练后自动跑：

1. 导出 best embedding。
2. 跑海淀空间留块 few-shot/linear/MLP probe。
3. 任务包括 building、road、water、green、land-use、construction。
4. 和 P4A、AEF 做同表对比。
5. 生成可视化报告。

## 推荐训练参数

| 项目 | 建议 |
|---|---|
| 设备 | NPU 0-5 |
| batch size | 6 起步，稳定后试 8 |
| epoch | 不固定看 epoch，先按 8 小时训练 |
| lr | `5e-5` 起步 |
| warmup | 5%-10% 总步数 |
| weight decay | `0.05` |
| AMP | 开 |
| gradient checkpointing | 开 |
| masked recon 权重 | 1.0 |
| weak map 权重 | 0.05 起步，最高 0.1 |
| uniformity 权重 | 0.005 |

## 需要验证的 ablation

第一轮不要太多，只做三个。

1. P5H-A：只有 masked recon，不加 OSM。
2. P5H-B：masked recon + OSM 六类弱语义。
3. P5H-C：P5H-B + 更强时间遮挡。

判断标准不是训练 loss 最低，而是冻结 embedding 后：

- linear probe 建筑/道路/水体是否提升；
- few-shot MLP 是否比 P4A 明显好；
- PCA/UMAP 是否能把建筑、水体、绿地、施工地分开；
- 月份之间是否不再完全一样。

## AEF 蒸馏是否加入

暂时不加。

原因：

- AEF 是年度 embedding，我们要做月度 embedding。
- 直接蒸馏可能把月度变化压平。
- 如果 P5H-B 已经明显提升，就没必要复杂化。

后续可选：

- 只在稳定区域低权重蒸馏 AEF。
- 只蒸馏 patch-level semantic prior，不蒸馏月度变化区域。
- 作为 P5H-D ablation，而不是主线。

## 下一步执行顺序

1. 审查海淀 manifest 和 OSM 六类标签路径，确认 target 完整。
2. 实现 `training/masking.py`。
3. 改 `prepare_batch`，保证 target 先构造、输入后遮挡。
4. 改 loss 和 trainer 日志。
5. 写自包含 P5H 配置。
6. 跑 CPU/smoke 测试。
7. 6 NPU 启动 8 小时训练。
8. 保存首个 checkpoint 后检查 loss、mask 比例、样例可视化。
9. 训练完成后跑固定下游评估和 AEF 对比。

## 我建议马上做的版本

先做 P5H-B：

`masked recon + OSM 六类弱语义 + uniformity`

这是最符合你要求的版本：不靠旧变化检测标签，不用 LocateAnything 伪标签，不让 OSM 主导模型，又能把海淀的建筑、道路、水体、绿地、农地、施工地语义补进 embedding。
