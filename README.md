# 玄女 Xuannv：月度地理嵌入

玄女是一个面向遥感制图的月度地理 embedding 项目，参考
[AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) 的思路，将多源时序遥感观测压缩为可复用的密集特征场。

它不是为建筑、道路、水体分别训练三套大分割模型。模型会为每个
`1280 m x 1280 m` patch 输出一张 `128 x 128 x 64` 的 embedding map；后续只需少量标注和一个轻量下游头，就可以快速训练不同地物的制图器。

## 当前版本

| 项目 | 当前状态 |
| --- | --- |
| 生产 embedding | `haidian-embedding-v1`，海淀 P10C `epoch_800` |
| 生产区域与时段 | 海淀区 320 个 patch，2025-12 至 2026-05 |
| 输出 | 10 m 等效分辨率、64 维月度 embedding |
| 默认制图头 | `binary_conv3x3` |
| 公平 probe / 消融 | `linear`、`mlp`、`wide_mlp`、`deep_wide_mlp` |
| 论文实验 | 40 组空间独立实验正在运行，进度见 `docs/paper/` |

**下游业务制图默认使用 `binary_conv3x3`，不是 MLP。** 它在 64 维 embedding 上增加很小的 3x3 局部空间上下文，对建筑边界、细道路和水体连续性更稳定。`MLP` 仍保留为公平、逐像素的 probe：它用于回答“embedding 本身是否容易被读出”，不再是默认的全域制图头。

> 说明：最早发布到 ModelScope 的 P10C V1 artifact 中包含 MLP fold-0 头，供历史复现使用；新的业务流程请按下文训练或加载 `binary_conv3x3` 头。不要把这两个用途混为一谈。

## 能力概览

### 2026-04 全标签轻量制图

同一海淀划分、验证集选阈值、测试集报告。玄女输入为 64 维 embedding，下游头为 `conv3x3`。

| 任务 | F1 | AP | AUC | 验证集阈值 |
| --- | ---: | ---: | ---: | ---: |
| 建筑提取 | 0.484 | 0.450 | 0.896 | 0.600 |
| 道路提取 | 0.520 | 0.566 | 0.834 | 0.578 |
| 水体提取 | 0.623 | 0.604 | 0.891 | 0.818 |

完整的 raw-image、AEF 与不同下游头对照见
[下游头与原始影像基线报告](docs/production/haidian_raw_image_baseline_comparison_20260708.md)。

### Few-shot 快速制图

`5/10/50-shot` 表示仅标注对应数量的正样本 patch，并配同等数量的负样本 patch；验证和测试仍在固定全域划分上完成。以下均为玄女 embedding + `conv3x3`：

| 任务 | 5-shot F1 | 10-shot F1 | 50-shot F1 |
| --- | ---: | ---: | ---: |
| 建筑 | 0.459 | 0.454 | 0.490 |
| 道路 | 0.474 | 0.487 | 0.517 |
| 水体 | 0.613 | 0.612 | 0.631 |

在 5-shot 和 10-shot 下，玄女在建筑、道路、水体三类任务均优于同协议的原始多源影像强头；50-shot 下建筑、道路仍领先，水体与 raw UNet 基本持平。详见
[Few-shot 对比报告](docs/production/haidian_fewshot_raw_vs_xuannv_20260708.md)。

## 全域结果

### 月度 embedding PCA

![海淀区 2026-04 embedding PCA](docs/production/assets/haidian_v1_20260708/p10_embedding_pca_202604_compare.png)

### 三类核心任务

![建筑、道路与水体全域结果](docs/production/assets/haidian_v1_20260708/raw_vs_xuannv_f1.png)

更多单 patch 的原始光学、GT、概率图和二值制图对照位于
[`docs/production/assets/haidian_v1_20260708/`](docs/production/assets/haidian_v1_20260708/)。

## 模型如何训练

每个月独立读取该月观测；训练时从连续月份窗口中采样，输出仍是一张对应目标月的 128x128 embedding map。

```text
Sentinel-2 (12 bands) ─┐
Sentinel-1 (VV/VH) ────┼─> STP multi-source temporal encoder
Landsat (7 bands) ────┤       + optional high-resolution fusion
High-resolution optical/SAR ──┘
                                  -> vMF bottleneck -> 64-D embedding map
                                  -> multi-source reconstruction / OSM weak semantics
```

训练使用以下信号，而不是使用最终下游任务的人工标注：

- Sentinel-2、Sentinel-1、Landsat 的月度质量筛选观测与像素有效掩膜；
- 稀疏高分光学与高分 SAR 重建，用于保留细边界和纹理；
- 清洗、合并后的 OSM 广义弱语义，用于提供建筑、道路、水体、绿地、土地利用等地物先验；
- 困难重建：随机遮挡数据源、月份和空间块，让模型学会在云雾、缺失观测和不完整模态下保持语义稳定。

完整模型卡、训练数据包与 artifact 清单：

- [模型卡](docs/production/haidian_model_card.md)
- [训练数据包说明](docs/production/haidian_training_data_package_20260705.md)
- [下游评测协议](docs/downstream_eval_protocol.md)
- [长期实验记忆与规范](docs/agent_memory/haidian_embedding_project_memory.md)

## 快速开始

### 1. 安装

```bash
pip install -e ".[downstream]"
export PYTHONPATH="$PWD/src:$PWD/downstreams:$PYTHONPATH"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### 2. 导出月度 embedding

从 ModelScope 获取生产 checkpoint 后：

```bash
python scripts/production/export_haidian_embedding.py \
  --checkpoint /path/to/haidian_embedding_v1_p10c_epoch800.pt \
  --output-root /data/xuannv_embedding/embeddings/production \
  --months 202512 202601 202602 202603 202604 202605 \
  --device npu:0
```

6 卡分片导出：

```bash
bash scripts/production/export_haidian_v1_embeddings.sh
```

### 3. 用 Conv 3x3 训练一个下游制图头

复制 `downstreams/configs/v2_probe_binary_conv3x3_single_202604.yaml`，将 `head_type` 保持为 `binary_conv3x3`，并按任务填写标签根目录。示例：

```bash
python downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config downstreams/configs/v2_probe_binary_conv3x3_single_202604.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/production/<export_dir> \
  --label-root /data/xuannv_embedding/processed/haidian/labels/building_osm \
  --region haidian \
  --output-root /data/xuannv_embedding/outputs/downstream/building_conv3x3 \
  --fold 0 \
  --save-predictions
```

该流程会在验证集选择阈值并输出 F1、AP、AUC、概率图和红白二值制图图。建筑、道路、水体和其他 OSM 类别的详细用法见
[下游头使用指南](docs/production/downstream_fewshot_head_usage_guide_20260709.md)。

## 发布内容

大文件不进入 Git，发布在 ModelScope 数据集
[`WeijieWu/xuannv_haidian_embdding`](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding)：

```text
artifacts/haidian-embedding-v1/
  checkpoints/       # P10C epoch-800 生产 embedding 权重
  embeddings/        # 2025-12 至 2026-05 月度 embedding 压缩包
  downstream_heads/  # 已发布头；旧 MLP 仅用于历史复现
  metrics/           # 下游指标与阈值记录
  visualizations/    # PCA、全域制图和单 patch 图
  manifests/         # SHA-256、来源与版本信息
```

仓库中的代码、配置和文档可复现训练、导出、下游评测和全国数据准备；原始影像、checkpoint、embedding 和大规模日志保留在数据盘或 ModelScope。

## 仓库结构

```text
src/xuannv_embedding/       # 多源月度 embedding 模型、数据与训练
downstreams/                # 下游头、任务、指标、可视化
configs/production/         # 海淀生产配置
configs/national/           # China V1 全国采样与准备策略
scripts/production/         # 导出、打包、上传 ModelScope
scripts/data/               # 下载、质量控制、全国分片
docs/production/            # 模型卡、评测报告、发布清单
docs/paper/                 # 论文实验协议与写作框架
```

## 全国版与论文

全国版 `China V1` 正在构建全国空间分层样本、月度 Sentinel-2/Sentinel-1/Landsat 数据和冻结的 OSM 弱语义快照。目标是从单城生产验证扩展到可复现的全国月度 embedding；其数据门禁、存储布局和质量检查见
[全国数据准备计划](docs/plans/china_v1_national_data_preparation_20260717.md)。

论文实验采用空间独立 5-fold、3-seed 下游评测，严格分开“全域传导式制图”和“空间泛化”结论。协议见
[论文实验协议](docs/paper/xuannv_paper_experiment_protocol_20260715_zh.md)。

## 限制与使用边界

- 海淀 V1 是单城生产验证，不应直接宣称跨城市泛化；
- OSM 是弱语义标签，存在漏标、错标和时间滞后，未知区域不应当作背景；
- 月度输入比年度合成更容易受云雾、阴影和观测稀疏影响，训练与导出必须使用像素级质量掩膜；
- 完整监督的原始影像 + UNet/DeepLab-lite 仍是强基线。玄女的核心价值是少量标注下的快速制图、跨任务复用和低存储 embedding 产品。
