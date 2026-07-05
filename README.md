# 玄女海淀区地理嵌入生产版

生产版本：`haidian-embedding-v1`

分支：`haidian`

ModelScope 数据集：[WeijieWu/xuannv_haidian_embdding](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding)

## 这个版本是什么

这是面向北京市海淀区的月度地理 embedding 生产包。模型输入多源遥感数据，输出 128×128 空间分辨率的 64 维地理嵌入。下游只用简单 MLP 头，就可以做建筑提取、道路提取和水体提取。

本生产版选择实验 `P10C epoch800` 作为当前最优候选：

- 训练区域：海淀区 320 个 patch
- 训练月份：2025-12 至 2026-05
- 生产评估月份：2026-04
- 输出分辨率：128×128，约 10m 等效分辨率
- embedding 维度：64

## 当前效果

快速验收方式：202604 月 embedding，固定 fold-0，简单 MLP 下游头。

| 任务 | AP | F1_best |
|---|---:|---:|
| 建筑提取 | 0.4392 | 0.4832 |
| 道路提取 | 0.5726 | 0.5232 |
| 水体提取 | 0.6092 | 0.6306 |
| 平均 | - | 0.5457 |

## 320 Patch 全域可视化

以下图片保存在 ModelScope 数据集的 `artifacts/haidian-embedding-v1/visualizations/` 目录。

### Embedding PCA

![P10 embedding PCA](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding/resolve/master/artifacts/haidian-embedding-v1/visualizations/p10_embedding_pca_202604_compare.png)

### 建筑提取

![Building prediction](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding/resolve/master/artifacts/haidian-embedding-v1/visualizations/building/building_gt_p10a_p10b_p10c_prediction_compare.png)

### 道路提取

![Road prediction](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding/resolve/master/artifacts/haidian-embedding-v1/visualizations/road/road_gt_p10a_p10b_p10c_prediction_compare.png)

### 水体提取

![Water prediction](https://modelscope.cn/datasets/WeijieWu/xuannv_haidian_embdding/resolve/master/artifacts/haidian-embedding-v1/visualizations/water/water_gt_p10a_p10b_p10c_prediction_compare.png)

## ModelScope Artifacts

大文件不进入 Git，统一放在 ModelScope：

```text
artifacts/haidian-embedding-v1/
  checkpoints/
    haidian_embedding_v1_p10c_epoch800.pt
  embeddings/
    haidian_202512_202605_p10c_epoch800/
  downstream_heads/
    building_mlp_fold0_best.pt
    road_mlp_fold0_best.pt
    water_mlp_fold0_best.pt
  visualizations/
    p10_embedding_pca_202604_compare.png
    building/
    road/
    water/
  manifests/
    haidian_artifacts_manifest.json
```

详细清单见 [docs/production/haidian_artifacts_manifest.json](docs/production/haidian_artifacts_manifest.json)。

## 本仓库包含什么

```text
configs/production/haidian_embedding_v1.yaml
scripts/production/export_haidian_embedding.py
scripts/production/run_haidian_downstream_probe.py
scripts/report/generate_p10_full_domain_visuals.py
docs/production/haidian_data_card.md
docs/production/haidian_metrics_report.md
docs/production/haidian_model_card.md
docs/production/haidian_artifacts_manifest.json
```

## 导出海淀区 Embedding

先从 ModelScope 下载生产权重到本地，然后执行：

```bash
cd /root/workspace/xuannv
export PYTHONPATH=$PWD/src:$PWD/downstreams:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0

python scripts/production/export_haidian_embedding.py \
  --checkpoint /path/to/haidian_embedding_v1_p10c_epoch800.pt \
  --output-root /data/xuannv_embedding/embeddings/production \
  --months 202512 202601 202602 202603 202604 202605 \
  --device npu:0
```

## 跑下游任务

```bash
python scripts/production/run_haidian_downstream_probe.py \
  --embedding-root /data/xuannv_embedding/embeddings/production/<export_dir> \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_downstream \
  --month 202604 \
  --tasks building road water \
  --device npu:0 \
  --save-predictions
```

## 训练数据与训练思路

训练数据只使用海淀区 320 个 patch，时间为 2025-12 至 2026-05。输入包含 Sentinel-2、Sentinel-1、Landsat、高分辨率光学、高分辨率 SAR，以及 OSM 弱语义标签。

训练目标包括：

- 多源重建，让 embedding 保留真实地表信息。
- 高分辨率重建，让 embedding 保留局部纹理和边界。
- OSM 弱语义，让 embedding 对建筑、道路、水体等地物更可分。
- 困难重建，随机遮挡模态、月份和空间块，提高对缺失和噪声的鲁棒性。

## 已知限制

- OSM 是弱标签，存在漏标、错标和时间滞后。
- 当前版本是海淀区专用生产版，不承诺跨城市泛化。
- README 指标是 quick eval，不是完整 5-fold 产品验收。
