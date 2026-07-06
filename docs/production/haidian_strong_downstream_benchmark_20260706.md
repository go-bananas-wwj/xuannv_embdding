# 玄女海淀 V1 强下游模型评测方案

## 目标

本轮评测用于回答一个更贴近业务的问题：如果传统方案或强监督分割模型可以使用更多标注数据，玄女海淀 V1 embedding 是否仍然有优势；如果玄女原来的线性/MLP 下游头不够强，换成更强的轻量分割头后能释放多少潜力。

## 对比对象

1. 玄女 embedding + 更强下游头：
   - `pixel_conv`：逐像素 1x1 卷积头，相当于增强版 MLP。
   - `unet`：轻量 U-Net，利用局部空间上下文。
   - `deeplab_lite`：轻量 DeepLab 风格，加入多尺度空洞卷积。
   - `segformer_lite`：轻量 SegFormer 风格，多尺度特征融合。

2. 强监督 raw-feature 分割模型：
   - 输入为同月 S2/S1/Landsat/高分光学/高分 SAR 的指数特征。
   - 使用同样的 `unet`、`deeplab_lite`、`segformer_lite`。
   - 训练标签预算使用 `50-shot` 和 `full train`，模拟标注更充分的业务环境。

3. 已完成的传统机器学习基线：
   - RandomForest、ExtraTrees、HistGradientBoosting、Logistic、KNN 等。
   - 结果目录：`/data/xuannv_embedding/experiments/production/haidian_v1_traditional_ml_benchmark_20260706_phaseB_report`

## 公平性规则

- 同一任务使用同一套 OSM 弱标签和同一个 fold split。
- `shot=N` 表示最多 N 个正样本 patch，并配套相同数量的负样本 patch。
- 训练集可以做正负均衡采样；验证集和测试集使用全像素指标。
- 阈值只在验证集选择，测试集主指标使用 `F1@val_threshold`。
- `f1_best` 只作为诊断项，不作为主对比指标。
- 全域 320 patch 可视化只用于展示，不参与模型选择和阈值选择。

## 任务

正式评测任务：

- 建筑：`building`
- 道路：`road`
- 水体：`water`
- 公园/绿地：`park_green`
- 教育/学校：`education`
- 体育场地：`sports_pitch`

## 推荐执行

玄女增强下游头：

```bash
python scripts/eval/run_strong_downstream_benchmark.py \
  --embedding-root /data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604 \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_strong_downstream_xuannv_20260706 \
  --tasks building road water park_green education sports_pitch \
  --feature-sets xuannv_embedding \
  --models pixel_conv unet deeplab_lite segformer_lite \
  --shots 5 10 50 \
  --epochs 80 \
  --batch-size 8 \
  --base-channels 32 \
  --device npu:0
```

强监督 raw-feature 分割模型：

```bash
python scripts/eval/run_strong_downstream_benchmark.py \
  --embedding-root /data/xuannv_embedding/embeddings/production/haidian_202512_202605_p10c_epoch800_202604 \
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_strong_downstream_raw_20260706 \
  --tasks building road water park_green education sports_pitch \
  --feature-sets s2_s1_landsat_highres_indices \
  --models unet deeplab_lite segformer_lite \
  --shots 50 full \
  --epochs 100 \
  --batch-size 8 \
  --base-channels 32 \
  --device npu:1
```

## 当前实现状态

- 新增脚本：`scripts/eval/run_strong_downstream_benchmark.py`
- 已通过 NPU smoke test：
  - 任务：`water`
  - 特征：`xuannv_embedding`、`s2_s1_landsat_highres_indices`
  - 模型：`pixel_conv`、`unet`
  - 输出：`/data/xuannv_embedding/experiments/production/haidian_v1_strong_downstream_smoke_20260706`

## 后续报告口径

主报告建议按三层结论组织：

1. 少标注能力：玄女 5/10-shot 是否接近或超过 raw-feature 50-shot。
2. 强 head 潜力：玄女从 `pixel_conv` 升级到 U-Net/DeepLab-lite/SegFormer-lite 后提升多少。
3. 业务上限对比：raw-feature full train 的强监督模型与玄女 50-shot/增强头的差距。
