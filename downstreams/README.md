# xuannv_embedding downstream tasks

本目录存放预训练 `AEFModel` 的下游任务评测框架。

## 安装

```bash
pip install -e ".[downstream]"
```

## 数据准备

```bash
python downstreams/scripts/prepare_downstream.py \
  --task construction_site \
  --region haidian \
  --labelme-rar /data/xuannv_embedding/raw/haidian/labels/6511215751_/haidianlabel.rar \
  --patch-dir /data/xuannv_embedding/processed/haidian/patches/s2 \
  --out-dir /data/xuannv_embedding/downstream/labels/haidian/construction_site
```

## 预生成 embedding

阶段二训练完成后：

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --regions haidian harbin \
  --output-root /data/xuannv_embedding/embeddings
```

## 训练 + 评测

```bash
python downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_a1b2c3d \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2
```

## 可视化

```bash
python downstreams/scripts/visualize_results.py \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2 \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --rgb-source /data/xuannv_embedding/processed/haidian/patches/s2
```

## 目录结构

```
downstreams/
├── downstreams/          # Python 包
│   ├── data/             # label loaders / split / EmbeddingDataset
│   ├── heads/            # 任务头（linear / FCN / UNet / UperNet / classification / change-detection stub）
│   ├── metrics/          # 指标与可视化
│   ├── tasks/            # BaseTask / ConstructionSegmentationTask
│   └── utils/            # device / reproducibility
├── scripts/              # CLI
├── configs/              # 下游任务配置
└── tests/                # 单元测试
```

## Mask 规范

- 格式：单波段 `uint8` GeoTIFF，与参考影像同尺寸、同 CRS、同 Affine。
- 像素值：`0`=背景，`1`=建筑工地（`jiazhudongdi`）。
- 命名：`{patch_id}.tif`，例如 `patch_000002.tif`。

## Random-init 基线

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --random-init \
  --regions haidian \
  --output-root /data/xuannv_embedding/embeddings

python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/YYYYMMDD_xxx_random \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_random
```

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `embedding 不存在` | 月份不匹配 | 确认 `--month` 与预生成目录名一致 |
| `split_5fold.json` 不存在 | 首次运行 | 脚本会自动按正像素比例分层生成 |
| mIoU 接近 0 | 类别极度不平衡 | 检查 `pos_weight` 与 Focal loss 配置 |

