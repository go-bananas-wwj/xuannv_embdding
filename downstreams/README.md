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
