# 海淀区 P10C 筛选训练数据包

本文档记录 `haidian-embedding-v1` 生产版本使用的筛选后训练数据包。大文件已上传至 ModelScope，Git 仓库只保留打包脚本和说明文档。

## ModelScope 路径

数据集：

```text
WeijieWu/xuannv_haidian_embdding
```

训练数据目录：

```text
artifacts/haidian-embedding-v1/training_data/
  haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst
  haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst.sha256
  README.md
  manifest.json
  filelist.sha256
```

## 本地生成路径

```text
/data/xuannv_embedding/releases/haidian_embedding_v1_p10c_training_data/
  haidian_embedding_v1_p10c_filtered_training_data_202512_202605/
  haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst
  haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst.sha256
```

## 来源配置

```text
configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml
configs/production/haidian_embedding_v1.yaml
```

P10C 训练 manifest：

```text
/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
```

随机验证划分 manifest：

```text
/data/xuannv_embedding/processed/haidian/manifest_p5h_random20_val_seed42.json
```

## 数据内容

- `patches/s2/`：Sentinel-2 月内质量筛选后的输入影像及像素级 mask。
- `patches/s1/`：Sentinel-1 SAR 输入影像及像素级 mask。
- `patches/landsat/`：Landsat 月内质量筛选后的输入影像及像素级 mask。
- `patches/highres_optical_haidian/`：高分辨率光学重建目标及 mask。
- `patches/highres_sar_haidian/`：高分辨率 SAR 重建目标及 mask。
- `patches/worldcover/`：原始 WorldCover 辅助栅格，作为兼容数据保留。
- `labels/osm_*`：OSM 弱语义标签，包含建筑、道路、水体、绿地、农业、住宅、商业、工业、施工、铁路、步行路径、运动/游乐场等类别。
- `statistics/haidian/`：训练归一化统计量。
- `configs/`：实验配置和生产配置快照。
- `meta/`：manifest、随机验证划分、OSM 合并说明和质量筛选记录。

## 数据规模

```text
patch 数量: 320
月份: 2025-12 至 2026-05
manifest 引用文件: 52,496
OSM 标签文件: 4,182
统计文件: 5
总文件数: 56,692
压缩包大小: 约 13 GB
```

压缩包 SHA256：

```text
138a000ed438a16f39637e617e5b251c4864df11bff4ddc7b6a5eea93a5a44e0
```

## 校验命令

```bash
cd /data/xuannv_embedding/releases/haidian_embedding_v1_p10c_training_data
sha256sum -c haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst.sha256
tar --use-compress-program=zstd -tf haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst | head
```

## 打包脚本

```bash
python scripts/release/package_haidian_training_data.py
```

脚本会读取 P10C 配置和 manifest，收集训练真实引用的数据、OSM 弱标签、统计量、配置快照和元数据，并生成 `README.md`、`manifest.json`、`filelist.sha256` 与 `.tar.zst` 压缩包。
