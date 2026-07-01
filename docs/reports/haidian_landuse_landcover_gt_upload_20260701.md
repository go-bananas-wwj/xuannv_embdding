# 海淀区 Land Cover / Land Use GT 标签上传记录

## 概要

已将海淀区 V1 的 land cover 与 land use 真实值/弱真实值按逐 patch PNG 形式上传到 ModelScope。

其中：

- land cover 来源为 `processed/haidian/labels/worldcover`，对应 ESRI/WorldCover 风格 10m 土地覆盖标签。
- land use 来源为 OSM 语义 mask 融合，属于 OSM 弱标签，不是人工逐像素精标。

## ModelScope 路径

```text
haidian/v1/reports/haidian_land_cover_gt_patch_labels_20260701/
haidian/v1/reports/haidian_land_use_gt_patch_labels_20260701/
```

数据集：

```text
WeijieWu/xuannv_embdding_api
```

## 本机路径

```text
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_land_cover_gt_patch_labels_20260701/
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_land_use_gt_patch_labels_20260701/
```

## 目录结构

两个目录都包含：

```text
README.md
manifest.json
labels/*.png
index_masks/*.png
*_320_patch_grid.png
archives/*.zip
archives/checksums.sha256
```

说明：

- `labels/` 为彩色可视化 PNG。
- `index_masks/` 为单通道类别 id PNG，可供程序按类别值读取。
- `*_320_patch_grid.png` 为 320 patch 全域拼图。

## Land Cover 类别

| id | 类别 |
|---:|---|
| 0 | ignore/background |
| 1 | tree |
| 2 | shrub/grass |
| 3 | cropland |
| 4 | built-up |
| 5 | bare/other |
| 6 | snow/wetland |
| 8 | water |

像素统计：

| id | pixels |
|---:|---:|
| 0 | 440 |
| 1 | 96,324 |
| 2 | 227,487 |
| 3 | 413 |
| 4 | 499,247 |
| 5 | 3,488,678 |
| 6 | 3,102 |
| 8 | 927,189 |

上传统计：

- 文件数：645
- 上传失败数：0

## Land Use 类别

| id | 类别 |
|---:|---|
| 0 | other |
| 1 | residential |
| 2 | commercial |
| 3 | industrial |
| 4 | agriculture |
| 5 | green |
| 6 | water |
| 7 | construction |
| 8 | transport |

像素统计：

| id | pixels |
|---:|---:|
| 0 | 1,061,359 |
| 1 | 553,006 |
| 2 | 211,525 |
| 3 | 72,077 |
| 4 | 294,303 |
| 5 | 1,398,151 |
| 6 | 197,024 |
| 7 | 229,845 |
| 8 | 1,225,590 |

上传统计：

- 文件数：645
- 上传失败数：0

## 可视化

```text
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_land_cover_gt_patch_labels_20260701/haidian_land_cover_gt_320_patch_grid.png
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_land_use_gt_patch_labels_20260701/haidian_land_use_gt_320_patch_grid.png
```

