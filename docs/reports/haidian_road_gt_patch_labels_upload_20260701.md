# 海淀区道路 GT 标签上传记录

## 概要

已将海淀区道路提取任务的 OSM GT 标签逐 patch PNG 上传到 ModelScope。

该标签来自本机 `road_osm` 栅格标签，属于 OSM 自动标注/弱标签，不是人工逐像素精标。

## ModelScope 路径

```text
haidian/v1/reports/haidian_road_gt_patch_labels_20260701/
```

数据集：

```text
WeijieWu/xuannv_embdding_api
```

## 本机路径

```text
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_road_gt_patch_labels_20260701/
```

## 内容

```text
README.md
manifest.json
haidian_road_gt_320_patch_grid.png
labels/haidian_road_gt_patch_000000.png
...
labels/haidian_road_gt_patch_000319.png
archives/haidian_road_gt_patch_labels_20260701.zip
archives/checksums.sha256
```

## 统计

- patch 总数：320
- 有道路标签 patch：317
- 空白 patch：3
- 道路像素总数：819,511
- 平均道路像素比例：0.156309
- 上传文件数：325
- 上传失败数：0

## 可视化

全域 320 patch 拼图：

```text
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/haidian_road_gt_patch_labels_20260701/haidian_road_gt_320_patch_grid.png
```

