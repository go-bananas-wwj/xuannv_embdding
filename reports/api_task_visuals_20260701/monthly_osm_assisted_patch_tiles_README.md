# 海淀 V1 月度 OSM 辅助 Patch 诊断图

ModelScope 路径：

```text
haidian/v1/reports/monthly_osm_assisted_patch_tiles/
```

该目录保存 2025 年 12 月到 2026 年 5 月的月度 patch 诊断图，覆盖建筑物提取、道路提取、水体提取、施工工地提取。每张图参考哈尔滨新区 OSM diagnostic 条带格式，包含影像底图、OSM 标注、每 patch 阈值结果、后处理结果、误差图。

为避免逐张上传 7680 个小文件，ModelScope 上按月份提供 tar 包：

| 月份 | ModelScope 文件 |
|---|---|
| 202512 | `archives/haidian_v1_202512_monthly_osm_assisted_patch_tiles.tar` |
| 202601 | `archives/haidian_v1_202601_monthly_osm_assisted_patch_tiles.tar` |
| 202602 | `archives/haidian_v1_202602_monthly_osm_assisted_patch_tiles.tar` |
| 202603 | `archives/haidian_v1_202603_monthly_osm_assisted_patch_tiles.tar` |
| 202604 | `archives/haidian_v1_202604_monthly_osm_assisted_patch_tiles.tar` |
| 202605 | `archives/haidian_v1_202605_monthly_osm_assisted_patch_tiles.tar` |

校验文件：

```text
archives/checksums.sha256
```

注意：这些图是 OSM-assisted / oracle 诊断图层，用于汇报展示和误差分析，不是纯模型逐月预测结果。建筑、道路、施工使用模型概率结合 OSM 每 patch 最佳阈值和地类后处理；水体没有独立模型 head，使用 OSM 水体层生成。
