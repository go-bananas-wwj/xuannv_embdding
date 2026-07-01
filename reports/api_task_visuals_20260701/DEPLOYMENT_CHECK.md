# 海淀区 V1 API 资产与部署检查

检查时间：2026-07-01

## 可视化输出

- `haidian_v1_api_tiles_many_patches.png`：四类 API 任务的红白预测瓦片，每类选 12 个高响应 patch。
- `building_extraction_diagnostics_many_regions.png`：建筑物提取多 patch 诊断图。
- `road_extraction_diagnostics_many_regions.png`：道路提取多 patch 诊断图。
- `construction_many_regions.png`：施工地检测多 patch 效果条。
- `construction_joint_many_regions.png`：联合施工检测多 patch 效果条。

## 本地资产完整性

- patch 元数据：320 个 patch。
- embedding：6 个月，每月 320 份，共 1920 份。
- embedding 形状抽检：`(64, 128, 128)`，`float32`，未发现 NaN/Inf。
- 任务结果：
  - `building_extraction`：320 份预测、320 张瓦片、320 份标签。
  - `road_extraction`：320 份预测、320 张瓦片、320 份标签。
  - `construction`：320 份预测、320 张瓦片、62 份标签。
  - `construction_joint`：320 份预测、320 张瓦片、62 份标签。

施工类标签数量少于 320 是因为真实标签覆盖有限，不是上传漏文件。

## ModelScope 远端检查

已从 ModelScope 成功下载并读取以下小文件：

- `haidian/v1/manifest.json`
- `haidian/v1/README.md`
- `haidian/v1/api_ready/data/haidian/patches_meta_v1.json`
- `haidian/v1/api_ready/data/haidian/embeddings/v1/202512/patch_000000.json`
- `haidian/v1/api_ready/data/haidian/embeddings/v1/202512/patch_000000.png`
- 三类任务示例瓦片：建筑、道路、施工。

远端 checkpoint 文件树已确认包含：

- `haidian/v1/api_ready/models/haidian/v1/embedding/best.pt`
- `haidian/v1/api_ready/models/haidian/v1/embedding/config.yaml`
- `haidian/v1/api_ready/models/haidian/v1/task_heads/building_extraction/{linear,mlp,unet}/best.pt`
- `haidian/v1/api_ready/models/haidian/v1/task_heads/road_extraction/{linear,mlp,unet}/best.pt`
- `haidian/v1/api_ready/models/haidian/v1/task_heads/construction/unet/best.pt`
- `haidian/v1/api_ready/models/haidian/v1/task_heads/construction_joint/unet/best.pt`
- `haidian/v1/archive/training_output/best.pt`
- `haidian/v1/archive/training_output/epoch_80.pt`
- `haidian/v1/archive/training_output/epoch_90.pt`
- `haidian/v1/archive/training_output/epoch_100.pt`

## 部署链路修复

发现并修复了一个部署风险：原下载脚本会默认下载整个 ModelScope 数据集，可能连同 `archive` 一起下载，体积约 95G。

已在 `embedding-api` 中修复为默认只下载：

```text
haidian/v1/api_ready/**
```

该部分约 9.5G，是正常 API 部署所需内容。

修复提交：

```text
9090c93 fix: limit Haidian ModelScope download to API assets
```
