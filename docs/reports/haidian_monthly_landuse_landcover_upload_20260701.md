# 海淀区月度 Land Use / Land Cover 散图归档上传说明

时间：2026-07-01

已将海淀区 V1 的 land use / land cover 月度预测散图按月份打包上传到 ModelScope。

## ModelScope 路径

```text
haidian/v1/reports/monthly_landuse_landcover_linear_probe/
```

## 文件结构

```text
archives/
  haidian_v1_202512_landuse_landcover_linear_probe.zip
  haidian_v1_202601_landuse_landcover_linear_probe.zip
  haidian_v1_202602_landuse_landcover_linear_probe.zip
  haidian_v1_202603_landuse_landcover_linear_probe.zip
  haidian_v1_202604_landuse_landcover_linear_probe.zip
  haidian_v1_202605_landuse_landcover_linear_probe.zip
  checksums.sha256
previews/
  haidian_202512_land_cover_classification_full_domain.png
  haidian_202512_land_use_classification_full_domain.png
  ...
README.md
manifest.json
```

每个月 zip 包内包含：

```text
{month}/land_cover_classification/haidian_{month}_land_cover_classification_patch_xxxxxx_pred.png
{month}/land_use_classification/haidian_{month}_land_use_classification_patch_xxxxxx_pred.png
{month}/haidian_{month}_land_cover_classification_full_domain.png
{month}/haidian_{month}_land_use_classification_full_domain.png
```

## 模型说明

- 使用 `202605` 月份训练得到的 sklearn linear probe head。
- 对 `202512` 到 `202605` 六个月 P2A embedding 分别推理。
- `land_cover_classification` 使用 WorldCover/ESRI LULC 重编码标签训练。
- `land_use_classification` 使用 OSM 弱标签融合训练，属于弱监督诊断结果。

## 注意事项

这批结果适合用于前端展示和汇报补充，但 `land_use_classification` 不是人工强标注结果，正式指标解读时需说明其弱监督来源。
