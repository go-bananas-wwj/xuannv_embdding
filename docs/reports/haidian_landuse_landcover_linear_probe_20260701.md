# 海淀区 Land Use / Land Cover 轻量下游测试

时间：2026-07-01

本次测试基于海淀区 P2A embedding 的 `202605` 月份结果，补充了两个此前未训练的下游任务：

- `land_cover`：使用 `processed/haidian/labels/worldcover` 作为土地覆盖标签。
- `land_use`：使用 OSM 弱标签融合生成土地利用类别，包括 residential、commercial、industrial、agriculture、green、water、construction、transport、other。

训练方式为轻量 linear probe：按 patch 做 80/20 划分，从训练 patch 中均衡采样像素，使用 sklearn `SGDClassifier(loss=log_loss)` 训练像素级线性分类器。

## 输出目录

数据盘结果目录：

```text
/data/xuannv_embedding/modelscope_upload/haidian/v1/reports/landuse_landcover_linear_probe_202605/
```

主要文件：

```text
haidian_land_cover_linear_probe_full_domain.png
haidian_land_cover_linear_probe_samples.png
haidian_land_use_linear_probe_full_domain.png
haidian_land_use_linear_probe_samples.png
summary.json
land_cover_linear_probe.joblib
land_use_linear_probe.joblib
archives/haidian_v1_landuse_landcover_linear_probe_202605.zip
```

## 快速结果

| 任务 | 标签来源 | Accuracy | Balanced Acc | Macro F1 | mIoU | 备注 |
|---|---|---:|---:|---:|---:|---|
| land_cover | WorldCover/ESRI LULC 重编码 | 0.8154 | 0.6853 | 0.4959 | 0.3928 | 建成区、裸地/其他、水体较好，稀有类较弱 |
| land_use | OSM 弱标签融合 | 0.4186 | 0.3990 | 0.3005 | 0.1947 | 标签较弱且类别重叠，只能作为语义可分性诊断 |

## 解读

`land_cover` 的结果说明当前 embedding 对粗粒度土地覆盖已经有一定线性可分性，尤其对 built-up、bare/other、water 等类别更明显。

`land_use` 的结果明显弱一些，主要原因是标签来自 OSM 弱监督，存在覆盖不全、类别重叠和语义不稳定问题。例如商业/居住/工业在影像纹理上相近，且 OSM 面状标签并不总是覆盖真实用地边界。

## 下一步建议

1. land_cover 可以继续做更正式的 MLP / UNet head，并上传到 ModelScope 作为海淀 V1 的补充任务。
2. land_use 需要先优化标签：把 OSM 弱标签转成三值标签，明确正样本、负样本和 ignore 区域。
3. 对 land_use 不建议马上用作强指标汇报，适合先作为“弱标签语义诊断”。
