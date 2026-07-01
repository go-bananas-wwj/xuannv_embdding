# 海淀区施工工地 GT patch 标签上传说明

时间：2026-07-01

已将海淀区施工工地真实标签逐 patch PNG 上传到 ModelScope。

## 标签来源

```text
/data/xuannv_embedding/processed/haidian/labels/construction/masks
```

## ModelScope 路径

```text
haidian/v1/reports/haidian_construction_gt_patch_labels_20260701/
```

## 上传内容

```text
README.md
manifest.json
haidian_construction_gt_320_patch_full_domain.png
archives/haidian_construction_gt_patch_labels_20260701.zip
archives/checksums.sha256
```

zip 包内包含 320 个 patch 的 GT PNG：

```text
labels/haidian_construction_gt_patch_000000.png
...
```

## 说明

- 总 patch 数：320。
- 有施工工地真实标签 patch：62。
- 无标注 patch：258。
- 有标签区域用红色表示。
- 无标注 patch 输出为纯白 PNG。
- 该目录只包含 GT 标签，不包含模型预测结果。
