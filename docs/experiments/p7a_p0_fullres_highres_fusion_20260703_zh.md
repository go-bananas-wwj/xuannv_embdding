# P7A P0 修复说明

日期：2026-07-03

## 这次先修的 P0

1. 真正 128x128 precision path

原 P6A 虽然导出 `128x128` embedding，但 STP 精度路径先降到 `64x64`，再上采样回 `128x128`。P7A 新增：

```yaml
model:
  stp:
    precision_scale: 1
```

含义是 STP precision path 保持输入网格大小，不再先变成 `H//2, W//2`。

2. 高分辨率影像注入 embedding

原 P6A 配置里：

```yaml
highres_fusion_to_embedding: false
```

P7A 改为：

```yaml
highres_fusion_to_embedding: true
```

同时高分 encoder 从“多次 stride=2 下采样 + 平均池化”改成“先对齐到 embedding 网格，再用 stride=1 卷积提取纹理和边界”。这样更适合保留水体小岛、岸线、道路边界、建筑边界。

## AEF 组件现状

| 组件 | 当前代码是否有 | 说明 |
| --- | --- | --- |
| vMF bottleneck | 有 | `VMFBottleneck` 会把每个像素 embedding 归一化到单位球面，训练时加噪声近似 vMF 采样。 |
| Batch uniformity | 有 | `batch_uniformity_loss` 已在训练 loss 中，但 P6A 权重只有 `0.005`，有效秩统计说明还不够。P7A 先调到 `0.02`。 |
| Teacher-student | 没有完整路径 | 当前没有 AEF 论文里的 teacher video model / student video model / text alignment 三模型联合训练。 |
| Text alignment | 没有 | 当前只用 OSM 弱语义和重建目标，没有 Wikipedia/GBIF 文本对齐。 |

## 新配置

`configs/v2_p7a_haidian_202512_202605_fullres_highres_fusion_20260703.yaml`

相对 P6A 只做关键变化：

- `precision_scale: 1`
- `highres_fusion_to_embedding: true`
- `uniformity_weight: 0.02`
- 实验名和输出目录改为 P7A

其余数据、月份、OSM landcover、下游弱语义任务保持一致，方便与 P6A 严格对比。

## 验证

- `PYTHONPATH=/root/workspace/xuannv/src pytest -q tests/test_model.py tests/test_smoke.py`
  - 结果：51 passed
- P7A 配置 smoke：
  - 输出 `embedding_map` shape：`(1, 6, 64, 16, 16)`
  - vMF 后像素范数均值：`1.0`
  - 高分 optical/SAR 重建头正常存在

## 下一步

P7A 代码层面已经可以进入训练前数据检查。训练前建议先可视化 1-2 个 patch 的 6 个月输入，确认高分、S2、Landsat、S1、OSM landcover 和 pixel mask 没问题，再启动 6 卡训练。

