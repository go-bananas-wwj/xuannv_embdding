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

## 审查后修复

子智能体审查发现了三个训练前必须处理的问题，已修复：

- STP 内部 `(B,T,H,W,C)` 到 `(B*T,C,H,W)` 的转换不能直接 `reshape`，否则会打乱空间和通道。现在已统一改为 `permute(...).reshape(...)`，并补了值级别回归测试。
- 高分 mask 现在会在模型内对齐到 embedding 网格，避免原生 mask 尺寸与 128x128 embedding 不一致。
- 高分不可用区域会把 `highres_feat` 置零，不再让无效高分内容进入融合。
- 6 卡 DDP 下 `TotalLoss` 中的 semantic probe 参数现在会随 criterion 一起 DDP 同步，避免各 rank 自己更新 probe。

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
  - 结果：54 passed
- `PYTHONPATH=/root/workspace/xuannv/src pytest -q tests/test_training.py::test_trainer_checkpoint_best_and_latest3 tests/test_training.py::test_trainer_load_restores_best_state tests/test_training.py::test_trainer_best_pt_when_save_and_eval_aligned tests/test_training.py::test_batch_uniformity_loss tests/test_training.py::test_batch_uniformity_loss_temporal`
  - 结果：5 passed
- P7A 真实 batch smoke：
  - 低分输入：`s2/s1/landsat = (3, 6, C, 128, 128)`
  - 高分输入：`highres_optical/highres_sar = (3, C, 427, 427)`
  - 输出 `embedding_map` shape：`(3, 6, 64, 128, 128)`
  - vMF 后像素范数均值：`1.0`
  - 高分 optical/SAR 重建头正常存在

备注：P7A 训练集和验证集 patch 有重叠，`val_loss` 只用于训练过程监控与 best checkpoint 选择，不作为严格泛化指标。最终效果仍以下游 5-fold 测评为准。

## 下一步

P7A 代码层面已经可以进入训练前数据检查。训练前建议先可视化 1-2 个 patch 的 6 个月输入，确认高分、S2、Landsat、S1、OSM landcover 和 pixel mask 没问题，再启动 6 卡训练。
