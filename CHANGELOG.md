# Changelog

## 2026-08-15

### Added

- 新增 China V1 隔离三路融合工程冒烟：以 4 个真实 2020—2021 S1/S2 patch 验证
  Base、Base+synthetic AEF、Base+synthetic 2 m 和 Full 的单 NPU 前向、受控反向、
  checkpoint 恢复、确定性复算与四组 FP16 Zarr 封存。
- 新增严格的 synthetic/formal-use、路径、设备映射、源快照、foreground tee、path-audit
  与 `SUCCESS` digest closure 防护；corrected current seal 的 fresh 验证通过。
- 新增工程证据报告 `docs/reports/china_v1_isolated_fusion_smoke_20260815.md`。结论仅为
  engineering PASS；模型精度、真实 AEF 语义、真实 2 m 质量与全国生产能力均未评估。

## 2026-06-21

### Added
- 新增 `downstreams/` 下游任务评测框架。
- 支持海淀建筑工地像素级分割（5-fold CV、10/25/50/100% 标签比例）。
- 支持 embedding 离线预生成、元数据溯源与 random-init 基线。
- 提供 linear / FCN / UNet / UperNet 任务头及 Focal + Dice 默认损失。
- 评测指标：mIoU、F1、AP、AUPRC；支持 PR 曲线与预测叠加图可视化。
