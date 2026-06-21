# Changelog

## 2026-06-21

### Added
- 新增 `downstreams/` 下游任务评测框架。
- 支持海淀建筑工地像素级分割（5-fold CV、10/25/50/100% 标签比例）。
- 支持 embedding 离线预生成、元数据溯源与 random-init 基线。
- 提供 linear / FCN / UNet / UperNet 任务头及 Focal + Dice 默认损失。
- 评测指标：mIoU、F1、AP、AUPRC；支持 PR 曲线与预测叠加图可视化。
- 新增端到端分割训练：`EndToEndSegmentationDataset`、`EndToEndModel` 与 `construction_end_to_end.yaml`。
- `train_task.py` 支持 `end_to_end` 模式，可联合训练预训练 AEF encoder 与下游分割头，支持 backbone/head 分层学习率与前 `freeze_backbone_epochs` 冻结。
