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

### Changed

- 完成最终审查修复波 `86095fb`、`966737c`、`f36a4de`：bootstrap 与日志写入改为
  symlink-safe/no-clobber；模型输入、straight-through gate gradients 与 vMF 单位范数
  合同收紧；prepare/CPU/NPU/finalizer evidence graph 改为跨 artifact 语义复验。
- 完成非 NPU 最终安全收口 `cbda7be`：仅在精确 schema/policy 与 current evidence digest
  全部匹配时恢复 replace 中断留下的 `SUCCESS.tmp`，恶意或残缺临时文件 fail closed；
  preliminary/final audit 必须具有精确三/四阶段历史；safe tee 对日志 fd 与 stdout 均执行
  write-all，拒绝 0、负数和越界写入。现有 `f36a4de` 物理运行经收紧 verifier 复验通过，
  无需也未执行 NPU rerun。
- 关闭 Critical TOCTOU 复审发现 `13efe98`：finalizer 通过 sandbox directory fd 与
  `O_EXCL/O_NOFOLLOW` 固定 `SUCCESS.tmp`，从 stable fd 验证单链接常规 inode 和内容，
  再以 hard-link no-replace 发布并核对目标 inode；新增 symlink-before-open、
  temporary-path swap-before-publish、concurrent SUCCESS winner 三类竞态回归。该后续
  hardening 未运行 NPU，物理运行仍绑定 `f36a4de`。
- 最终 physical-NPU-2 synthetic smoke seal 绑定运行 commit `f36a4de`，四组输出均为
  `[4,8,64,128,128]`，最终 combined SHA-256 为
  `45ec55838b74b09d48be96bf510e1e35c8368ff996ea6953f86414f0ba1944f9`。
- 登记两份仅供诊断的 rejected attempts：`rejected_20260815_first_seal` 与
  `rejected_pre_f36a4de`。后者已从沙箱同级位置纠正到固定沙箱内；当前 350 条 audit
  entries 不含 `attempts/` 路径，两份归档均不得作为成功、精度或训练证据。
- 明确延期 192 个 grid headers/全量 statistics、archive/header 共享 worker pool 与源
  ZIP 内容哈希；这些缺口及 synthetic AEF/high-resolution 输入使本次结果不能外推为
  真实 AEF、真实 2 m、正式训练或正式评测结论。

## 2026-06-21

### Added
- 新增 `downstreams/` 下游任务评测框架。
- 支持海淀建筑工地像素级分割（5-fold CV、10/25/50/100% 标签比例）。
- 支持 embedding 离线预生成、元数据溯源与 random-init 基线。
- 提供 linear / FCN / UNet / UperNet 任务头及 Focal + Dice 默认损失。
- 评测指标：mIoU、F1、AP、AUPRC；支持 PR 曲线与预测叠加图可视化。
